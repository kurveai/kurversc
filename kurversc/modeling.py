"""A deliberately small, stable CatBoost validation runner."""

from __future__ import annotations

import json
from time import perf_counter
from typing import Any, Mapping

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, roc_auc_score


DEFAULT_CATBOOST_TUNING_CONFIGS: tuple[dict[str, Any], ...] = (
    {
        "name": "depth6_regularized",
        "iterations": 2000,
        "depth": 6,
        "learning_rate": 0.03,
        "l2_leaf_reg": 10.0,
        "early_stopping_rounds": 200,
    },
    {
        "name": "depth7_regularized",
        "iterations": 1500,
        "depth": 7,
        "learning_rate": 0.03,
        "l2_leaf_reg": 10.0,
        "early_stopping_rounds": 200,
    },
    {
        "name": "depth5_regularized",
        "iterations": 2500,
        "depth": 5,
        "learning_rate": 0.02,
        "l2_leaf_reg": 12.0,
        "early_stopping_rounds": 200,
    },
    {
        "name": "depth6_fast",
        "iterations": 1000,
        "depth": 6,
        "learning_rate": 0.05,
        "l2_leaf_reg": 12.0,
        "early_stopping_rounds": 200,
    },
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {
            str(key): _jsonable(item)
            for key, item in sorted(value.items(), key=lambda pair: repr(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted((_jsonable(item) for item in value), key=repr)
    return value


def _normalize_unhashable(series: pd.Series) -> pd.Series:
    """Encode nested relational cells deterministically for tabular modeling."""

    if not pd.api.types.is_object_dtype(series.dtype):
        return series

    def normalize(value: Any) -> Any:
        try:
            hash(value)
            return value
        except TypeError:
            try:
                return json.dumps(
                    _jsonable(value),
                    sort_keys=True,
                    separators=(",", ":"),
                    default=str,
                )
            except (TypeError, ValueError):
                return repr(value)

    return series.map(normalize)


def _binary_classes(labels: pd.Series) -> list[Any]:
    values = list(pd.unique(labels))
    return sorted(values, key=lambda value: (type(value).__name__, repr(value)))


def prepare_features(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    *,
    target: str,
    excluded: set[str],
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, list[int]]:
    train = train.copy()
    validation = validation.copy()
    for column in train.columns:
        train[column] = _normalize_unhashable(train[column])
        if column in validation:
            validation[column] = _normalize_unhashable(validation[column])
    candidates = [
        column
        for column in train.columns
        if column != target and column not in excluded
    ]
    candidates = [
        column
        for column in candidates
        # Selection is learned from training only. Validation may be used to
        # score values, never to decide whether a feature exists in the model.
        if not train[column].isna().all() and train[column].nunique(dropna=False) > 1
    ]
    if not candidates:
        raise ValueError("GraphReduce produced no usable feature columns")
    missing_validation = sorted(set(candidates) - set(validation.columns))
    if missing_validation:
        raise ValueError(
            "Frozen GraphReduce plan did not reproduce training features on "
            f"validation: {missing_validation[:10]}"
        )
    train_x = train[candidates].copy()
    validation_x = validation[candidates].copy()
    categorical: list[int] = []
    for index, column in enumerate(candidates):
        left = train_x[column]
        right = validation_x[column]
        if pd.api.types.is_datetime64_any_dtype(left.dtype):
            train_x[column] = left.astype("int64") / 1_000_000_000
            validation_x[column] = pd.to_datetime(right).astype("int64") / 1_000_000_000
        elif (
            isinstance(left.dtype, pd.CategoricalDtype)
            or pd.api.types.is_object_dtype(left.dtype)
            or pd.api.types.is_string_dtype(left.dtype)
            or pd.api.types.is_bool_dtype(left.dtype)
        ):
            train_x[column] = left.astype("string").fillna("__missing__")
            validation_x[column] = right.astype("string").fillna("__missing__")
            categorical.append(index)
        else:
            train_x[column] = pd.to_numeric(left, errors="coerce").replace(
                [np.inf, -np.inf], np.nan
            )
            validation_x[column] = pd.to_numeric(right, errors="coerce").replace(
                [np.inf, -np.inf], np.nan
            )
    return train_x, train[target], validation_x, validation[target], categorical


def prepare_prediction_features(
    frame: pd.DataFrame,
    *,
    columns: list[str],
    categorical_columns: set[str],
    datetime_columns: set[str],
) -> pd.DataFrame:
    """Apply the inference-time half of :func:`prepare_features`."""

    inputs = frame.reindex(columns=columns).copy()
    for column in columns:
        inputs[column] = _normalize_unhashable(inputs[column])
        if column in datetime_columns:
            inputs[column] = (
                pd.to_datetime(inputs[column]).astype("int64") / 1_000_000_000
            )
        elif column in categorical_columns:
            inputs[column] = inputs[column].astype("string").fillna("__missing__")
        else:
            inputs[column] = pd.to_numeric(inputs[column], errors="coerce").replace(
                [np.inf, -np.inf], np.nan
            )
    return inputs


def fit_catboost(
    train_x: pd.DataFrame,
    train_y: pd.Series,
    validation_x: pd.DataFrame,
    validation_y: pd.Series,
    *,
    task: str,
    categorical: list[int],
    random_state: int,
    model_params: Mapping[str, Any] | None,
) -> tuple[Any, str, float, float]:
    try:
        from catboost import CatBoostClassifier, CatBoostRegressor
    except ImportError as exc:  # pragma: no cover - dependency error is actionable
        raise ImportError("kurversc.fit requires CatBoost; install kurversc") from exc

    params: dict[str, Any] = {
        "iterations": 300,
        "depth": 6,
        "learning_rate": 0.05,
        "l2_leaf_reg": 10.0,
        "random_strength": 1.0,
        "bagging_temperature": 1.0,
        "random_seed": random_state,
        "verbose": False,
        "allow_writing_files": False,
    }
    params.update(model_params or {})
    params.pop("name", None)
    early_stopping_rounds = int(
        params.pop(
            "early_stopping_rounds",
            min(50, int(params["iterations"]) // 2 or 1),
        )
    )
    if early_stopping_rounds < 1:
        raise ValueError("early_stopping_rounds must be positive")
    started = perf_counter()
    fit_kwargs: dict[str, Any] = {"cat_features": categorical}
    if not validation_x.empty:
        fit_kwargs.update(
            eval_set=(validation_x, validation_y),
            use_best_model=True,
            early_stopping_rounds=early_stopping_rounds,
        )
    if task == "classification":
        classes = _binary_classes(pd.concat([train_y, validation_y], ignore_index=True))
        if len(classes) != 2:
            raise ValueError(
                f"classification requires exactly two target classes; got {len(classes)}"
            )
        class_map = {value: index for index, value in enumerate(classes)}
        encoded_train = train_y.map(class_map).astype("int8")
        encoded_validation = validation_y.map(class_map).astype("int8")
        if encoded_validation.nunique() != 2:
            raise ValueError("validation labels must contain both classes for ROC AUC")
        model = CatBoostClassifier(loss_function="Logloss", eval_metric="AUC", **params)
        if "eval_set" in fit_kwargs:
            fit_kwargs["eval_set"] = (validation_x, encoded_validation)
        model.fit(train_x, encoded_train, **fit_kwargs)
        score = float(
            roc_auc_score(encoded_validation, model.predict_proba(validation_x)[:, 1])
        )
        metric = "roc_auc"
    else:
        numeric_train = pd.to_numeric(train_y, errors="raise").astype("float64")
        numeric_validation = pd.to_numeric(validation_y, errors="raise").astype(
            "float64"
        )
        model = CatBoostRegressor(loss_function="MAE", eval_metric="MAE", **params)
        if "eval_set" in fit_kwargs:
            fit_kwargs["eval_set"] = (validation_x, numeric_validation)
        model.fit(train_x, numeric_train, **fit_kwargs)
        score = float(
            mean_absolute_error(numeric_validation, model.predict(validation_x))
        )
        metric = "mae"
    return model, metric, score, perf_counter() - started


def tune_catboost(
    train_x: pd.DataFrame,
    train_y: pd.Series,
    validation_x: pd.DataFrame,
    validation_y: pd.Series,
    *,
    task: str,
    categorical: list[int],
    random_state: int,
    model_params: Mapping[str, Any] | None,
    tuning_configs: tuple[Mapping[str, Any], ...] | None = None,
) -> tuple[Any, str, float, float, dict[str, Any], tuple[dict[str, Any], ...]]:
    """Tune CatBoost once on the winning full-data GraphReduce feature plan.

    GraphReduce trials intentionally retain the inexpensive single estimator
    in :func:`fit_catboost`. Explicit ``model_params`` preserve the original
    single-configuration behavior unless ``tuning_configs`` are also supplied.
    """

    if tuning_configs is None:
        candidates: tuple[Mapping[str, Any], ...] = (
            (dict(model_params),)
            if model_params is not None
            else DEFAULT_CATBOOST_TUNING_CONFIGS
        )
        global_overrides: Mapping[str, Any] = {}
    else:
        candidates = tuple(tuning_configs)
        if not candidates:
            raise ValueError("model_tuning_configs must not be empty")
        global_overrides = model_params or {}

    best_model: Any | None = None
    best_metric = ""
    best_score = float("-inf") if task == "classification" else float("inf")
    best_params: dict[str, Any] | None = None
    total_seconds = 0.0
    records: list[dict[str, Any]] = []
    for number, candidate in enumerate(candidates, start=1):
        raw = dict(candidate)
        name = str(raw.pop("name", f"candidate_{number}"))
        raw.update(global_overrides)
        model, metric, score, seconds = fit_catboost(
            train_x,
            train_y,
            validation_x,
            validation_y,
            task=task,
            categorical=categorical,
            random_state=random_state,
            model_params=raw,
        )
        total_seconds += seconds
        tree_count = max(
            1, int(getattr(model, "tree_count_", raw.get("iterations", 300)))
        )
        records.append(
            {
                "name": name,
                "validation_score": score,
                "tree_count": tree_count,
                "model_seconds": seconds,
                "params": dict(raw),
            }
        )
        improved = score > best_score if metric == "roc_auc" else score < best_score
        if improved:
            best_model = model
            best_metric = metric
            best_score = score
            best_params = dict(raw)
            best_params["iterations"] = tree_count
            best_params.pop("early_stopping_rounds", None)

    if best_model is None or best_params is None:
        raise RuntimeError("CatBoost tuning produced no model")
    return (
        best_model,
        best_metric,
        best_score,
        total_seconds,
        best_params,
        tuple(records),
    )


def fit_final_catboost(
    features: pd.DataFrame,
    labels: pd.Series,
    *,
    task: str,
    categorical: list[int],
    random_state: int,
    model_params: Mapping[str, Any] | None,
) -> tuple[Any, float, tuple[Any, ...]]:
    """Fit the production estimator without consulting a holdout set."""

    try:
        from catboost import CatBoostClassifier, CatBoostRegressor
    except ImportError as exc:  # pragma: no cover - dependency error is actionable
        raise ImportError("kurversc.fit requires CatBoost; install kurversc") from exc

    params: dict[str, Any] = {
        "iterations": 300,
        "depth": 6,
        "learning_rate": 0.05,
        "l2_leaf_reg": 10.0,
        "random_strength": 1.0,
        "bagging_temperature": 1.0,
        "random_seed": random_state,
        "verbose": False,
        "allow_writing_files": False,
    }
    params.update(model_params or {})
    params.pop("name", None)
    params.pop("early_stopping_rounds", None)
    started = perf_counter()
    if task == "classification":
        classes = _binary_classes(labels)
        if len(classes) != 2:
            raise ValueError(
                f"classification requires exactly two target classes; got {len(classes)}"
            )
        encoded = labels.map(
            {value: index for index, value in enumerate(classes)}
        ).astype("int8")
        model = CatBoostClassifier(loss_function="Logloss", eval_metric="AUC", **params)
        model.fit(features, encoded, cat_features=categorical)
    else:
        classes = []
        numeric = pd.to_numeric(labels, errors="raise").astype("float64")
        model = CatBoostRegressor(loss_function="MAE", eval_metric="MAE", **params)
        model.fit(features, numeric, cat_features=categorical)
    return model, perf_counter() - started, tuple(classes)
