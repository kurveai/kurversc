"""Stable estimator adapters for CatBoost and optional TabPFN v3."""

from __future__ import annotations

import json
from time import perf_counter
from typing import Any, Mapping

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, roc_auc_score


class FrameEnsemble:
    """Average independent estimators trained on separate temporal frames."""

    def __init__(self, models: list[Any] | tuple[Any, ...], *, task: str) -> None:
        if not models:
            raise ValueError("FrameEnsemble requires at least one estimator")
        self.models = tuple(models)
        self.task = task

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        if self.task != "classification":
            raise AttributeError(
                "Regression frame ensembles do not predict probabilities"
            )
        return np.mean(
            [model.predict_proba(features) for model in self.models],
            axis=0,
        )

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        if self.task == "classification":
            return (self.predict_proba(features)[:, 1] >= 0.5).astype("int8")
        return np.mean([model.predict(features) for model in self.models], axis=0)


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
    (
        train_x,
        train_y,
        categorical,
        datetime_columns,
    ) = prepare_training_features(train, target=target, excluded=excluded)
    missing = set(train_x.columns) - set(validation.columns)
    if missing:
        raise ValueError(
            f"Validation is missing training feature columns: {sorted(missing)}"
        )
    validation_x = prepare_prediction_features(
        validation,
        columns=list(train_x.columns),
        categorical_columns={train_x.columns[index] for index in categorical},
        datetime_columns=set(datetime_columns),
    )
    return train_x, train_y, validation_x, validation[target], categorical


def prepare_training_features(
    frame: pd.DataFrame,
    *,
    target: str,
    excluded: set[str],
) -> tuple[pd.DataFrame, pd.Series, list[int], tuple[str, ...]]:
    """Prepare one training frame without requiring a resident holdout frame."""

    train = frame.copy()
    for column in train.columns:
        train[column] = _normalize_unhashable(train[column])
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
    train_x = train[candidates].copy()
    categorical: list[int] = []
    datetime_columns: list[str] = []
    for index, column in enumerate(candidates):
        left = train_x[column]
        if pd.api.types.is_datetime64_any_dtype(left.dtype):
            datetime_columns.append(column)
            train_x[column] = left.astype("int64") / 1_000_000_000
        elif (
            isinstance(left.dtype, pd.CategoricalDtype)
            or pd.api.types.is_object_dtype(left.dtype)
            or pd.api.types.is_string_dtype(left.dtype)
            or pd.api.types.is_bool_dtype(left.dtype)
        ):
            train_x[column] = left.astype("string").fillna("__missing__")
            categorical.append(index)
        else:
            train_x[column] = pd.to_numeric(left, errors="coerce").replace(
                [np.inf, -np.inf], np.nan
            )
    return train_x, train[target], categorical, tuple(datetime_columns)


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


def sample_training_rows(
    features: pd.DataFrame,
    labels: pd.Series,
    *,
    limit: int | None,
    task: str,
    random_state: int,
) -> tuple[pd.DataFrame, pd.Series]:
    """Return a deterministic estimator-only sample without changing features."""

    if limit is None or len(features) <= limit:
        return features, labels
    if limit < 2:
        raise ValueError("estimator_train_rows must be at least 2 or None")
    from sklearn.model_selection import train_test_split

    positions = np.arange(len(features))
    stratify = None
    if task == "classification":
        counts = labels.value_counts(dropna=False)
        if len(counts) > 1 and counts.min() >= 2 and limit >= len(counts):
            stratify = labels
    selected, _ = train_test_split(
        positions,
        train_size=limit,
        random_state=random_state,
        stratify=stratify,
    )
    selected = np.sort(selected)
    return (
        features.iloc[selected].reset_index(drop=True),
        labels.iloc[selected].reset_index(drop=True),
    )


def _tabpfn_v3_estimator(
    *,
    task: str,
    categorical: list[int],
    random_state: int,
    model_params: Mapping[str, Any] | None,
) -> Any:
    try:
        from tabpfn import TabPFNClassifier, TabPFNRegressor
        from tabpfn.constants import ModelVersion
    except ImportError as exc:  # pragma: no cover - actionable dependency error
        raise ImportError(
            "TabPFN v3 support requires `pip install 'kurversc[tabpfn]'`"
        ) from exc
    if not hasattr(ModelVersion, "V3"):
        raise ImportError(
            "TabPFN v3 support requires tabpfn>=8.5; the installed tabpfn "
            "does not expose ModelVersion.V3"
        )
    params: dict[str, Any] = {
        "n_estimators": 2,
        "device": "auto",
        "fit_mode": "low_memory",
        "memory_saving_mode": "auto",
        "random_state": random_state,
        "categorical_features_indices": categorical,
    }
    params.update(model_params or {})
    estimator_type = TabPFNClassifier if task == "classification" else TabPFNRegressor
    return estimator_type.create_default_for_version(ModelVersion.V3, **params)


def fit_tabpfn_v3(
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
    """Fit and validate the local TabPFN v3 sklearn-compatible estimator."""

    model = _tabpfn_v3_estimator(
        task=task,
        categorical=categorical,
        random_state=random_state,
        model_params=model_params,
    )
    started = perf_counter()
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
        model.fit(train_x, encoded_train)
        score = float(
            roc_auc_score(encoded_validation, model.predict_proba(validation_x)[:, 1])
        )
        metric = "roc_auc"
    else:
        numeric_train = pd.to_numeric(train_y, errors="raise").astype("float64")
        numeric_validation = pd.to_numeric(validation_y, errors="raise").astype(
            "float64"
        )
        model.fit(train_x, numeric_train)
        score = float(
            mean_absolute_error(numeric_validation, model.predict(validation_x))
        )
        metric = "mae"
    return model, metric, score, perf_counter() - started


def fit_final_tabpfn_v3(
    features: pd.DataFrame,
    labels: pd.Series,
    *,
    task: str,
    categorical: list[int],
    random_state: int,
    model_params: Mapping[str, Any] | None,
) -> tuple[Any, float, tuple[Any, ...]]:
    """Fit a production TabPFN v3 estimator without a holdout set."""

    model = _tabpfn_v3_estimator(
        task=task,
        categorical=categorical,
        random_state=random_state,
        model_params=model_params,
    )
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
        model.fit(features, encoded)
    else:
        classes = []
        model.fit(features, pd.to_numeric(labels, errors="raise").astype("float64"))
    return model, perf_counter() - started, tuple(classes)


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
        "random_seed": random_state,
        "verbose": False,
        "allow_writing_files": False,
        # Large RelArena hosts can expose hundreds of logical CPUs. Letting
        # every task use all of them creates hundreds of native worker stacks
        # and severe allocator retention when tasks run in parallel.
        "thread_count": 8,
    }
    params.update(model_params or {})
    started = perf_counter()
    fit_kwargs: dict[str, Any] = {"cat_features": categorical}
    if len(validation_x) >= 20:
        fit_kwargs.update(
            eval_set=(validation_x, validation_y),
            use_best_model=True,
            early_stopping_rounds=min(50, int(params["iterations"]) // 2 or 1),
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

    return fit_incremental_catboost_frame(
        features,
        labels,
        task=task,
        categorical=categorical,
        random_state=random_state,
        model_params=model_params,
    )


def fit_incremental_catboost_frame(
    features: pd.DataFrame,
    labels: pd.Series,
    *,
    task: str,
    categorical: list[int],
    random_state: int,
    model_params: Mapping[str, Any] | None,
    init_model: Any = None,
    target_classes: tuple[Any, ...] | None = None,
    iterations: int | None = None,
) -> tuple[Any, float, tuple[Any, ...]]:
    """Fit one CatBoost increment and optionally continue an existing model.

    CatBoost continuation appends trees to ``init_model``. The caller owns the
    total tree budget and feeds exactly one materialized cutoff frame per call.
    """

    try:
        from catboost import CatBoostClassifier, CatBoostRegressor
    except ImportError as exc:  # pragma: no cover - dependency error is actionable
        raise ImportError("kurversc.fit requires CatBoost; install kurversc") from exc

    params: dict[str, Any] = {
        "iterations": 300,
        "depth": 6,
        "learning_rate": 0.05,
        "l2_leaf_reg": 10.0,
        "random_seed": random_state,
        "verbose": False,
        "allow_writing_files": False,
        "thread_count": 8,
    }
    params.update(model_params or {})
    if iterations is not None:
        if iterations < 1:
            raise ValueError("incremental CatBoost iterations must be positive")
        params["iterations"] = int(iterations)
    started = perf_counter()
    fit_kwargs: dict[str, Any] = {"cat_features": categorical}
    if init_model is not None:
        fit_kwargs["init_model"] = init_model
    if task == "classification":
        classes = (
            list(target_classes)
            if target_classes is not None
            else _binary_classes(labels)
        )
        if len(classes) != 2:
            raise ValueError(
                f"classification requires exactly two target classes; got {len(classes)}"
            )
        encoded = labels.map(
            {value: index for index, value in enumerate(classes)}
        ).astype("int8")
        model = CatBoostClassifier(loss_function="Logloss", eval_metric="AUC", **params)
        model.fit(features, encoded, **fit_kwargs)
    else:
        classes = []
        numeric = pd.to_numeric(labels, errors="raise").astype("float64")
        model = CatBoostRegressor(loss_function="MAE", eval_metric="MAE", **params)
        model.fit(features, numeric, **fit_kwargs)
    return model, perf_counter() - started, tuple(classes)


def score_estimator(
    model: Any,
    features: pd.DataFrame,
    labels: pd.Series,
    *,
    task: str,
    target_classes: tuple[Any, ...] = (),
) -> tuple[str, float, np.ndarray]:
    """Score one validation frame and return its predictions for aggregation."""

    if task == "classification":
        if len(target_classes) != 2:
            raise ValueError("classification scoring requires two target classes")
        encoded = labels.map(
            {value: index for index, value in enumerate(target_classes)}
        )
        if encoded.isna().any():
            raise ValueError("validation labels contain a class absent from training")
        encoded = encoded.astype("int8")
        if encoded.nunique() != 2:
            raise ValueError("validation labels must contain both classes for ROC AUC")
        predictions = np.asarray(model.predict_proba(features)[:, 1], dtype=float)
        return "roc_auc", float(roc_auc_score(encoded, predictions)), predictions
    numeric = pd.to_numeric(labels, errors="raise").astype("float64")
    predictions = np.asarray(model.predict(features), dtype=float)
    return "mae", float(mean_absolute_error(numeric, predictions)), predictions
