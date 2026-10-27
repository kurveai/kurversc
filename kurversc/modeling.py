"""Stable estimator adapters for CatBoost and optional foundation models."""

from __future__ import annotations

import copy
import json
from time import perf_counter
from typing import Any, Mapping

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, mean_absolute_error, roc_auc_score


def _is_classification(task: str) -> bool:
    return task in {"classification", "multiclass_classification"}


class FrameEnsemble:
    """Average independent estimators trained on separate temporal frames."""

    def __init__(
        self,
        models: list[Any] | tuple[Any, ...],
        *,
        task: str,
        target_classes: tuple[Any, ...] = (),
    ) -> None:
        if not models:
            raise ValueError("FrameEnsemble requires at least one estimator")
        self.models = tuple(models)
        self.task = task
        self.target_classes = target_classes

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        if not _is_classification(self.task):
            raise AttributeError(
                "Regression frame ensembles do not predict probabilities"
            )
        probabilities = [model.predict_proba(features) for model in self.models]
        if self.task != "multiclass_classification" or not self.target_classes:
            return np.mean(probabilities, axis=0)
        aligned = []
        for model, values in zip(self.models, probabilities, strict=True):
            native_classes = np.asarray(getattr(model, "classes_", ()))
            if len(native_classes) != values.shape[1]:
                raise ValueError("Estimator probabilities do not match its classes")
            output = np.zeros((len(values), len(self.target_classes)), dtype=float)
            for source, label in enumerate(native_classes):
                try:
                    destination = int(label)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        "Multiclass estimator classes must be encoded"
                    ) from exc
                if not 0 <= destination < len(self.target_classes):
                    raise ValueError(
                        "Multiclass estimator class is outside the target set"
                    )
                output[:, destination] = values[:, source]
            aligned.append(output)
        return np.mean(aligned, axis=0)

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        if _is_classification(self.task):
            probabilities = self.predict_proba(features)
            if self.task == "classification":
                return (probabilities[:, 1] >= 0.5).astype("int8")
            return np.argmax(probabilities, axis=1)
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

    # Normalize one column at a time. Reindexing and copying a complete wide
    # matrix first can retain several full numeric buffers during replacement.
    # The resulting columns have the same values/dtypes and caller index.
    prepared = {}
    for column in columns:
        values = frame[column] if column in frame else pd.Series(np.nan, index=frame.index)
        values = _normalize_unhashable(values)
        if column in datetime_columns:
            prepared[column] = (
                pd.to_datetime(values).astype("int64") / 1_000_000_000
            )
        elif column in categorical_columns:
            prepared[column] = values.astype("string").fillna("__missing__")
        else:
            prepared[column] = pd.to_numeric(values, errors="coerce").replace(
                [np.inf, -np.inf], np.nan
            )
    inputs = pd.DataFrame(prepared, index=frame.index, copy=False)
    inputs.attrs = copy.deepcopy(frame.attrs)
    if "__kurversc_label_available_at" in frame:
        # Routing metadata, not a predictor. Tuple storage avoids ambiguous
        # numpy-array comparisons when pandas propagates DataFrame.attrs.
        inputs.attrs["kurversc_label_available_at"] = tuple(pd.to_datetime(frame["__kurversc_label_available_at"]))
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
    if _is_classification(task):
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


def _tabpfn_estimator(
    *,
    task: str,
    categorical: list[int],
    random_state: int,
    model_params: Mapping[str, Any] | None,
    version_attribute: str,
) -> Any:
    try:
        from tabpfn import TabPFNClassifier, TabPFNRegressor
        from tabpfn.constants import ModelVersion
    except ImportError as exc:  # pragma: no cover - actionable dependency error
        raise ImportError(
            "TabPFN support requires `pip install 'kurversc[tabpfn]'`"
        ) from exc
    if not hasattr(ModelVersion, version_attribute):
        display_version = version_attribute.removeprefix("V").replace("_", ".")
        raise ImportError(
            f"TabPFN {display_version} support requires a tabpfn release that "
            f"exposes ModelVersion.{version_attribute}"
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
    version = getattr(ModelVersion, version_attribute)
    return estimator_type.create_default_for_version(version, **params)


def _fit_tabpfn(
    train_x: pd.DataFrame,
    train_y: pd.Series,
    validation_x: pd.DataFrame,
    validation_y: pd.Series,
    *,
    task: str,
    categorical: list[int],
    random_state: int,
    model_params: Mapping[str, Any] | None,
    version_attribute: str,
) -> tuple[Any, str, float, float]:
    """Fit and validate a local TabPFN sklearn-compatible estimator."""

    model = _tabpfn_estimator(
        task=task,
        categorical=categorical,
        random_state=random_state,
        model_params=model_params,
        version_attribute=version_attribute,
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
    elif task == "multiclass_classification":
        raise ValueError("TabPFN v3 multiclass support is not implemented")
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
    """Fit and validate the local TabPFN v3 estimator."""

    return _fit_tabpfn(
        train_x,
        train_y,
        validation_x,
        validation_y,
        task=task,
        categorical=categorical,
        random_state=random_state,
        model_params=model_params,
        version_attribute="V3",
    )


def fit_tabpfn_v3_5(
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
    """Fit and validate the local TabPFN v3.5 estimator."""

    return _fit_tabpfn(
        train_x,
        train_y,
        validation_x,
        validation_y,
        task=task,
        categorical=categorical,
        random_state=random_state,
        model_params=model_params,
        version_attribute="V3_5",
    )


def _fit_final_tabpfn(
    features: pd.DataFrame,
    labels: pd.Series,
    *,
    task: str,
    categorical: list[int],
    random_state: int,
    model_params: Mapping[str, Any] | None,
    version_attribute: str,
) -> tuple[Any, float, tuple[Any, ...]]:
    """Fit a production TabPFN estimator without a holdout set."""

    model = _tabpfn_estimator(
        task=task,
        categorical=categorical,
        random_state=random_state,
        model_params=model_params,
        version_attribute=version_attribute,
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
    elif task == "multiclass_classification":
        raise ValueError("TabPFN v3 multiclass support is not implemented")
    else:
        classes = []
        model.fit(features, pd.to_numeric(labels, errors="raise").astype("float64"))
    return model, perf_counter() - started, tuple(classes)


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

    return _fit_final_tabpfn(
        features,
        labels,
        task=task,
        categorical=categorical,
        random_state=random_state,
        model_params=model_params,
        version_attribute="V3",
    )


def fit_final_tabpfn_v3_5(
    features: pd.DataFrame,
    labels: pd.Series,
    *,
    task: str,
    categorical: list[int],
    random_state: int,
    model_params: Mapping[str, Any] | None,
) -> tuple[Any, float, tuple[Any, ...]]:
    """Fit a production TabPFN v3.5 estimator without a holdout set."""

    return _fit_final_tabpfn(
        features,
        labels,
        task=task,
        categorical=categorical,
        random_state=random_state,
        model_params=model_params,
        version_attribute="V3_5",
    )


def _tabicl_estimator(
    *,
    task: str,
    random_state: int,
    model_params: Mapping[str, Any] | None,
) -> Any:
    try:
        from tabicl import TabICLClassifier, TabICLRegressor
    except ImportError as exc:  # pragma: no cover - actionable dependency error
        raise ImportError(
            "TabICL support requires `pip install 'kurversc[tabicl]'`"
        ) from exc
    params: dict[str, Any] = {
        "n_jobs": 8,
        "random_state": random_state,
    }
    params.update(model_params or {})
    estimator_type = TabICLClassifier if task == "classification" else TabICLRegressor
    return estimator_type(**params)


def fit_tabicl(
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
    """Fit and validate the local TabICLv2 sklearn-compatible estimator."""

    # TabICL detects categorical pandas columns itself; ``categorical`` remains
    # part of the common backend contract used by CatBoost and TabPFN.
    del categorical
    model = _tabicl_estimator(
        task=task,
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
    elif task == "multiclass_classification":
        raise ValueError("TabICL multiclass support is not implemented")
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


def fit_final_tabicl(
    features: pd.DataFrame,
    labels: pd.Series,
    *,
    task: str,
    categorical: list[int],
    random_state: int,
    model_params: Mapping[str, Any] | None,
) -> tuple[Any, float, tuple[Any, ...]]:
    """Fit a production TabICLv2 estimator without a holdout set."""

    del categorical
    model = _tabicl_estimator(
        task=task,
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


def _tabfm_estimator(
    *,
    task: str,
    random_state: int,
    model_params: Mapping[str, Any] | None,
) -> Any:
    """Load frozen TabFM and wrap it in its sklearn-compatible ICL interface."""

    if task == "multiclass_classification":
        raise ValueError("TabFM multiclass support is not implemented")
    try:
        from tabfm import TabFMClassifier, TabFMRegressor, tabfm_v1_0_0_pytorch
    except ImportError as exc:  # pragma: no cover - actionable dependency error
        raise ImportError(
            "TabFM support requires `pip install 'kurversc[tabfm]'`"
        ) from exc

    model_type = "classification" if task == "classification" else "regression"
    # ``load`` returns published frozen weights. TabFM's estimator ``fit``
    # constructs its in-context examples and preprocessing state; it does not
    # update these weights or run task-specific optimization.
    foundation_model = tabfm_v1_0_0_pytorch.load(
        model_type=model_type,
        device="cpu",
        dtype=None,
    )
    params: dict[str, Any] = {
        "model": foundation_model,
        "n_estimators": 32,
        "max_num_features": 500,
        "max_num_rows": 100,
        "random_state": random_state,
        "use_amp": False,
        "batch_size": 1,
        "enable_nnls": False,
    }
    params.update(model_params or {})
    estimator_type = TabFMClassifier if task == "classification" else TabFMRegressor
    return estimator_type(**params)


def fit_tabfm(
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
    """Score frozen TabFM over one GraphReduce feature frame."""

    del categorical
    model = _tabfm_estimator(
        task=task,
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


def fit_final_tabfm(
    features: pd.DataFrame,
    labels: pd.Series,
    *,
    task: str,
    categorical: list[int],
    random_state: int,
    model_params: Mapping[str, Any] | None,
) -> tuple[Any, float, tuple[Any, ...]]:
    """Prepare one frozen TabFM context for a production GraphReduce frame."""

    del categorical
    model = _tabfm_estimator(
        task=task,
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
        model.fit(
            features,
            labels.map({value: index for index, value in enumerate(classes)}).astype(
                "int8"
            ),
        )
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
    elif task == "multiclass_classification":
        classes = _binary_classes(pd.concat([train_y, validation_y], ignore_index=True))
        if len(classes) < 2:
            raise ValueError("multiclass classification requires at least two classes")
        class_map = {value: index for index, value in enumerate(classes)}
        encoded_train = train_y.map(class_map).astype("int32")
        encoded_validation = validation_y.map(class_map).astype("int32")
        model = CatBoostClassifier(
            loss_function="MultiClass", eval_metric="Accuracy", **params
        )
        if "eval_set" in fit_kwargs:
            fit_kwargs["eval_set"] = (validation_x, encoded_validation)
        model.fit(train_x, encoded_train, **fit_kwargs)
        score = float(
            accuracy_score(
                encoded_validation, np.asarray(model.predict(validation_x)).reshape(-1)
            )
        )
        metric = "accuracy"
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
    elif task == "multiclass_classification":
        classes = (
            list(target_classes)
            if target_classes is not None
            else _binary_classes(labels)
        )
        if len(classes) < 2:
            raise ValueError("multiclass classification requires at least two classes")
        encoded = labels.map(
            {value: index for index, value in enumerate(classes)}
        ).astype("int32")
        model = CatBoostClassifier(
            loss_function="MultiClass", eval_metric="Accuracy", **params
        )
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
    if task == "multiclass_classification":
        if len(target_classes) < 2:
            raise ValueError("multiclass scoring requires at least two target classes")
        encoded = labels.map(
            {value: index for index, value in enumerate(target_classes)}
        )
        if encoded.isna().any():
            raise ValueError("validation labels contain a class absent from training")
        probabilities = np.asarray(model.predict_proba(features), dtype=float)
        predictions = np.argmax(probabilities, axis=1)
        return "accuracy", float(accuracy_score(encoded, predictions)), probabilities
    numeric = pd.to_numeric(labels, errors="raise").astype("float64")
    predictions = np.asarray(model.predict(features), dtype=float)
    return "mae", float(mean_absolute_error(numeric, predictions)), predictions
