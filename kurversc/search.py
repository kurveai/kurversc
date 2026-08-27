"""Configuration candidates and search results."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from math import isfinite
from typing import Any, Iterable, Sequence

import pandas as pd


DEFAULT_FAMILY_STAGES: tuple[tuple[str, ...], ...] = (
    ("base",),
    ("base", "temporal"),
    ("base", "temporal", "conditional"),
)


@dataclass(frozen=True)
class GraphConfig:
    feature_families: tuple[str, ...] = ("base",)
    depth: int = 1
    auto_annotate_features: bool = True
    feature_family_max_columns: int | None = None

    def __post_init__(self) -> None:
        if self.depth < 1:
            raise ValueError("depth must be at least 1")
        if not self.feature_families or self.feature_families[0] != "base":
            raise ValueError("feature_families must start with 'base'")

    @property
    def complexity(self) -> float:
        annotation_factor = 1.25 if self.auto_annotate_features else 1.0
        return float(self.depth * len(self.feature_families) * annotation_factor)

    def graphreduce_kwargs(self) -> dict[str, Any]:
        return {
            "auto_features": True,
            "auto_feature_hops_back": self.depth,
            "auto_feature_hops_front": 0,
        }


def incremental_configs(
    *,
    max_depth: int = 3,
    feature_family_stages: Sequence[Sequence[str]] = DEFAULT_FAMILY_STAGES,
    auto_annotate_options: Sequence[bool] = (True, False),
    feature_family_max_columns: int | None = None,
) -> tuple[GraphConfig, ...]:
    """Return deterministic candidates, beginning with the requested baseline."""

    if max_depth < 1:
        raise ValueError("max_depth must be at least 1")
    stages = [tuple(dict.fromkeys(stage)) for stage in feature_family_stages]
    if not stages:
        raise ValueError("feature_family_stages must not be empty")
    annotations = tuple(dict.fromkeys(bool(value) for value in auto_annotate_options))
    if not annotations:
        raise ValueError("auto_annotate_options must not be empty")

    configs: list[GraphConfig] = []
    # Family stages are the outer loop so base-only depth 1, 2, 3 is measured
    # before wider feature families. Keep annotation=True first by default.
    for families in stages:
        for annotate in annotations:
            for depth in range(1, max_depth + 1):
                configs.append(
                    GraphConfig(
                        feature_families=families,
                        depth=depth,
                        auto_annotate_features=annotate,
                        feature_family_max_columns=feature_family_max_columns,
                    )
                )
    return tuple(configs)


@dataclass
class Trial:
    config: GraphConfig
    metric: str
    validation_score: float
    objective_score: float
    feature_count: int
    train_rows: int
    validation_rows: int
    feature_seconds: float
    model_seconds: float
    model: Any = None
    feature_columns: tuple[str, ...] = ()
    categorical_columns: tuple[str, ...] = ()
    datetime_columns: tuple[str, ...] = ()
    execution_plan: dict[str, Any] | None = field(default=None, repr=False)
    negligible_gain: bool = False
    note: str | None = None
    status: str = "completed"
    error: str | None = None

    def as_record(self) -> dict[str, Any]:
        return {
            **asdict(self.config),
            "metric": self.metric,
            "validation_score": self.validation_score,
            "feature_count": self.feature_count,
            "train_rows": self.train_rows,
            "validation_rows": self.validation_rows,
            "feature_seconds": self.feature_seconds,
            "model_seconds": self.model_seconds,
            "negligible_gain": self.negligible_gain,
            "note": self.note,
            "status": self.status,
            "error": self.error,
        }


@dataclass
class FittedModel:
    """Production graph plan, feature schema, and estimator."""

    config: GraphConfig
    execution_plan: dict[str, Any]
    plan_fingerprint: str
    estimator: Any
    validation_estimator: Any
    feature_columns: tuple[str, ...]
    categorical_columns: tuple[str, ...]
    datetime_columns: tuple[str, ...]
    target: str
    task: str
    metric: str
    validation_score: float
    train_rows: int
    validation_rows: int
    training_frames: int
    target_classes: tuple[Any, ...] = ()
    test_predictions: pd.DataFrame | None = None
    test_score: float | None = None


@dataclass
class FitResult:
    """The fitted winner plus the full validation/complexity audit trail."""

    task: str
    metric: str
    best_trial: Trial
    recommended_trial: Trial
    trials: tuple[Trial, ...]
    fitted_model: FittedModel | None = None

    @property
    def best_config(self) -> GraphConfig:
        return self.best_trial.config

    @property
    def recommended_config(self) -> GraphConfig:
        return self.recommended_trial.config

    @property
    def model(self) -> Any:
        if self.fitted_model is not None:
            return self.fitted_model.estimator
        return self.best_trial.model

    @property
    def execution_plan(self) -> dict[str, Any] | None:
        if self.fitted_model is not None:
            return self.fitted_model.execution_plan
        return self.best_trial.execution_plan

    @property
    def full_validation_score(self) -> float | None:
        return (
            self.fitted_model.validation_score
            if self.fitted_model is not None
            else None
        )

    @property
    def plan_fingerprint(self) -> str | None:
        return (
            self.fitted_model.plan_fingerprint
            if self.fitted_model is not None
            else None
        )

    @property
    def test_score(self) -> float | None:
        return self.fitted_model.test_score if self.fitted_model is not None else None

    @property
    def test_predictions(self) -> pd.DataFrame | None:
        return (
            self.fitted_model.test_predictions
            if self.fitted_model is not None
            else None
        )

    @property
    def results(self) -> pd.DataFrame:
        return pd.DataFrame([trial.as_record() for trial in self.trials])

    @property
    def complexity_notes(self) -> tuple[str, ...]:
        return tuple(trial.note for trial in self.trials if trial.note)

    def predict(self, frame: pd.DataFrame) -> Any:
        from .modeling import prepare_prediction_features

        fitted = self.fitted_model
        feature_columns = (
            fitted.feature_columns if fitted is not None else self.best_trial.feature_columns
        )
        categorical_columns = (
            fitted.categorical_columns
            if fitted is not None
            else self.best_trial.categorical_columns
        )
        datetime_columns = (
            fitted.datetime_columns
            if fitted is not None
            else self.best_trial.datetime_columns
        )
        inputs = prepare_prediction_features(
            frame,
            columns=list(feature_columns),
            categorical_columns=set(categorical_columns),
            datetime_columns=set(datetime_columns),
        )
        if self.task == "classification":
            return self.model.predict_proba(inputs)[:, 1]
        return self.model.predict(inputs)


def annotate_complexity(
    trials: Iterable[Trial],
    *,
    classification_gain: float,
    regression_relative_gain: float,
    feature_growth: float,
) -> None:
    """Mark expensive trials whose improvement over a simpler trial is tiny."""

    completed: list[Trial] = []
    for trial in trials:
        if trial.status != "completed" or not isfinite(trial.objective_score):
            completed.append(trial)
            continue
        simpler = [
            other
            for other in completed
            if other.status == "completed" and isfinite(other.objective_score)
            if other.feature_count <= trial.feature_count
            and other.config.complexity <= trial.config.complexity
        ]
        if not simpler:
            completed.append(trial)
            continue
        prior = max(simpler, key=lambda item: item.objective_score)
        growth = trial.feature_count / max(1, prior.feature_count)
        if trial.metric == "roc_auc":
            gain = trial.validation_score - prior.validation_score
            threshold = classification_gain
            gain_label = f"AUC gain {gain:+.4f}"
        else:
            gain = (prior.validation_score - trial.validation_score) / max(
                abs(prior.validation_score), 1e-12
            )
            threshold = regression_relative_gain
            gain_label = f"relative MAE gain {gain:+.2%}"
        if growth >= feature_growth and gain <= threshold:
            trial.negligible_gain = True
            trial.note = (
                f"Negligible {gain_label} for {growth:.1f}x as many features "
                f"as a simpler configuration."
            )
        completed.append(trial)
