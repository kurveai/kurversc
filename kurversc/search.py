"""Configuration candidates and search results."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from itertools import combinations
from math import isfinite
from typing import Any, Iterable, Sequence

import pandas as pd


DEFAULT_FAMILY_STAGES: tuple[tuple[str, ...], ...] = (
    ("base",),
    ("base", "temporal"),
    ("base", "temporal", "sequence"),
    ("base", "temporal", "sequence", "conditional"),
    ("base", "temporal", "sequence", "conditional", "episode"),
)


def resolve_feature_family_column_budgets(
    initial_budget: int | None,
    options: Sequence[int | None] | None,
) -> tuple[int | None, ...]:
    """Return validated source-column budgets from narrow to broad."""

    if options is None:
        budgets = (initial_budget,)
    else:
        budgets = tuple(dict.fromkeys(options))
        if not budgets:
            raise ValueError("feature_family_max_column_options must not be empty")
    if any(budget is not None and budget < 1 for budget in budgets):
        raise ValueError(
            "feature_family_max_column_options must contain positive values or None"
        )
    return budgets


@dataclass(frozen=True)
class GraphConfig:
    feature_families: tuple[str, ...] = ("base",)
    depth: int = 1
    auto_annotate_features: bool = True
    feature_family_max_columns: int | None = 4
    feature_family_max_features_per_column: int | None = 32
    feature_propagation_max_functions_per_column: int | None = 1
    auto_text_features: bool = False
    auto_annotate_max_text_columns: int | None = None

    def __post_init__(self) -> None:
        if self.depth < 1:
            raise ValueError("depth must be at least 1")
        if not self.feature_families or self.feature_families[0] != "base":
            raise ValueError("feature_families must start with 'base'")
        if (
            self.feature_family_max_columns is not None
            and self.feature_family_max_columns < 1
        ):
            raise ValueError("feature_family_max_columns must be positive or None")
        if (
            self.feature_family_max_features_per_column is not None
            and self.feature_family_max_features_per_column < 1
        ):
            raise ValueError(
                "feature_family_max_features_per_column must be positive or None"
            )
        if (
            self.feature_propagation_max_functions_per_column is not None
            and self.feature_propagation_max_functions_per_column < 1
        ):
            raise ValueError(
                "feature_propagation_max_functions_per_column must be positive or None"
            )
        if (
            self.auto_annotate_max_text_columns is not None
            and self.auto_annotate_max_text_columns < 1
        ):
            raise ValueError("auto_annotate_max_text_columns must be positive or None")

    @property
    def complexity(self) -> float:
        annotation_factor = 1.25 if self.auto_annotate_features else 1.0
        return float(self.depth * len(self.feature_families) * annotation_factor)

    def graphreduce_kwargs(self) -> dict[str, Any]:
        return {
            "auto_features": True,
            "auto_feature_hops_back": self.depth,
            "auto_feature_hops_front": 0,
            "auto_text_features": self.auto_text_features,
            "auto_annotate_max_text_columns": self.auto_annotate_max_text_columns,
        }


def incremental_configs(
    *,
    max_depth: int = 3,
    feature_family_stages: Sequence[Sequence[str]] = DEFAULT_FAMILY_STAGES,
    auto_annotate_options: Sequence[bool] = (True, False),
    feature_family_max_columns: int | None = 4,
    feature_family_max_column_options: Sequence[int | None] | None = None,
    feature_family_max_features_per_column: int | None = 32,
    feature_propagation_max_functions_per_column: int | None = 1,
) -> tuple[GraphConfig, ...]:
    """Return the deterministic potential lattice for forward configuration search.

    The returned order is significant: base-only variants precede every
    one-family addition, which precedes pairs, and so on. ``fit`` uses that
    ordering with a bounded beam so only descendants of strong configurations
    are actually materialized. Explicit ``graph_configs`` continue to bypass
    beam pruning.
    """

    if max_depth < 1:
        raise ValueError("max_depth must be at least 1")
    stages = [tuple(dict.fromkeys(stage)) for stage in feature_family_stages]
    if not stages:
        raise ValueError("feature_family_stages must not be empty")
    annotations = tuple(dict.fromkeys(bool(value) for value in auto_annotate_options))
    if not annotations:
        raise ValueError("auto_annotate_options must not be empty")
    column_budgets = resolve_feature_family_column_budgets(
        feature_family_max_columns,
        feature_family_max_column_options,
    )

    optional_families = tuple(
        dict.fromkeys(
            family for stage in stages for family in stage if family != "base"
        )
    )
    family_sets = [
        ("base", *addition)
        for count in range(len(optional_families) + 1)
        for addition in combinations(optional_families, count)
    ]

    configs: list[GraphConfig] = []
    for column_budget in column_budgets:
        for families in family_sets:
            stage_max_depth = (
                max_depth
                if set(families).issubset({"base", "temporal", "sequence"})
                else min(max_depth, 2)
            )
            # Evaluate both annotation policies at a shallow depth before
            # deciding which one is worth propagating farther.
            for depth in range(1, stage_max_depth + 1):
                for annotate in annotations:
                    configs.append(
                        GraphConfig(
                            feature_families=families,
                            depth=depth,
                            auto_annotate_features=annotate,
                            feature_family_max_columns=column_budget,
                            feature_family_max_features_per_column=(
                                feature_family_max_features_per_column
                            ),
                            feature_propagation_max_functions_per_column=(
                                feature_propagation_max_functions_per_column
                            ),
                        )
                    )
    return tuple(configs)


def forward_candidate_allowed(
    config: GraphConfig,
    completed_trials: Sequence["Trial"],
    *,
    beam_width: int,
    feature_family_max_column_options: Sequence[int | None] | None = None,
    column_budget_promotions: Sequence[GraphConfig] | None = None,
) -> bool:
    """Return whether a lattice candidate descends from the current beam.

    Parent comparisons keep depth, annotation, text, and cap settings fixed;
    only one optional family may be added at a time. The beam is global within
    each family-count level so strong depth/annotation choices receive the
    remaining family budget.
    """

    if beam_width < 1:
        raise ValueError("beam_width must be positive")
    column_budgets = tuple(feature_family_max_column_options or ())
    if column_budgets and config.feature_family_max_columns in column_budgets:
        budget_index = column_budgets.index(config.feature_family_max_columns)
        if budget_index > 0:
            previous_budget = column_budgets[budget_index - 1]
            parents = [
                trial
                for trial in completed_trials
                if trial.status == "completed"
                and isfinite(trial.objective_score)
                and trial.config.feature_family_max_columns == previous_budget
            ]
            if column_budget_promotions is not None:
                promoted = tuple(column_budget_promotions)
                parents = [trial for trial in parents if trial.config in promoted]
                parents.sort(key=lambda trial: promoted.index(trial.config))
            else:
                parents.sort(
                    key=lambda trial: (
                        -trial.objective_score,
                        trial.feature_count,
                        trial.config.complexity,
                        trial.feature_seconds,
                    )
                )
            return any(
                parent.config.feature_families == config.feature_families
                and parent.config.depth == config.depth
                and parent.config.auto_annotate_features
                == config.auto_annotate_features
                and parent.config.auto_text_features == config.auto_text_features
                and parent.config.auto_annotate_max_text_columns
                == config.auto_annotate_max_text_columns
                and parent.config.feature_family_max_features_per_column
                == config.feature_family_max_features_per_column
                and parent.config.feature_propagation_max_functions_per_column
                == config.feature_propagation_max_functions_per_column
                for parent in parents[:beam_width]
            )

    level = len(config.feature_families) - 1
    if level == 0:
        return True
    parent_level = level - 1
    parents = [
        trial
        for trial in completed_trials
        if trial.status == "completed"
        and isfinite(trial.objective_score)
        and len(trial.config.feature_families) - 1 == parent_level
    ]
    parents.sort(
        key=lambda trial: (
            -trial.objective_score,
            trial.feature_count,
            trial.config.complexity,
            trial.feature_seconds,
        )
    )
    beam = parents[:beam_width]
    config_families = set(config.feature_families)
    for trial in beam:
        parent = trial.config
        if (
            parent.depth == config.depth
            and parent.auto_annotate_features == config.auto_annotate_features
            and parent.feature_family_max_columns == config.feature_family_max_columns
            and parent.feature_family_max_features_per_column
            == config.feature_family_max_features_per_column
            and parent.feature_propagation_max_functions_per_column
            == config.feature_propagation_max_functions_per_column
            and parent.auto_text_features == config.auto_text_features
            and parent.auto_annotate_max_text_columns
            == config.auto_annotate_max_text_columns
            and set(parent.feature_families).issubset(config_families)
            and len(config_families) - len(parent.feature_families) == 1
        ):
            return True
    return False


def adaptive_depth_candidate_allowed(
    config: GraphConfig,
    completed_trials: Sequence["Trial"],
    *,
    classification_gain: float,
    regression_relative_gain: float,
    uncertainty_multiplier: float,
    beam_width: int = 1,
) -> bool:
    """Promote only the strongest annotation policy to deeper graph hops.

    Both annotation policies are screened at depth 1. The strongest policy is
    promoted to depth 2. Depth 3 is considerably more expensive, so it also
    requires depth 2 to improve meaningfully over the matching depth-1 frame.
    Explicit ``graph_configs`` bypass this default-search guard.
    """

    if config.depth == 1:
        return True
    if beam_width < 1:
        raise ValueError("beam_width must be positive")
    matching = [
        trial
        for trial in completed_trials
        if trial.status == "completed"
        and trial.config.feature_families == config.feature_families
        and trial.config.feature_family_max_columns == config.feature_family_max_columns
        and trial.config.feature_family_max_features_per_column
        == config.feature_family_max_features_per_column
        and trial.config.feature_propagation_max_functions_per_column
        == config.feature_propagation_max_functions_per_column
        and trial.config.depth < config.depth
    ]
    immediate_parents = sorted(
        (trial for trial in matching if trial.config.depth == config.depth - 1),
        key=lambda trial: (
            -trial.objective_score,
            trial.feature_count,
            trial.feature_seconds,
        ),
    )[:beam_width]
    parent = next(
        (
            trial
            for trial in immediate_parents
            if trial.config.auto_annotate_features == config.auto_annotate_features
        ),
        None,
    )
    if parent is None:
        return False
    if config.depth == 2:
        return True
    shallow = next(
        (
            trial
            for trial in matching
            if trial.config.depth == 1
            and trial.config.auto_annotate_features == config.auto_annotate_features
        ),
        None,
    )
    if shallow is None:
        return False
    uncertainty = (
        uncertainty_multiplier
        * (shallow.validation_standard_error**2 + parent.validation_standard_error**2)
        ** 0.5
    )
    if parent.metric == "roc_auc":
        required = max(classification_gain, uncertainty)
        return parent.validation_score - shallow.validation_score > required
    required = max(
        abs(shallow.validation_score) * regression_relative_gain,
        uncertainty,
    )
    return shallow.validation_score - parent.validation_score > required


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
    validation_standard_error: float = 0.0
    model: Any = None
    feature_columns: tuple[str, ...] = ()
    categorical_columns: tuple[str, ...] = ()
    datetime_columns: tuple[str, ...] = ()
    execution_plan: dict[str, Any] | None = field(default=None, repr=False)
    negligible_gain: bool = False
    note: str | None = None
    status: str = "completed"
    error: str | None = None
    stage: str = "screening"
    sample_rows: int | None = None
    estimated_feature_count: int | None = None

    def as_record(self) -> dict[str, Any]:
        return {
            **asdict(self.config),
            "metric": self.metric,
            "validation_score": self.validation_score,
            "validation_standard_error": self.validation_standard_error,
            "feature_count": self.feature_count,
            "train_rows": self.train_rows,
            "validation_rows": self.validation_rows,
            "feature_seconds": self.feature_seconds,
            "model_seconds": self.model_seconds,
            "negligible_gain": self.negligible_gain,
            "note": self.note,
            "status": self.status,
            "error": self.error,
            "stage": self.stage,
            "sample_rows": self.sample_rows,
            "estimated_feature_count": self.estimated_feature_count,
        }


def diverse_confirmation_trials(
    successful: Sequence[Trial],
    *,
    top_k: int,
    complexity_recommendation: Trial | None = None,
) -> tuple[Trial, ...]:
    """Return score-ordered finalists while preserving structural diversity.

    A small screening frame can let near-identical shapes occupy every leading
    rank. Confirmation instead keeps the raw winner, the complexity-aware
    candidate, representatives of optional families, a deeper shape, and the
    alternate annotation policy before filling remaining slots by raw rank.
    """

    if top_k < 0:
        raise ValueError("top_k must be non-negative")
    if top_k == 0:
        return ()
    ranked = sorted(
        (
            trial
            for trial in successful
            if trial.status == "completed" and isfinite(trial.objective_score)
        ),
        key=lambda trial: (
            -trial.objective_score,
            trial.feature_count,
            trial.config.complexity,
            trial.feature_seconds,
        ),
    )
    if not ranked:
        return ()

    selected: list[Trial] = []
    selected_configs: set[GraphConfig] = set()

    def add(trial: Trial | None) -> None:
        if (
            trial is not None
            and len(selected) < top_k
            and trial.config not in selected_configs
        ):
            selected.append(trial)
            selected_configs.add(trial.config)

    raw_winner = ranked[0]
    add(raw_winner)
    add(complexity_recommendation)

    optional_families = tuple(
        family
        for family in dict.fromkeys(
            family for trial in ranked for family in trial.config.feature_families
        )
        if family != "base"
    )
    for family in optional_families:
        add(
            next(
                (
                    trial
                    for trial in ranked
                    if family in trial.config.feature_families
                    and trial.config not in selected_configs
                ),
                None,
            )
        )

    add(
        next(
            (
                trial
                for trial in ranked
                if trial.config.depth > 1 and trial.config not in selected_configs
            ),
            None,
        )
    )
    add(
        next(
            (
                trial
                for trial in ranked
                if trial.config.auto_annotate_features
                != raw_winner.config.auto_annotate_features
                and trial.config not in selected_configs
            ),
            None,
        )
    )
    for trial in ranked:
        add(trial)
        if len(selected) == top_k:
            break
    return tuple(selected)


def stability_adjusted_winner(successful: Sequence[Trial]) -> Trial:
    """Return the raw winner after any stability penalty in objective_score."""

    completed = [
        trial
        for trial in successful
        if trial.status == "completed" and isfinite(trial.objective_score)
    ]
    if not completed:
        raise ValueError("At least one successful trial is required")
    return max(completed, key=lambda trial: trial.objective_score)


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
    model_backend: str
    target_classes: tuple[Any, ...] = ()
    test_predictions: pd.DataFrame | None = None
    test_score: float | None = None


@dataclass
class FitResult:
    """The fitted winner plus screening and full-history audit trails."""

    task: str
    metric: str
    best_trial: Trial
    recommended_trial: Trial
    trials: tuple[Trial, ...]
    confirmation_trials: tuple[Trial, ...] = ()
    rerank_trials: tuple[Trial, ...] = ()
    fitted_model: FittedModel | None = None
    feature_audit: pd.DataFrame = field(default_factory=pd.DataFrame)

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
    def rerank_results(self) -> pd.DataFrame:
        return pd.DataFrame([trial.as_record() for trial in self.rerank_trials])

    @property
    def confirmation_results(self) -> pd.DataFrame:
        return pd.DataFrame([trial.as_record() for trial in self.confirmation_trials])

    @property
    def complexity_notes(self) -> tuple[str, ...]:
        return tuple(trial.note for trial in self.trials if trial.note)

    def predict(self, frame: pd.DataFrame) -> Any:
        from .modeling import prepare_prediction_features

        fitted = self.fitted_model
        feature_columns = (
            fitted.feature_columns
            if fitted is not None
            else self.best_trial.feature_columns
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
    uncertainty_multiplier: float = 1.0,
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
        uncertainty = (
            uncertainty_multiplier
            * (trial.validation_standard_error**2 + prior.validation_standard_error**2)
            ** 0.5
        )
        if trial.metric == "roc_auc":
            gain = trial.validation_score - prior.validation_score
            threshold = max(classification_gain, uncertainty)
            gain_label = f"AUC gain {gain:+.4f}"
        else:
            gain = (prior.validation_score - trial.validation_score) / max(
                abs(prior.validation_score), 1e-12
            )
            relative_uncertainty = uncertainty / max(abs(prior.validation_score), 1e-12)
            threshold = max(regression_relative_gain, relative_uncertainty)
            gain_label = f"relative MAE gain {gain:+.2%}"
        if growth >= feature_growth and gain <= threshold:
            trial.negligible_gain = True
            trial.note = (
                f"Negligible {gain_label} for {growth:.1f}x as many features "
                f"as a simpler configuration (meaningful-gain threshold "
                f"{threshold:.4g})."
            )
        completed.append(trial)
