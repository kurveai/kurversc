#!/usr/bin/env python3
"""Evaluate fidelity-audit winners on the official RelArena test protocol.

The graph configuration is fixed before RelArena exposes the outer split.  The
inner arm re-materializes that configuration on the uncapped, test-censored
database so automatic annotations can produce a frozen execution plan.  The
outer arm then replays the configuration and plan, fits on the official
train/validation data, predicts the masked test rows, and lets RelArena score
those predictions against the held-out test labels.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass, replace
import gc
import json
import os
from pathlib import Path
import pickle
import sys
import time
from typing import Any

import pandas as pd
import duckdb

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

import kurversc  # noqa: E402
from kurversc.checkpointing import search_runtime_identity  # noqa: E402
from kurversc.temporal_sampling import choose_temporal_fit_plan  # noqa: E402
from kurversc.staged_adapter import (  # noqa: E402
    KURVERSC_DEFAULTS,
    KurveRSCSystem,
    _runtime_config,
    summary_to_dataframe,
    run_system_experiment,
)


@dataclass(frozen=True)
class Recipe:
    dataset: str
    task: str
    config: kurversc.GraphConfig
    search_training_frames: int
    published_score: float | None
    higher_is_better: bool | None
    audit_source: str


RECIPES = {
    ("rel-f1", "driver-dnf"): Recipe(
        dataset="rel-f1",
        task="driver-dnf",
        config=kurversc.GraphConfig(
            feature_families=("base", "state"),
            depth=1,
            auto_annotate_features=False,
            feature_family_max_columns=4,
            feature_family_max_features_per_column=32,
            feature_propagation_max_functions_per_column=1,
            auto_text_features=False,
            auto_annotate_max_text_columns=None,
            high_cardinality_policy="top_k",
            text_categorical_fallback=True,
            include_target_history=False,
            max_base_table_occurrences=1,
        ),
        search_training_frames=12,
        published_score=0.753755,
        higher_is_better=True,
        audit_source="full-three temporal confirmation",
    ),
    ("rel-f1", "driver-top3"): Recipe(
        dataset="rel-f1",
        task="driver-top3",
        config=kurversc.GraphConfig(
            feature_families=("base", "state"),
            depth=1,
            auto_annotate_features=False,
            feature_family_max_columns=4,
            feature_family_max_features_per_column=32,
            feature_propagation_max_functions_per_column=1,
            auto_text_features=False,
            auto_annotate_max_text_columns=None,
            high_cardinality_policy="top_k",
            text_categorical_fallback=True,
            include_target_history=False,
            max_base_table_occurrences=1,
        ),
        search_training_frames=12,
        published_score=0.682359,
        higher_is_better=True,
        audit_source="full-three temporal confirmation with recent-fold veto",
    ),
    ("rel-f1", "driver-position"): Recipe(
        dataset="rel-f1",
        task="driver-position",
        config=kurversc.GraphConfig(
            feature_families=("base",),
            depth=1,
            auto_annotate_features=True,
            feature_family_max_columns=4,
            feature_family_max_features_per_column=32,
            feature_propagation_max_functions_per_column=1,
            auto_text_features=False,
            auto_annotate_max_text_columns=None,
            high_cardinality_policy="top_k",
            text_categorical_fallback=True,
            include_target_history=False,
            max_base_table_occurrences=1,
        ),
        search_training_frames=12,
        published_score=3.913433,
        higher_is_better=False,
        audit_source="full-three temporal confirmation",
    ),
    ("rel-avito", "user-clicks"): Recipe(
        dataset="rel-avito",
        task="user-clicks",
        config=kurversc.GraphConfig(
            feature_families=("base", "sequence", "episode"),
            depth=2,
            auto_annotate_features=False,
            feature_family_max_columns=4,
            feature_family_max_features_per_column=32,
            feature_propagation_max_functions_per_column=1,
            auto_text_features=False,
            auto_annotate_max_text_columns=None,
            high_cardinality_policy="top_k",
            text_categorical_fallback=True,
            include_target_history=True,
            max_base_table_occurrences=2,
        ),
        search_training_frames=3,
        published_score=0.663718,
        higher_is_better=True,
        audit_source="32-config diverse-promotion 50K confirmation",
    ),
}


# Held-out reference values published in this repository.  Generic search
# profiles use these only after prediction, for reporting; they are never
# visible to configuration selection or model fitting.
PUBLISHED_REFERENCES: dict[tuple[str, str], tuple[float, bool]] = {
    ("rel-amazon", "user-churn"): (0.709873, True),
    ("rel-amazon", "item-churn"): (0.826457, True),
    ("rel-amazon", "user-ltv"): (14.141898, False),
    ("rel-amazon", "item-ltv"): (42.379024, False),
    ("rel-avito", "user-visits"): (0.674190, True),
    ("rel-avito", "user-clicks"): (0.663718, True),
    ("rel-avito", "ad-ctr"): (0.033658, False),
    ("rel-event", "user-repeat"): (0.754278, True),
    ("rel-event", "user-ignore"): (0.831978, True),
    ("rel-event", "user-attendance"): (0.260420, False),
    ("rel-f1", "driver-dnf"): (0.753755, True),
    ("rel-f1", "driver-top3"): (0.682359, True),
    ("rel-f1", "driver-position"): (3.913433, False),
    ("rel-hm", "user-churn"): (0.696306, True),
    ("rel-hm", "item-sales"): (0.031837, False),
    ("rel-stack", "user-engagement"): (0.903012, True),
    ("rel-stack", "user-badge"): (0.875081, True),
    ("rel-stack", "post-votes"): (0.063356, False),
    ("rel-trial", "study-outcome"): (0.708141, True),
    ("rel-trial", "study-adverse"): (41.333896, False),
    ("rel-trial", "site-success"): (0.401755, False),
}


def _recipe_for(key: tuple[str, str], *, generic: bool = False, task_type=None) -> Recipe:
    if generic:
        return Recipe(dataset=key[0],task=key[1],config=kurversc.GraphConfig(),
            search_training_frames=1,published_score=None,
            higher_is_better=("classification" in str(task_type).lower()) if task_type is not None else None,
            audit_source="schema-derived staged search; no benchmark recipe or reference score")
    if key in RECIPES:
        return RECIPES[key]
    score, higher_is_better = PUBLISHED_REFERENCES[key]
    return Recipe(
        dataset=key[0],
        task=key[1],
        config=kurversc.GraphConfig(),
        search_training_frames=1,
        published_score=score,
        higher_is_better=higher_is_better,
        audit_source="schema-generic staged production search",
    )


def _adequate_search_training_frames(
    task: Any,
    train_table: Any,
    *,
    minimum_cutoffs: int = 3,
    maximum_cutoffs: int = 12,
    minimum_labels: int = 256,
    minimum_class_labels: int = 24,
) -> int:
    """Choose the smallest recent temporal window with usable label evidence."""

    frame = train_table.df if hasattr(train_table, "df") else train_table
    labels = frame.loc[
        frame[task.target_col].notna() & frame[task.time_col].notna(),
        [task.time_col, task.target_col],
    ].copy()
    if labels.empty:
        return minimum_cutoffs
    labels[task.time_col] = pd.to_datetime(labels[task.time_col])
    cutoffs = tuple(sorted(labels[task.time_col].unique()))
    upper = min(maximum_cutoffs, len(cutoffs))
    lower = min(minimum_cutoffs, upper)
    classification = "classification" in str(task.task_type).lower()
    for count in range(lower, upper + 1):
        recent = labels.loc[labels[task.time_col].isin(cutoffs[-count:])]
        if len(recent) < minimum_labels:
            continue
        if classification:
            class_counts = recent[task.target_col].value_counts(dropna=False)
            if len(class_counts) < 2 or int(class_counts.min()) < minimum_class_labels:
                continue
        return count
    return max(1, upper)


# Diagnostic controls recovered by running the exact published KurveRSC 0.1.1
# selector on the inner split.  These are deliberately kept separate from the
# production recipes: they let us distinguish representation drift from search
# regret without selecting a configuration on the official test labels.
PUBLISHED_LINEAGE_CONTROLS = {
    ("rel-f1", "driver-dnf"): kurversc.GraphConfig(
        feature_families=("base", "temporal", "conditional", "episode"),
        depth=1,
        auto_annotate_features=True,
        feature_family_max_columns=4,
        feature_family_max_features_per_column=32,
        feature_propagation_max_functions_per_column=1,
        auto_text_features=False,
        auto_annotate_max_text_columns=None,
        high_cardinality_policy="top_k",
        categorical_low_cardinality_policy="all",
        text_categorical_fallback=True,
        include_target_history=False,
        max_base_table_occurrences=1,
        published_cutoff_compatibility=True,
        categorical_tie_break="legacy",
        legacy_training_row_order=True,
    ),
}


class AuditWinnerSystem(KurveRSCSystem):
    def run(self, task, *, inner_split, outer_split, seed, time_limit=None):
        # Native graph arms are internal stages, not recursive invocations of
        # the registered complete system. Older adapters expose only run().
        native = getattr(super(), "_run_native", None)
        if native is None:
            native = super().run
        return native(task, inner_split=inner_split, outer_split=outer_split,
            seed=seed, time_limit=time_limit)

    """KurveRSC system with either one finalist or a compatibility oracle."""

    name = "kurversc-fidelity-audit-winner"
    checkpoint_root: Path | None = None
    profile = "audited-finalist"
    search_parallelism = 1
    root_reachable_materialization = False
    rerank_nested_depth_safety = False
    fixed_config: kurversc.GraphConfig | None = None
    fixed_cdf_pair = False
    fixed_search_training_frames = 12
    full_training_frames = 1
    fixed_rerank = False
    fixed_execution_plan: dict[str, Any] | None = None
    inner_records: dict[tuple[str, str], dict[str, Any]] = {}
    temporal_fit_policy = "fixed"
    temporal_fit_max_rows = 100_000
    temporal_fit_max_features = 5_000
    temporal_fit_max_search_seconds = 7_200

    def _fit_arm(
        self,
        task: Any,
        db: Any,
        train_table: Any,
        val_table: Any,
        *,
        seed: int,
        phase: str,
        selection: Any | None = None,
    ) -> tuple[Any, Any]:
        if phase != "inner":
            if self.peer_context_mode is not None:
                if selection is None:
                    raise RuntimeError("The reporting arm requires an inner selection")
                config = _runtime_config()
                full_training_frames = (
                    selection.training_frames
                    if selection.training_frames is not None
                    else int(config["full_training_frames"])
                )
                config["full_training_frames"] = full_training_frames
                config["temporal_estimator_mode"] = getattr(
                    self,
                    "selected_temporal_estimator_mode",
                    config["temporal_estimator_mode"],
                )
                identity = self.run_identity
                problem = kurversc.relbench_problem_from_objects(
                    task,
                    db,
                    train_table,
                    val_table,
                    dataset_name=(
                        identity.dataset if identity is not None else "relbench"
                    ),
                    task_name=(identity.task if identity is not None else None),
                    sample_rows=int(config["sample_rows"]),
                    search_full_data=bool(config["search_full_data"]),
                    search_training_frames=int(config["search_training_frames"]),
                    max_train_timestamps=full_training_frames,
                    schema_depth=self.schema_depth,
                    max_base_table_occurrences=int(
                        os.environ.get("KURVERSC_MAX_BASE_TABLE_OCCURRENCES", "1")
                    ),
                    include_target_history=(
                        os.environ.get("KURVERSC_INCLUDE_TARGET_HISTORY", "1")
                        .lower()
                        not in {"0", "false", "no"}
                    ),
                    random_state=seed,
                )
                # The reporting arm may use observed validation labels as
                # history, but never masked test labels. Attach the same peer
                # feature schema as the inner arm before replaying its plan.
                problem = kurversc.add_relbench_peer_context(
                    problem,
                    train_table,
                    additional_history_table=val_table,
                    mode=self.peer_context_mode,
                    max_history_rows=self.peer_max_history_rows,
                )
                fit_kwargs = {
                    **problem.fit_kwargs(),
                    **config,
                    "random_state": seed,
                    "preselected_config": selection.config,
                    "preselected_execution_plan": copy.deepcopy(
                        selection.execution_plan
                    ),
                }
                catboost_iterations = os.environ.get("KURVERSC_CATBOOST_ITERATIONS")
                if catboost_iterations is not None:
                    fit_kwargs["model_params"] = {
                        "iterations": int(catboost_iterations)
                    }
                final_backend = os.environ.get("KURVERSC_FINAL_MODEL_BACKEND")
                if final_backend:
                    fit_kwargs["final_model_backend"] = final_backend
                return kurversc.fit(**fit_kwargs), problem
            return super()._fit_arm(
                task,
                db,
                train_table,
                val_table,
                seed=seed,
                phase=phase,
                selection=selection,
            )

        identity = self.run_identity
        if identity is None:
            raise RuntimeError("The selected-config runner requires a run identity")
        arm_started = time.perf_counter()
        recipe = _recipe_for((identity.dataset, identity.task),
            generic=self.profile in {"staged-adaptive","staged-bounded","staged-retention",
                "staged-calibrated","staged-diverse","staged-production","published-compat-search"},
            task_type=task.task_type)
        search_full_data = True
        screening_rows: int | None = None
        screening_cost_ratio_threshold: float | None = None
        structural_promotion = False
        shallow_sparse_promotion = 0
        confirmation_top_k = 0
        fidelity_promotion_top_k = 0
        fidelity_promotion_training_frames: int | None = None
        categorical_refinement_top_k = 0
        stage_candidate_budgets: kurversc.SearchStageBudgets | None = None
        max_base_table_occurrence_options: tuple[int, ...] | None = None
        capability_pruning = False
        full_training_frames = self.full_training_frames
        if self.profile == "fixed-config":
            if self.fixed_config is None:
                raise RuntimeError("fixed-config profile requires --fixed-config-json")
            if self.fixed_execution_plan is not None:
                graph_configs = None
            elif self.fixed_cdf_pair:
                graph_configs = tuple(
                    replace(self.fixed_config, base_distribution_features=enabled)
                    for enabled in (False, True)
                )
            else:
                graph_configs = (self.fixed_config,)
            search_training_frames = self.fixed_search_training_frames
            rerank_top_k = 2 if self.fixed_cdf_pair else 1 if self.fixed_rerank else 0
            staged_search_top_k = 0
            search_max_features = 10_000
            intelligent_column_selection = True
            complexity_selector = False
        elif self.profile == "published-compat-search":
            graph_configs = kurversc.published_compatible_configs()
            search_training_frames = 1
            rerank_top_k = 3
            staged_search_top_k = 0
            search_max_features = 8_000
            intelligent_column_selection = True
            # The published rerank treated expensive full-cutoff evidence as
            # authoritative.  Complexity remains in the audit but must not
            # replace its raw winner in the release-reproduction arm.
            complexity_selector = False
        elif self.profile == "staged-production":
            # Search cheaply at the most recent cutoff, retain structural
            # representatives through a 50K row confirmation, then give that
            # diverse set enough historical label frames to expose signals
            # that a one-cutoff screen cannot measure.  Only after this
            # successive-fidelity funnel do we pay for full walk-forward
            # reranking.  The release winner remains a symmetric control.
            graph_configs = None
            search_training_frames = 1
            rerank_top_k = 8
            staged_search_top_k = 6
            search_max_features = 10_000
            intelligent_column_selection = True
            complexity_selector = False
            search_full_data = False
            screening_rows = 10_000
            confirmation_top_k = 24
            fidelity_promotion_top_k = 16
            fidelity_promotion_training_frames = _adequate_search_training_frames(
                task, train_table
            )
        elif self.profile in {
            "staged-bounded",
            "staged-retention",
            "staged-calibrated",
            "staged-diverse",
        }:
            # Complete schema-generic staged search under hard evaluation
            # ceilings. The initial panel preserves structural coverage, the
            # challenger stage exposes depth four and bounded self-relational
            # paths, and categorical hash seeds are evaluated only for two
            # temporally promoted finalists.
            graph_configs = None
            search_training_frames = _adequate_search_training_frames(task, train_table)
            rerank_top_k = 12 if self.profile == "staged-retention" else 8
            staged_search_top_k = 6
            search_max_features = 10_000
            intelligent_column_selection = True
            complexity_selector = True
            search_full_data = False
            screening_rows = 10_000
            if self.profile in {"staged-calibrated", "staged-diverse"}:
                screening_cost_ratio_threshold = 1.5
                structural_promotion = True
            if self.profile == "staged-diverse":
                shallow_sparse_promotion = 3
            confirmation_top_k = 40 if self.profile == "staged-retention" else 24
            fidelity_promotion_top_k = 16
            fidelity_promotion_training_frames = search_training_frames
            # Search confidence and the deployed estimator horizon are
            # independent.  The published RelArena contract fits production
            # from the latest complete training cutoff; widening search to
            # historical cutoffs must not silently add equally weighted old
            # models to the final frame ensemble.
            full_training_frames = 1
            categorical_refinement_top_k = 2
            max_base_table_occurrence_options = (1, 2)
            stage_candidate_budgets = kurversc.SearchStageBudgets(
                screening=32,
                challenger=24,
                combined_challenger=12,
                confirmation=40 if self.profile == "staged-retention" else 12,
                fidelity_promotion=8,
                categorical_refinement=8,
                rerank=(
                    12
                    if self.profile == "staged-retention"
                    else 7 if self.profile == "staged-diverse" else 4
                ),
            )
            capability_pruning = True
        elif self.profile == "staged-adaptive":
            graph_configs = None
            search_training_frames = _adequate_search_training_frames(task, train_table)
            rerank_top_k = 5
            staged_search_top_k = 6
            search_max_features = 10_000
            intelligent_column_selection = True
            complexity_selector = False
            search_full_data = False
            screening_rows = 10_000
            confirmation_top_k = 16
            full_training_frames = 1
        elif self.profile == "published-lineage-control":
            try:
                graph_configs = (
                    PUBLISHED_LINEAGE_CONTROLS[(identity.dataset, identity.task)],
                )
            except KeyError as exc:
                raise RuntimeError(
                    "No published-lineage control is recorded for "
                    f"{identity.dataset}/{identity.task}"
                ) from exc
            search_training_frames = 1
            rerank_top_k = 0
            staged_search_top_k = 0
            search_max_features = 8_000
            # KurveRSC 0.1.1 already used the deterministic metadata-utility
            # ordering that became the explicit intelligent-selection knob.
            # Source order remains available as an ablation, but it changes
            # both the bounded feature payload and seeded CatBoost training.
            intelligent_column_selection = True
            complexity_selector = False
        else:
            graph_configs = (recipe.config,)
            search_training_frames = recipe.search_training_frames
            rerank_top_k = 0
            staged_search_top_k = 0
            search_max_features = 10_000
            intelligent_column_selection = True
            complexity_selector = False
        defaults = {
            **KURVERSC_DEFAULTS,
            "search_training_frames": search_training_frames,
            "fidelity_promotion_top_k": fidelity_promotion_top_k,
            "fidelity_promotion_training_frames": (fidelity_promotion_training_frames),
            "categorical_refinement_top_k": categorical_refinement_top_k,
            "categorical_refinement_seeds": (0, 1, 2, 3, 4, 5, 6, 7),
            "stage_candidate_budgets": stage_candidate_budgets,
            "max_base_table_occurrence_options": (max_base_table_occurrence_options),
            "full_training_frames": full_training_frames,
            "search_full_data": search_full_data,
            "screening_rows": screening_rows,
            "screening_cost_ratio_threshold": screening_cost_ratio_threshold,
            "structural_promotion": structural_promotion,
            "confidence_frontier_promotion": self.profile == "staged-retention",
            "shallow_sparse_promotion": shallow_sparse_promotion,
            "rerank_nested_depth_safety": self.rerank_nested_depth_safety,
            "confirmation_top_k": confirmation_top_k,
            "rerank_top_k": rerank_top_k,
            "staged_search_top_k": staged_search_top_k,
            "adaptive_depth_promotion": False,
            "capability_pruning": capability_pruning,
            "complexity_selector": complexity_selector,
            "intelligent_column_selection": intelligent_column_selection,
            "high_cardinality_policy": "skip",
            "text_categorical_fallback": False,
            "search_parallelism": self.search_parallelism,
            "search_max_features": search_max_features,
            "root_reachable_materialization": self.root_reachable_materialization,
        }
        # RelArena's outer arm is implemented by the upstream system class and
        # reads this shared profile. Carry the independently configured
        # production horizon forward; each campaign task runs in its own
        # process.
        KURVERSC_DEFAULTS["full_training_frames"] = full_training_frames
        adapter_train_timestamps = max(
            search_training_frames,
            fidelity_promotion_training_frames or 1,
            int(defaults["rerank_cutoff_frames"]) if rerank_top_k else 1,
        )
        # Walk-forward reranking reads historical labels from Labels.source;
        # it does not need those cutoffs injected into the task-connected
        # search cohort. Expanding the cohort merely to satisfy reranking
        # changes unsupervised feature planning before the candidate is scored.
        adapter_fidelity_training_frames = (
            fidelity_promotion_training_frames or search_training_frames
        )
        problem = kurversc.relbench_problem_from_objects(
            task,
            db,
            train_table,
            val_table,
            dataset_name=identity.dataset,
            task_name=identity.task,
            sample_rows=int(defaults["sample_rows"]),
            # The adapter owns deterministic nested task-connected samples;
            # full-data profiles simply leave those sources uncapped.
            search_full_data=search_full_data,
            # Keep cheap plan discovery and higher-fidelity evaluation in
            # separate task-connected cohorts. Promotion replays the frozen
            # plan instead of changing the candidate as cutoffs are added.
            search_training_frames=search_training_frames,
            fidelity_training_frames=adapter_fidelity_training_frames,
            max_train_timestamps=adapter_train_timestamps,
            schema_depth=self.schema_depth,
            random_state=seed,
        )
        if self.peer_context_mode is not None:
            problem = kurversc.add_relbench_peer_context(
                problem,
                train_table,
                mode=self.peer_context_mode,
                max_history_rows=self.peer_max_history_rows,
            )
        fit_kwargs: dict[str, Any] = {
            **problem.fit_kwargs(),
            **defaults,
            "random_state": seed,
        }
        catboost_iterations = os.environ.get("KURVERSC_CATBOOST_ITERATIONS")
        if catboost_iterations is not None:
            fit_kwargs["model_params"] = {
                "iterations": int(catboost_iterations)
            }
        if graph_configs is not None:
            fit_kwargs["graph_configs"] = graph_configs
        if self.fixed_execution_plan is not None:
            fit_kwargs["preselected_config"] = self.fixed_config
            fit_kwargs["preselected_execution_plan"] = copy.deepcopy(
                self.fixed_execution_plan
            )
        if self.checkpoint_root is not None:
            cdf_checkpoint_suffix = (
                "default"
                if KURVERSC_DEFAULTS.get("base_distribution_features") is None
                else "on"
                if KURVERSC_DEFAULTS["base_distribution_features"]
                else "off"
            )
            checkpoint_namespace = f"{self.profile}-base-cdf-{cdf_checkpoint_suffix}"
            if self.peer_context_mode is not None:
                checkpoint_namespace += (
                    f"-peer-{self.peer_context_mode}-h{self.peer_max_history_rows}"
                )
            fit_kwargs["search_checkpoint_path"] = (
                self.checkpoint_root
                / checkpoint_namespace
                / f"{identity.dataset}_{identity.task}"
            )
            fit_kwargs["search_checkpoint_key"] = (
                "official-test/"
                f"{checkpoint_namespace}/{identity.dataset}/{identity.task}/inner"
            )
        result = kurversc.fit(**fit_kwargs)
        search_result = result
        temporal_promotion: dict[str, Any] = {
            "policy": self.temporal_fit_policy,
            "status": "disabled",
            "training_frames": full_training_frames,
            "estimator_mode": defaults["temporal_estimator_mode"],
        }
        if self.temporal_fit_policy == "cutoff_density":
            label_source = problem.label_node.source
            split_column = problem.label_node.split
            if not isinstance(label_source, pd.DataFrame) or split_column is None:
                raise RuntimeError(
                    "Cutoff-density promotion requires task label split metadata"
                )
            root_source = problem.parent_node.source
            root_frame = (
                root_source if isinstance(root_source, pd.DataFrame) else None
            )
            temporal_plan = choose_temporal_fit_plan(
                label_source.loc[
                    label_source[split_column] == problem.label_node.train_value
                ],
                label_source.loc[
                    label_source[split_column] == problem.label_node.validation_value
                ],
                timestamp=problem.timestamp_column,
                root_frame=root_frame,
                label_key=problem.label_node.key,
                root_key=problem.parent_node.key,
                root_timestamp=(
                    problem.parent_node.date if root_frame is not None else None
                ),
                max_training_rows=self.temporal_fit_max_rows,
            )
            temporal_promotion.update(
                status=temporal_plan.reason,
                recent_frame_rows=temporal_plan.recent_frame_rows,
            )
            if (
                temporal_plan.training_frames == 3
                and search_result.recommended_trial.feature_count
                <= self.temporal_fit_max_features
                and time.perf_counter() - arm_started
                <= self.temporal_fit_max_search_seconds
            ):
                promotion_kwargs = {
                    **fit_kwargs,
                    "preselected_config": search_result.recommended_config,
                    "preselected_execution_plan": copy.deepcopy(
                        search_result.execution_plan
                    ),
                    "full_training_frames": 3,
                    "temporal_estimator_mode": temporal_plan.estimator_mode,
                    "rerank_top_k": 0,
                    "staged_search_top_k": 0,
                    "fidelity_promotion_top_k": 0,
                    "categorical_refinement_top_k": 0,
                    "confirmation_top_k": 0,
                }
                promotion_kwargs.pop("graph_configs", None)
                try:
                    result = kurversc.fit(**promotion_kwargs)
                except (duckdb.OutOfMemoryException, MemoryError) as exc:
                    temporal_promotion.update(
                        status="resource_fallback",
                        error=f"{type(exc).__name__}: {exc}",
                    )
                    result = search_result
                    gc.collect()
                else:
                    temporal_promotion.update(
                        status="promoted",
                        training_frames=3,
                        estimator_mode=temporal_plan.estimator_mode,
                        validation_score=result.full_validation_score,
                    )
                    KURVERSC_DEFAULTS["full_training_frames"] = 3
                    KURVERSC_DEFAULTS[
                        "temporal_estimator_mode"
                    ] = temporal_plan.estimator_mode
            elif temporal_plan.training_frames == 3:
                temporal_promotion["status"] = "promotion_budget"
        self.selected_temporal_estimator_mode = temporal_promotion["estimator_mode"]
        print(
            json.dumps(
                {"event": "temporal_fit_promotion", **temporal_promotion},
                default=str,
                sort_keys=True,
            ),
            flush=True,
        )
        stage_frames = (
            search_result.results.copy(),
            search_result.confirmation_results.copy(),
            search_result.rerank_results.copy(),
        )
        stage_counts: dict[str, dict[str, int]] = {}
        for frame in stage_frames:
            if frame.empty:
                continue
            for (stage, status), count in (
                frame.groupby(["stage", "status"], dropna=False).size().items()
            ):
                stage_counts.setdefault(str(stage), {})[str(status)] = int(count)
        self.inner_records[(identity.dataset, identity.task)] = {
            "selected_config": search_result.recommended_config,
            "best_config": search_result.best_config,
            "validation_score": search_result.recommended_trial.validation_score,
            "full_validation_score": result.full_validation_score,
            "plan_fingerprint": result.plan_fingerprint,
            "temporal_fit_promotion": temporal_promotion,
            "search_training_frames": search_training_frames,
            "screening_rows_used": next(
                (
                    trial.sample_rows
                    for trial in search_result.trials
                    if trial.stage == "screening" and trial.sample_rows is not None
                ),
                screening_rows,
            ),
            "fidelity_promotion_top_k": fidelity_promotion_top_k,
            "fidelity_promotion_training_frames": (fidelity_promotion_training_frames),
            "categorical_refinement_top_k": categorical_refinement_top_k,
            "stage_candidate_budgets": (
                None
                if stage_candidate_budgets is None
                else asdict(stage_candidate_budgets)
            ),
            "stage_counts": stage_counts,
            "peer_context": (
                problem.peer_context.contract()
                if problem.peer_context is not None
                else None
            ),
            "trials": search_result.results.copy(),
            "confirmation_trials": search_result.confirmation_results.copy(),
            "rerank_trials": search_result.rerank_results.copy(),
        }
        return result, problem


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _atomic_json(value: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def run(args: argparse.Namespace) -> int:
    if args.temporal_fit_policy != "fixed" and args.full_training_frames != 1:
        raise SystemExit(
            "--temporal-fit-policy requires --full-training-frames 1 as its "
            "fallback regime"
        )
    key = (args.dataset, args.task)
    generic = args.profile in {"staged-adaptive","staged-bounded","staged-retention",
        "staged-calibrated","staged-diverse","staged-production","published-compat-search"}
    if key not in PUBLISHED_REFERENCES and not generic:
        choices = ", ".join(
            f"{dataset}/{task}" for dataset, task in PUBLISHED_REFERENCES
        )
        raise SystemExit(
            f"No audited recipe for {args.dataset}/{args.task}; use {choices}"
        )
    recipe = _recipe_for(key,generic=generic)
    rerank_nested_depth_safety = (
        args.profile == "staged-diverse"
        if args.rerank_nested_depth_safety is None
        else args.rerank_nested_depth_safety
    )
    fixed_config: kurversc.GraphConfig | None = None
    if args.profile == "fixed-config":
        if not args.fixed_config_json:
            raise SystemExit("--fixed-config-json is required for fixed-config")
        try:
            fixed_payload = json.loads(args.fixed_config_json)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"Invalid --fixed-config-json: {exc}") from exc
        if "feature_families" in fixed_payload:
            fixed_payload["feature_families"] = tuple(fixed_payload["feature_families"])
        fixed_config = kurversc.GraphConfig(**fixed_payload)
    fixed_execution_plan = None
    if args.fixed_cdf_pair and (
        args.profile != "fixed-config"
        or args.fixed_trial_checkpoint is not None
        or args.base_cdf != "default"
    ):
        raise SystemExit(
            "--fixed-cdf-pair requires --profile fixed-config, no frozen plan, "
            "and --base-cdf default"
        )
    if args.fixed_trial_checkpoint is not None:
        if args.profile != "fixed-config":
            raise SystemExit("--fixed-trial-checkpoint requires --profile fixed-config")
        # Search checkpoints are trusted local artifacts; pickle must never be
        # loaded from an untrusted source.
        with args.fixed_trial_checkpoint.open("rb") as handle:
            fixed_trial = pickle.load(handle)["trial"]  # noqa: S301
        # Root-cohort scoping is a materialization optimization, not a change
        # to the frozen feature SQL. Allow it to be enabled when replaying an
        # otherwise identical trusted search checkpoint.
        source_config_matches = fixed_trial.config == fixed_config or (
            fixed_config.root_reachable_materialization
            and not fixed_trial.config.root_reachable_materialization
            and replace(
                fixed_trial.config, root_reachable_materialization=True
            ) == fixed_config
        )
        if not source_config_matches or not fixed_trial.execution_plan:
            raise SystemExit("Frozen trial config/plan does not match --fixed-config-json")
        fixed_execution_plan = copy.deepcopy(fixed_trial.execution_plan)
        if args.frozen_schema_reference is not None:
            with args.frozen_schema_reference.open("rb") as handle:
                reference_trial = pickle.load(handle)["trial"]  # noqa: S301
            columns = tuple(dict.fromkeys((
                *reference_trial.feature_columns,
                *args.frozen_schema_add_column,
            )))
            available = set(fixed_trial.feature_columns)
            missing = sorted(set(columns) - available)
            if missing:
                raise SystemExit(f"Frozen plan cannot provide columns: {missing[:10]}")
            schema = fixed_execution_plan["kurversc_feature_schema"]
            schema["columns"] = columns
            schema["categorical_columns"] = tuple(
                column for column in schema["categorical_columns"] if column in columns
            )
            schema["datetime_columns"] = tuple(
                column for column in schema["datetime_columns"] if column in columns
            )
    elif args.frozen_schema_reference is not None or args.frozen_schema_add_column:
        raise SystemExit("Frozen schema controls require --fixed-trial-checkpoint")

    # These are resource ceilings, not feature/model changes.  They leave ample
    # headroom on the 48 GB local workstation and allow DuckDB to spill.
    KURVERSC_DEFAULTS["duckdb_memory_limit"] = args.duckdb_memory_limit
    KURVERSC_DEFAULTS["duckdb_max_temp_directory_size"] = args.duckdb_temp_limit
    KURVERSC_DEFAULTS["base_distribution_features"] = (
        None if args.base_cdf == "default" else args.base_cdf == "on"
    )
    automatic_estimator_mode = (
        "frame_ensemble"
        if args.profile
        in {
            "published-compat-search",
            "published-lineage-control",
            "staged-production",
            "staged-bounded",
            "staged-retention",
            "staged-calibrated",
            "staged-diverse",
            "staged-adaptive",
            "fixed-config",
        }
        else "pooled"
    )
    KURVERSC_DEFAULTS["temporal_estimator_mode"] = (
        automatic_estimator_mode
        if args.temporal_estimator_mode == "auto"
        else args.temporal_estimator_mode
    )
    AuditWinnerSystem.checkpoint_root = Path(args.checkpoint_root)
    AuditWinnerSystem.profile = args.profile
    AuditWinnerSystem.search_parallelism = args.search_parallelism
    AuditWinnerSystem.root_reachable_materialization = (
        args.root_reachable_materialization
    )
    AuditWinnerSystem.rerank_nested_depth_safety = rerank_nested_depth_safety
    AuditWinnerSystem.fixed_config = fixed_config
    AuditWinnerSystem.fixed_cdf_pair = args.fixed_cdf_pair
    AuditWinnerSystem.fixed_search_training_frames = args.fixed_search_training_frames
    AuditWinnerSystem.full_training_frames = args.full_training_frames
    AuditWinnerSystem.temporal_fit_policy = args.temporal_fit_policy
    AuditWinnerSystem.temporal_fit_max_rows = args.temporal_fit_max_rows
    AuditWinnerSystem.temporal_fit_max_features = args.temporal_fit_max_features
    AuditWinnerSystem.temporal_fit_max_search_seconds = (
        args.temporal_fit_max_search_seconds
    )
    AuditWinnerSystem.fixed_rerank = args.fixed_rerank
    AuditWinnerSystem.fixed_execution_plan = fixed_execution_plan
    AuditWinnerSystem.inner_records = {}
    AuditWinnerSystem.peer_context_mode = (
        None if args.peer_context_mode == "off" else args.peer_context_mode
    )
    AuditWinnerSystem.peer_max_history_rows = args.peer_max_history_rows

    effective_search_training_frames: int | str = (
        "auto"
        if args.profile
        in {"staged-adaptive", "staged-bounded", "staged-retention", "staged-calibrated", "staged-diverse"}
        else args.fixed_search_training_frames
        if args.profile == "fixed-config"
        else recipe.search_training_frames
        if args.profile == "audited-finalist"
        else 1
    )

    print(
        json.dumps(
            {
                "event": "official_test_started",
                "dataset": recipe.dataset,
                "task": recipe.task,
                "published_score": recipe.published_score,
                "higher_is_better": recipe.higher_is_better,
                "search_training_frames": effective_search_training_frames,
                "full_training_frames": args.full_training_frames,
                "profile": args.profile,
                "base_distribution_features": KURVERSC_DEFAULTS[
                    "base_distribution_features"
                ],
                "temporal_estimator_mode": KURVERSC_DEFAULTS["temporal_estimator_mode"],
                "temporal_fit_policy": args.temporal_fit_policy,
                "root_reachable_materialization": args.root_reachable_materialization,
                "peer_context_mode": args.peer_context_mode,
                "peer_max_history_rows": args.peer_max_history_rows,
                "config": (
                    asdict(fixed_config)
                    if args.profile == "fixed-config"
                    else asdict(recipe.config)
                    if args.profile == "audited-finalist"
                    else asdict(PUBLISHED_LINEAGE_CONTROLS[key])
                    if args.profile == "published-lineage-control"
                    else "published-compatible anchor plus modern staged challengers"
                    if args.profile == "staged-production"
                    else "bounded depth-4/cyclic staged policy with finalist-only categorical refinement"
                    if args.profile == "staged-bounded"
                    else "bounded confidence-frontier retention through 50K and full rerank"
                    if args.profile == "staged-retention"
                    else "cost-calibrated depth-4/cyclic policy with structural promotion"
                    if args.profile == "staged-calibrated"
                    else "cost-calibrated policy with shallow family controls"
                    if args.profile == "staged-diverse"
                    else "published-compatible anchor plus adaptive-fidelity modern challengers"
                    if args.profile == "staged-adaptive"
                    else "published-compatible 72-shape exhaustive panel"
                ),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    started = time.perf_counter()
    from kurversc.task_source import CONTEXT_ENV, run_context_system
    runner = run_context_system if os.environ.get(CONTEXT_ENV) else run_system_experiment
    summary = runner(
        AuditWinnerSystem,
        recipe.dataset,
        recipe.task,
        seed=args.seed,
        cache_dir=args.cache_dir,
        cache_predictions=False,
        evaluate_test=True,
    )
    elapsed = time.perf_counter() - started
    test_score = float(summary.result.test_score) if summary.result.test_score is not None else None
    inner = AuditWinnerSystem.inner_records[key]
    selected_config = inner["selected_config"]
    improvement = None if recipe.published_score is None or test_score is None else (
        test_score - recipe.published_score
        if recipe.higher_is_better
        else recipe.published_score - test_score
    )
    row = {
        "dataset": recipe.dataset,
        "task": recipe.task,
        "metric": summary.metric_name,
        "test_score": test_score,
        "published_kurversc_test_score": recipe.published_score,
        "improvement": improvement,
        "beats_published_kurversc": improvement > 0.0 if improvement is not None else None,
        "elapsed_seconds": elapsed,
        "search_training_frames": inner["search_training_frames"],
        "full_training_frames": inner["temporal_fit_promotion"]["training_frames"],
        "screening_rows_used": inner["screening_rows_used"],
        "fidelity_promotion_top_k": inner["fidelity_promotion_top_k"],
        "fidelity_promotion_training_frames": inner[
            "fidelity_promotion_training_frames"
        ],
        "categorical_refinement_top_k": inner["categorical_refinement_top_k"],
        "stage_candidate_budgets": json.dumps(
            inner["stage_candidate_budgets"], sort_keys=True
        ),
        "stage_counts": json.dumps(inner["stage_counts"], sort_keys=True),
        "profile": args.profile,
        "base_distribution_features": KURVERSC_DEFAULTS["base_distribution_features"],
        "fixed_cdf_pair": args.fixed_cdf_pair,
        "rerank_nested_depth_safety": rerank_nested_depth_safety,
        "temporal_estimator_mode": inner["temporal_fit_promotion"]["estimator_mode"],
        "temporal_fit_promotion": json.dumps(
            inner["temporal_fit_promotion"], default=str, sort_keys=True
        ),
        "inner_validation_score": inner["validation_score"],
        "inner_full_validation_score": inner["full_validation_score"],
        "plan_fingerprint": inner["plan_fingerprint"],
        "seed": args.seed,
        "runtime": search_runtime_identity(),
        "peer_context_mode": args.peer_context_mode,
        "peer_context": inner["peer_context"],
        "audit_source": (
            recipe.audit_source
            if args.profile == "audited-finalist"
            else "matched CDF-on/off graph control with full-fidelity rerank; test labels never used for selection"
            if args.fixed_cdf_pair
            else "explicit frozen-config diagnostic; test labels never used for selection"
            if args.profile == "fixed-config"
            else "published 0.1.1 inner-selected control; test labels never used for selection"
            if args.profile == "published-lineage-control"
            else "generic 10K/50K, diverse temporal-fidelity promotion, and full walk-forward rerank"
            if args.profile == "staged-production"
            else "bounded 10K/50K/temporal promotion, finalist-only categorical refinement, and full walk-forward rerank"
            if args.profile == "staged-bounded"
            else "uncertainty-competitive graph retention through bounded 50K and full-fidelity stages"
            if args.profile == "staged-retention"
            else "cost-calibrated screening, structural promotion, and full walk-forward rerank"
            if args.profile == "staged-calibrated"
            else "cost-calibrated screening, shallow family promotion, and bounded seven-finalist rerank"
            if args.profile == "staged-diverse"
            else "adaptive 10K/50K/full staged search with adequate recent-cutoff labels"
            if args.profile == "staged-adaptive"
            else "exhaustive published-compatible 72-shape panel with three-fold rerank"
        ),
        "config": json.dumps(asdict(selected_config), sort_keys=True),
    }
    output = Path(args.output)
    _atomic_csv(pd.DataFrame([row]), output)
    _atomic_json(row, output.with_suffix(".json"))
    # Preserve the native RelArena result schema next to the comparison row.
    _atomic_csv(
        summary_to_dataframe(summary),
        output.with_name(output.stem + "_relarena.csv"),
    )
    _atomic_csv(inner["trials"], output.with_name(output.stem + "_inner_trials.csv"))
    _atomic_csv(
        inner["confirmation_trials"],
        output.with_name(output.stem + "_confirmation_trials.csv"),
    )
    _atomic_csv(
        inner["rerank_trials"],
        output.with_name(output.stem + "_rerank_trials.csv"),
    )
    print(json.dumps({"event": "official_test_completed", **row}), flush=True)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--cache-dir")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--profile",
        choices=(
            "audited-finalist",
            "published-compat-search",
            "published-lineage-control",
            "staged-production",
            "staged-bounded",
            "staged-retention",
            "staged-calibrated",
            "staged-diverse",
            "staged-adaptive",
            "fixed-config",
        ),
        default="audited-finalist",
    )
    parser.add_argument("--search-parallelism", type=int, default=1)
    parser.add_argument("--root-reachable-materialization", action="store_true")
    parser.add_argument(
        "--peer-context-mode",
        choices=("off", "unweighted", "both"),
        default="off",
    )
    parser.add_argument("--peer-max-history-rows", type=int, default=50_000)
    parser.add_argument(
        "--base-cdf",
        choices=("default", "on", "off"),
        default="default",
        help="Force base-family CDF distributions on/off across the whole search.",
    )
    parser.add_argument(
        "--rerank-nested-depth-safety",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--fixed-config-json")
    parser.add_argument(
        "--fixed-cdf-pair",
        action="store_true",
        help="Compare matched CDF-off/on variants of one fixed graph at full fidelity.",
    )
    parser.add_argument("--fixed-search-training-frames", type=int, default=12)
    parser.add_argument("--full-training-frames", type=int, choices=(1, 3, 5), default=1)
    parser.add_argument(
        "--temporal-fit-policy",
        choices=("fixed", "cutoff_density"),
        default="fixed",
        help="Optionally promote the selected frozen plan to a bounded three-cutoff fit.",
    )
    parser.add_argument("--temporal-fit-max-rows", type=int, default=100_000)
    parser.add_argument("--temporal-fit-max-features", type=int, default=5_000)
    parser.add_argument("--temporal-fit-max-search-seconds", type=int, default=7_200)
    parser.add_argument("--fixed-rerank", action="store_true")
    parser.add_argument("--fixed-trial-checkpoint", type=Path)
    parser.add_argument("--frozen-schema-reference", type=Path)
    parser.add_argument("--frozen-schema-add-column", action="append", default=[])
    parser.add_argument(
        "--temporal-estimator-mode",
        choices=("auto", "pooled", "frame_ensemble"),
        default="auto",
    )
    parser.add_argument("--duckdb-memory-limit", default="24GB")
    parser.add_argument("--duckdb-temp-limit", default="100GB")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
