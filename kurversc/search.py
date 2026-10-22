"""Configuration candidates and search results."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from itertools import combinations
from math import hypot, isfinite
import random
from typing import Any, Iterable, Sequence

import pandas as pd


DEFAULT_FAMILY_STAGES: tuple[tuple[str, ...], ...] = (
    ("base",),
    ("base", "temporal"),
    ("base", "temporal", "state"),
    ("base", "temporal", "state", "sequence"),
    ("base", "temporal", "state", "sequence", "conditional"),
    (
        "base",
        "temporal",
        "state",
        "sequence",
        "conditional",
        "episode",
        "context",
    ),
)

# The candidate family order used by the initially published RelArena recipe.
# Keep this separate from ``DEFAULT_FAMILY_STAGES``: the modern families are
# promoted as challengers only after the stable reference search has produced
# evidence, rather than multiplying the complete power set.
PUBLISHED_FAMILY_STAGES: tuple[tuple[str, ...], ...] = (
    ("base",),
    ("base", "temporal"),
    ("base", "temporal", "sequence"),
    ("base", "temporal", "sequence", "conditional"),
    ("base", "temporal", "sequence", "conditional", "episode"),
)

# A compact, deterministic panel for production search.  Single-family controls
# make individual signal sources attributable, while the two interaction shapes
# retain the event-sequence and state combinations that cannot be recovered by
# evaluating each family in isolation.  Both annotation policies and depths one
# and two are evaluated for every shape (10 * 2 * 2 = 40 candidates).
COMPACT_MODERN_FAMILY_TEMPLATES: tuple[tuple[str, ...], ...] = (
    ("base",),
    ("base", "temporal"),
    ("base", "state"),
    ("base", "sequence"),
    ("base", "conditional"),
    ("base", "episode"),
    ("base", "context"),
    ("base", "temporal", "state"),
    ("base", "temporal", "state", "sequence", "episode"),
    ("base", "temporal", "sequence", "conditional", "episode", "context"),
)

# Target history is an orthogonal capability rather than a feature family.
# Pair it only with representative controls and interaction shapes so the wide
# panel can measure its incremental value without doubling the entire design.
WIDE_TARGET_HISTORY_TEMPLATES: tuple[tuple[str, ...], ...] = (
    ("base",),
    ("base", "conditional"),
    ("base", "context"),
    ("base", "temporal", "state", "sequence", "episode"),
    ("base", "temporal", "sequence", "conditional", "episode", "context"),
)

_CANONICAL_FAMILY_ORDER = DEFAULT_FAMILY_STAGES[-1]


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
    categorical_cardinality_threshold: int = 20
    categorical_top_k: int = 5
    high_cardinality_policy: str = "skip"
    categorical_low_cardinality_policy: str = "top_k"
    text_categorical_fallback: bool = False
    include_target_history: bool = True
    max_base_table_occurrences: int = 1
    published_cutoff_compatibility: bool = False
    categorical_tie_break: str = "value"
    categorical_tie_break_seed: int = 0
    legacy_training_row_order: bool = False
    # ``None`` keeps the compatibility-derived default; booleans support
    # matched CDF ablations without adding another searched family.
    base_distribution_features: bool | None = None
    # Full-data execution controls. They do not alter the feature plan and
    # remain off unless a memory-bounded run explicitly requests them.
    materialization_batch_rows: int | None = None
    root_reachable_materialization: bool = False

    def __post_init__(self) -> None:
        if self.depth < 1:
            raise ValueError("depth must be at least 1")
        if self.materialization_batch_rows is not None and self.materialization_batch_rows < 1:
            raise ValueError("materialization_batch_rows must be positive or None")
        if not self.feature_families or self.feature_families[0] != "base":
            raise ValueError("feature_families must start with 'base'")
        if self.root_reachable_materialization and "context" in self.feature_families:
            raise ValueError(
                "root_reachable_materialization cannot be combined with "
                "cross-root context features"
            )
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
        if self.categorical_cardinality_threshold < 1:
            raise ValueError("categorical_cardinality_threshold must be positive")
        if self.categorical_top_k < 0:
            raise ValueError("categorical_top_k must be non-negative")
        if self.high_cardinality_policy not in {"skip", "top_k"}:
            raise ValueError("high_cardinality_policy must be either 'skip' or 'top_k'")
        if self.categorical_low_cardinality_policy not in {"top_k", "all"}:
            raise ValueError(
                "categorical_low_cardinality_policy must be either 'top_k' or 'all'"
            )
        if self.max_base_table_occurrences < 1:
            raise ValueError("max_base_table_occurrences must be positive")
        if self.categorical_tie_break not in {
            "value",
            "appearance",
            "hash",
            "legacy",
        }:
            raise ValueError(
                "categorical_tie_break must be 'value', 'appearance', 'hash', "
                "or 'legacy'"
            )
        if self.include_target_history and self.published_cutoff_compatibility:
            raise ValueError(
                "published_cutoff_compatibility cannot be combined with "
                "include_target_history because the historical target join "
                "must remain strictly before the exact cutoff"
            )

    @property
    def complexity(self) -> float:
        annotation_factor = 1.25 if self.auto_annotate_features else 1.0
        topology_factor = 1.0 + 0.25 * (self.max_base_table_occurrences - 1)
        return float(
            self.depth
            * len(self.feature_families)
            * annotation_factor
            * topology_factor
        )

    def graphreduce_kwargs(self) -> dict[str, Any]:
        return {
            "auto_features": True,
            "auto_feature_hops_back": self.depth,
            "auto_feature_hops_front": 0,
            "auto_text_features": self.auto_text_features,
            "auto_annotate_max_text_columns": self.auto_annotate_max_text_columns,
            "categorical_cardinality_threshold": (
                self.categorical_cardinality_threshold
            ),
            "categorical_top_k": self.categorical_top_k,
            "high_cardinality_policy": self.high_cardinality_policy,
            "categorical_low_cardinality_policy": (
                self.categorical_low_cardinality_policy
            ),
            "categorical_tie_break": self.categorical_tie_break,
            "categorical_tie_break_seed": self.categorical_tie_break_seed,
            "text_categorical_fallback": self.text_categorical_fallback,
        }


@dataclass(frozen=True)
class SearchStageBudgets:
    """Hard ceilings on the number of candidates evaluated at each stage.

    ``None`` preserves the stage's ordinary candidate count.  A zero budget
    disables an optional stage; screening must retain at least one candidate.
    These limits bound *evaluations*, unlike the existing ``*_top_k`` knobs,
    which primarily control promotion from one fidelity to the next.
    """

    screening: int | None = None
    challenger: int | None = None
    combined_challenger: int | None = None
    confirmation: int | None = None
    fidelity_promotion: int | None = None
    categorical_refinement: int | None = None
    rerank: int | None = None

    def __post_init__(self) -> None:
        for field_name, value in asdict(self).items():
            if value is not None and value < 0:
                raise ValueError(
                    f"Search-stage budget {field_name!r} must be non-negative or None"
                )
        if self.screening == 0:
            raise ValueError("Search-stage screening budget must be positive or None")


def graph_config_sort_key(config: GraphConfig) -> tuple[Any, ...]:
    """Return a stable total ordering for configuration tie-breaks."""

    return (
        config.feature_families,
        config.depth,
        config.auto_annotate_features,
        -1
        if config.feature_family_max_columns is None
        else config.feature_family_max_columns,
        (
            -1
            if config.feature_family_max_features_per_column is None
            else config.feature_family_max_features_per_column
        ),
        (
            -1
            if config.feature_propagation_max_functions_per_column is None
            else config.feature_propagation_max_functions_per_column
        ),
        config.auto_text_features,
        (
            -1
            if config.auto_annotate_max_text_columns is None
            else config.auto_annotate_max_text_columns
        ),
        config.categorical_cardinality_threshold,
        config.categorical_top_k,
        config.high_cardinality_policy,
        config.categorical_low_cardinality_policy,
        config.text_categorical_fallback,
        config.include_target_history,
        -1
        if config.base_distribution_features is None
        else int(config.base_distribution_features),
        config.max_base_table_occurrences,
        config.published_cutoff_compatibility,
        config.categorical_tie_break,
        config.categorical_tie_break_seed,
        config.legacy_training_row_order,
    )


def incremental_configs(
    *,
    max_depth: int = 4,
    feature_family_stages: Sequence[Sequence[str]] = DEFAULT_FAMILY_STAGES,
    auto_annotate_options: Sequence[bool] = (True, False),
    feature_family_max_columns: int | None = 4,
    feature_family_max_column_options: Sequence[int | None] | None = None,
    feature_family_max_features_per_column: int | None = 32,
    feature_propagation_max_functions_per_column: int | None = 1,
    include_target_history: bool = True,
    target_history_options: Sequence[bool] | None = None,
    max_base_table_occurrence_options: Sequence[int] | None = None,
    limit_expensive_family_depth: bool = True,
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
    history_options = (
        (bool(include_target_history),)
        if target_history_options is None
        else tuple(dict.fromkeys(bool(value) for value in target_history_options))
    )
    if not history_options:
        raise ValueError("target_history_options must not be empty")
    topology_options = (
        (1,)
        if max_base_table_occurrence_options is None
        else tuple(
            dict.fromkeys(int(value) for value in max_base_table_occurrence_options)
        )
    )
    if not topology_options or any(value < 1 for value in topology_options):
        raise ValueError(
            "max_base_table_occurrence_options must contain positive integers"
        )
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
            stage_max_depth = max_depth
            if limit_expensive_family_depth:
                stage_max_depth = (
                    max_depth
                    if set(families).issubset({"base", "temporal", "state", "sequence"})
                    else min(max_depth, 2)
                )
            # Evaluate both annotation policies at a shallow depth before
            # deciding which one is worth propagating farther.
            for depth in range(1, stage_max_depth + 1):
                for topology_limit in topology_options:
                    for history_enabled in history_options:
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
                                    include_target_history=history_enabled,
                                    max_base_table_occurrences=topology_limit,
                                )
                            )
    return tuple(configs)


def published_compatible_configs(
    *,
    auto_annotate_options: Sequence[bool] = (True, False),
    feature_family_max_columns: int | None = 4,
    feature_family_max_column_options: Sequence[int | None] | None = None,
    feature_family_max_features_per_column: int | None = 32,
    feature_propagation_max_functions_per_column: int | None = 1,
    target_history_options: Sequence[bool] | None = None,
    max_base_table_occurrence_options: Sequence[int] | None = None,
    limit_expensive_family_depth: bool = True,
    legacy_plan_compatibility: bool = True,
) -> tuple[GraphConfig, ...]:
    """Return the 72-shape candidate lattice from the published recipe.

    The anchor deliberately excludes automatic target history. Data-window,
    column-ranking, and cardinality compatibility remain independent fit
    options so callers can perform controlled ablations.
    """

    configs = incremental_configs(
        max_depth=3,
        feature_family_stages=PUBLISHED_FAMILY_STAGES,
        auto_annotate_options=auto_annotate_options,
        feature_family_max_columns=feature_family_max_columns,
        feature_family_max_column_options=feature_family_max_column_options,
        feature_family_max_features_per_column=(feature_family_max_features_per_column),
        feature_propagation_max_functions_per_column=(
            feature_propagation_max_functions_per_column
        ),
        include_target_history=False,
        target_history_options=target_history_options,
        max_base_table_occurrence_options=max_base_table_occurrence_options,
        limit_expensive_family_depth=limit_expensive_family_depth,
    )
    return tuple(
        replace(
            config,
            high_cardinality_policy="top_k",
            categorical_low_cardinality_policy="all",
            text_categorical_fallback=True,
            published_cutoff_compatibility=not config.include_target_history,
            categorical_tie_break=(
                "legacy"
                if legacy_plan_compatibility and not config.include_target_history
                else "value"
            ),
            legacy_training_row_order=(
                legacy_plan_compatibility and not config.include_target_history
            ),
        )
        for config in configs
    )


def staged_challenger_configs(
    parents: Sequence[GraphConfig],
    *,
    additions: Sequence[Sequence[str]] = (("state",), ("context",)),
    depths: Sequence[int] = (1, 2),
    include_target_history: bool | None = None,
) -> tuple[GraphConfig, ...]:
    """Add modern feature families to promoted reference configurations.

    Annotation and feature-budget controls are inherited from each parent.
    Modern families always use native cutoff semantics instead of inheriting
    the release-only cutoff compatibility switch. Families are canonicalized
    so the result is stable across parent and worker completion order.
    Duplicate challengers are removed deterministically.
    """

    resolved_depths = tuple(dict.fromkeys(int(depth) for depth in depths))
    if not resolved_depths or any(depth < 1 for depth in resolved_depths):
        raise ValueError("depths must contain positive values")
    resolved_additions = tuple(
        tuple(dict.fromkeys(str(family) for family in addition))
        for addition in additions
    )
    if not resolved_additions or any(not addition for addition in resolved_additions):
        raise ValueError("additions must contain non-empty family groups")

    family_rank = {
        family: index for index, family in enumerate(_CANONICAL_FAMILY_ORDER)
    }
    challengers: list[GraphConfig] = []
    for parent in parents:
        for addition in resolved_additions:
            families = tuple(dict.fromkeys((*parent.feature_families, *addition)))
            families = tuple(
                sorted(
                    families,
                    key=lambda family: (
                        family_rank.get(family, len(family_rank)),
                        family,
                    ),
                )
            )
            if not families or families[0] != "base":
                raise ValueError("staged challenger families must retain base first")
            for depth in resolved_depths:
                resolved_history = (
                    parent.include_target_history
                    if include_target_history is None
                    else include_target_history
                )
                challengers.append(
                    replace(
                        parent,
                        feature_families=families,
                        depth=depth,
                        # Cross-root context needs the full source population.
                        # A root-scoped parent may still seed a context
                        # challenger, but that execution optimization cannot
                        # be inherited by the new feature family.
                        root_reachable_materialization=(
                            parent.root_reachable_materialization
                            and "context" not in families
                        ),
                        include_target_history=resolved_history,
                        published_cutoff_compatibility=False,
                        categorical_tie_break="value",
                        categorical_tie_break_seed=0,
                        legacy_training_row_order=False,
                    )
                )
    return tuple(dict.fromkeys(challengers))


def staged_modern_seed_configs(
    controls: Sequence[GraphConfig],
    *,
    depths: Sequence[int] = (1, 2),
) -> tuple[GraphConfig, ...]:
    """Return sparse native controls that a forward-only stage can miss.

    Promoting only the strongest legacy graphs cannot discover a modern winner
    whose useful representation is *smaller* than every promoted parent (for
    example ``base + state``).  Build matched state, context, interaction, and
    target-history seeds from the legacy base controls under current cutoff and
    low-cardinality semantics.  The design is schema-generic and deterministic;
    it contains no dataset- or task-specific configuration.
    """

    resolved_depths = tuple(dict.fromkeys(int(depth) for depth in depths))
    if not resolved_depths or any(depth < 1 for depth in resolved_depths):
        raise ValueError("depths must contain positive values")

    base_controls = tuple(
        dict.fromkeys(
            replace(
                control,
                feature_families=("base",),
                depth=1,
                categorical_low_cardinality_policy="top_k",
                include_target_history=False,
                published_cutoff_compatibility=False,
                categorical_tie_break="value",
                categorical_tie_break_seed=0,
                legacy_training_row_order=False,
            )
            for control in controls
        )
    )
    seeds: list[GraphConfig] = []
    for control in base_controls:
        seeds.extend(
            staged_challenger_configs(
                (control,),
                additions=(("state",), ("context",), ("state", "context")),
                depths=resolved_depths,
                include_target_history=False,
            )
        )
        seeds.append(replace(control, include_target_history=True))
        seeds.extend(
            staged_challenger_configs(
                (control,),
                additions=(("state",), ("context",)),
                depths=(1,),
                include_target_history=True,
            )
        )
    return tuple(dict.fromkeys(seeds))


def state_focused_configs(
    *,
    auto_annotate_options: Sequence[bool] = (True, False),
    feature_family_max_columns: int | None = 4,
    feature_family_max_features_per_column: int | None = 32,
    feature_propagation_max_functions_per_column: int | None = 1,
) -> tuple[GraphConfig, ...]:
    """Return a compact panel that isolates ordered state signal.

    The panel retains the base control, tests state at one and two hops, and
    checks whether calendar windows complement event-relative state. It is
    useful for fast diagnosis and constrained deployments; the ordinary
    default lattice remains the broader general-purpose search.
    """

    annotations = tuple(dict.fromkeys(bool(value) for value in auto_annotate_options))
    if not annotations:
        raise ValueError("auto_annotate_options must not be empty")
    shapes = (
        (("base",), 1),
        (("base", "state"), 1),
        (("base", "state"), 2),
        (("base", "temporal", "state"), 1),
    )
    return tuple(
        GraphConfig(
            feature_families=families,
            depth=depth,
            auto_annotate_features=annotate,
            feature_family_max_columns=feature_family_max_columns,
            feature_family_max_features_per_column=(
                feature_family_max_features_per_column
            ),
            feature_propagation_max_functions_per_column=(
                feature_propagation_max_functions_per_column
            ),
        )
        for families, depth in shapes
        for annotate in annotations
    )


def compact_modern_configs(
    *,
    auto_annotate_options: Sequence[bool] = (True, False),
    depths: Sequence[int] = (1, 2),
    feature_family_max_columns: int | None = 4,
    feature_family_max_features_per_column: int | None = 32,
    feature_propagation_max_functions_per_column: int | None = 1,
    include_target_history: bool = True,
) -> tuple[GraphConfig, ...]:
    """Return a small panel with explicit controls and modern interactions.

    Unlike a random slice of the full power set, this design guarantees that
    ``state`` and its sequence/episode interaction are evaluated under both
    annotation policies at useful shallow depths.  The panel is intended for
    full-data, multi-cutoff scoring where a 200+ candidate lattice is wasteful.
    """

    annotations = tuple(dict.fromkeys(bool(value) for value in auto_annotate_options))
    resolved_depths = tuple(dict.fromkeys(int(depth) for depth in depths))
    if not annotations:
        raise ValueError("auto_annotate_options must not be empty")
    if not resolved_depths or any(depth < 1 for depth in resolved_depths):
        raise ValueError("depths must contain positive values")
    return tuple(
        GraphConfig(
            feature_families=families,
            depth=depth,
            auto_annotate_features=annotate,
            feature_family_max_columns=feature_family_max_columns,
            feature_family_max_features_per_column=(
                feature_family_max_features_per_column
            ),
            feature_propagation_max_functions_per_column=(
                feature_propagation_max_functions_per_column
            ),
            include_target_history=include_target_history,
        )
        for families in COMPACT_MODERN_FAMILY_TEMPLATES
        for depth in resolved_depths
        for annotate in annotations
    )


def wide_modern_configs(
    *,
    auto_annotate_options: Sequence[bool] = (True, False),
    feature_family_max_columns: int | None = 4,
    feature_family_max_features_per_column: int | None = 32,
    feature_propagation_max_functions_per_column: int | None = 1,
) -> tuple[GraphConfig, ...]:
    """Return a depth-three modern panel with explicit history ablations.

    Sixty structural candidates cover every compact template at depths one
    through three with target history disabled.  Twenty matched candidates
    then enable target history for representative depth-one/two controls and
    interactions.  This keeps the design at 80 candidates instead of doubling
    the full panel or returning to the 200+ configuration power set.
    """

    structural = compact_modern_configs(
        auto_annotate_options=auto_annotate_options,
        depths=(1, 2, 3),
        feature_family_max_columns=feature_family_max_columns,
        feature_family_max_features_per_column=(feature_family_max_features_per_column),
        feature_propagation_max_functions_per_column=(
            feature_propagation_max_functions_per_column
        ),
        include_target_history=False,
    )
    annotations = tuple(dict.fromkeys(bool(value) for value in auto_annotate_options))
    history = tuple(
        GraphConfig(
            feature_families=families,
            depth=depth,
            auto_annotate_features=annotate,
            feature_family_max_columns=feature_family_max_columns,
            feature_family_max_features_per_column=(
                feature_family_max_features_per_column
            ),
            feature_propagation_max_functions_per_column=(
                feature_propagation_max_functions_per_column
            ),
            include_target_history=True,
        )
        for families in WIDE_TARGET_HISTORY_TEMPLATES
        for depth in (1, 2)
        for annotate in annotations
    )
    return tuple(dict.fromkeys((*structural, *history)))


def campaign_modern_configs(
    *,
    max_candidates: int = 32,
    random_state: int = 42,
    auto_annotate_options: Sequence[bool] = (True, False),
    feature_family_max_columns: int | None = 4,
    feature_family_max_features_per_column: int | None = 32,
    feature_propagation_max_functions_per_column: int | None = 1,
) -> tuple[GraphConfig, ...]:
    """Return a balanced, resource-conscious subset of the modern panel.

    The full wide panel is useful for focused ablations, but evaluating all 80
    candidates over every RelArena task is unnecessarily expensive.  This
    profile uses the deterministic collection design to retain depth-three
    controls, every modern feature family, both annotation policies, and
    representative target-history interactions in a smaller campaign panel.
    """

    if max_candidates < 1:
        raise ValueError("max_candidates must be positive")
    return diverse_collection_configs(
        wide_modern_configs(
            auto_annotate_options=auto_annotate_options,
            feature_family_max_columns=feature_family_max_columns,
            feature_family_max_features_per_column=(
                feature_family_max_features_per_column
            ),
            feature_propagation_max_functions_per_column=(
                feature_propagation_max_functions_per_column
            ),
        ),
        max_candidates=max_candidates,
        random_state=random_state,
    )


def diverse_collection_configs(
    configs: Sequence[GraphConfig],
    *,
    max_candidates: int | None = 24,
    random_state: int = 42,
) -> tuple[GraphConfig, ...]:
    """Select a balanced full-fidelity teacher design from a valid lattice.

    Baselines span source-column budgets first, then each available feature
    family receives both annotation policies, useful depth variants are added,
    and the remaining capacity is filled round-robin across structural strata.
    Passing ``None`` retains the complete supplied lattice.
    """

    candidates = tuple(dict.fromkeys(configs))
    if max_candidates is None or len(candidates) <= max_candidates:
        return candidates
    if max_candidates < 1:
        raise ValueError("max_candidates must be positive or None")

    selected: list[GraphConfig] = []
    selected_set: set[GraphConfig] = set()
    rng = random.Random(random_state)

    def add(candidate: GraphConfig | None) -> None:
        if (
            candidate is not None
            and len(selected) < max_candidates
            and candidate not in selected_set
        ):
            selected.append(candidate)
            selected_set.add(candidate)

    def first(**conditions: Any) -> GraphConfig | None:
        return next(
            (
                candidate
                for candidate in candidates
                if all(
                    getattr(candidate, key) == value
                    for key, value in conditions.items()
                )
            ),
            None,
        )

    budgets = tuple(
        dict.fromkeys(candidate.feature_family_max_columns for candidate in candidates)
    )
    default_budget = 4 if 4 in budgets else budgets[0]
    for budget in budgets:
        add(
            first(
                feature_families=("base",),
                depth=1,
                auto_annotate_features=True,
                feature_family_max_columns=budget,
            )
        )
    add(
        first(
            feature_families=("base",),
            depth=1,
            auto_annotate_features=False,
            feature_family_max_columns=default_budget,
        )
    )

    # Reserve orthogonal search dimensions before filling family/depth
    # strata.  This is especially important when the caller uses the helper
    # as a hard stage budget: cyclic topology and target history must not
    # disappear merely because their candidates occur late in the lattice.
    for topology_limit in tuple(
        dict.fromkeys(candidate.max_base_table_occurrences for candidate in candidates)
    ):
        add(
            next(
                (
                    candidate
                    for candidate in candidates
                    if candidate.max_base_table_occurrences == topology_limit
                ),
                None,
            )
        )
    for history_enabled in tuple(
        dict.fromkeys(candidate.include_target_history for candidate in candidates)
    ):
        add(
            next(
                (
                    candidate
                    for candidate in candidates
                    if candidate.include_target_history == history_enabled
                ),
                None,
            )
        )
    for depth in tuple(dict.fromkeys(candidate.depth for candidate in candidates)):
        add(
            next(
                (candidate for candidate in candidates if candidate.depth == depth),
                None,
            )
        )

    optional_families = tuple(
        dict.fromkeys(
            family
            for candidate in candidates
            for family in candidate.feature_families
            if family != "base"
        )
    )
    annotation_offset = rng.randrange(2)
    for index, family in enumerate(optional_families):
        annotate = bool((index + annotation_offset) % 2)
        add(
            first(
                feature_families=("base", family),
                depth=1,
                auto_annotate_features=annotate,
                feature_family_max_columns=default_budget,
            )
        )
    for family in optional_families:
        add(
            first(
                feature_families=("base", family),
                depth=2,
                auto_annotate_features=False,
                feature_family_max_columns=default_budget,
            )
        )
    for depth in (3, 4):
        for families in (
            ("base",),
            ("base", "state"),
        ):
            add(
                first(
                    feature_families=families,
                    depth=depth,
                    auto_annotate_features=True,
                    feature_family_max_columns=default_budget,
                )
            )

    family_counts = sorted(
        {
            len(candidate.feature_families)
            for candidate in candidates
            if len(candidate.feature_families) >= 3
        }
    )
    for family_count in family_counts:
        matching = [
            candidate
            for candidate in candidates
            if len(candidate.feature_families) == family_count
            and candidate.feature_family_max_columns == default_budget
        ]
        rng.shuffle(matching)
        add(next((item for item in matching if item not in selected_set), None))

    remaining = [candidate for candidate in candidates if candidate not in selected_set]
    buckets: dict[tuple[int, int, bool, int | None], list[GraphConfig]] = {}
    for candidate in remaining:
        key = (
            len(candidate.feature_families),
            candidate.depth,
            candidate.auto_annotate_features,
            candidate.feature_family_max_columns,
        )
        buckets.setdefault(key, []).append(candidate)
    for bucket in buckets.values():
        rng.shuffle(bucket)
    ordered_keys = list(buckets)
    rng.shuffle(ordered_keys)
    while len(selected) < max_candidates and ordered_keys:
        next_keys = []
        for key in ordered_keys:
            bucket = buckets[key]
            add(bucket.pop(0) if bucket else None)
            if bucket:
                next_keys.append(key)
            if len(selected) == max_candidates:
                break
        ordered_keys = next_keys
    return tuple(selected)


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
                        graph_config_sort_key(trial.config),
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
                and parent.config.categorical_cardinality_threshold
                == config.categorical_cardinality_threshold
                and parent.config.categorical_top_k == config.categorical_top_k
                and parent.config.feature_family_max_features_per_column
                == config.feature_family_max_features_per_column
                and parent.config.feature_propagation_max_functions_per_column
                == config.feature_propagation_max_functions_per_column
                and parent.config.high_cardinality_policy
                == config.high_cardinality_policy
                and parent.config.categorical_low_cardinality_policy
                == config.categorical_low_cardinality_policy
                and parent.config.text_categorical_fallback
                == config.text_categorical_fallback
                and parent.config.include_target_history
                == config.include_target_history
                and parent.config.max_base_table_occurrences
                == config.max_base_table_occurrences
                and parent.config.published_cutoff_compatibility
                == config.published_cutoff_compatibility
                and parent.config.categorical_tie_break == config.categorical_tie_break
                and parent.config.categorical_tie_break_seed
                == config.categorical_tie_break_seed
                and parent.config.legacy_training_row_order
                == config.legacy_training_row_order
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
            graph_config_sort_key(trial.config),
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
            and parent.categorical_cardinality_threshold
            == config.categorical_cardinality_threshold
            and parent.categorical_top_k == config.categorical_top_k
            and parent.high_cardinality_policy == config.high_cardinality_policy
            and parent.categorical_low_cardinality_policy
            == config.categorical_low_cardinality_policy
            and parent.text_categorical_fallback == config.text_categorical_fallback
            and parent.include_target_history == config.include_target_history
            and parent.max_base_table_occurrences == config.max_base_table_occurrences
            and parent.published_cutoff_compatibility
            == config.published_cutoff_compatibility
            and parent.categorical_tie_break == config.categorical_tie_break
            and parent.categorical_tie_break_seed == config.categorical_tie_break_seed
            and parent.legacy_training_row_order == config.legacy_training_row_order
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
    promoted to depth 2. Depths 3 and higher are considerably more expensive,
    so each requires its immediate parent to exist and to improve meaningfully
    over the matching depth-1 frame. Explicit ``graph_configs`` bypass this
    default-search guard.
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
        and trial.config.auto_text_features == config.auto_text_features
        and trial.config.auto_annotate_max_text_columns
        == config.auto_annotate_max_text_columns
        and trial.config.categorical_cardinality_threshold
        == config.categorical_cardinality_threshold
        and trial.config.categorical_top_k == config.categorical_top_k
        and trial.config.high_cardinality_policy == config.high_cardinality_policy
        and trial.config.categorical_low_cardinality_policy
        == config.categorical_low_cardinality_policy
        and trial.config.text_categorical_fallback == config.text_categorical_fallback
        and trial.config.include_target_history == config.include_target_history
        and trial.config.max_base_table_occurrences == config.max_base_table_occurrences
        and trial.config.published_cutoff_compatibility
        == config.published_cutoff_compatibility
        and trial.config.categorical_tie_break == config.categorical_tie_break
        and trial.config.categorical_tie_break_seed == config.categorical_tie_break_seed
        and trial.config.legacy_training_row_order == config.legacy_training_row_order
        and trial.config.depth < config.depth
    ]
    immediate_parents = sorted(
        (trial for trial in matching if trial.config.depth == config.depth - 1),
        key=lambda trial: (
            -trial.objective_score,
            trial.feature_count,
            graph_config_sort_key(trial.config),
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
    if parent.metric in {"roc_auc", "accuracy"}:
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
    cutoff_labels: tuple[str, ...] = ()
    cutoff_scores: tuple[float | None, ...] = ()
    cutoff_standard_errors: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        self.cutoff_labels = tuple(self.cutoff_labels)
        self.cutoff_scores = tuple(self.cutoff_scores)
        self.cutoff_standard_errors = tuple(self.cutoff_standard_errors)
        if len(self.cutoff_labels) != len(self.cutoff_scores):
            raise ValueError("cutoff_labels and cutoff_scores must have equal length")
        if not self.cutoff_standard_errors and self.cutoff_scores:
            self.cutoff_standard_errors = (0.0,) * len(self.cutoff_scores)
        if len(self.cutoff_standard_errors) != len(self.cutoff_scores):
            raise ValueError(
                "cutoff_standard_errors and cutoff_scores must have equal length"
            )

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
            "cutoff_labels": self.cutoff_labels,
            "cutoff_scores": self.cutoff_scores,
            "cutoff_standard_errors": self.cutoff_standard_errors,
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
            graph_config_sort_key(trial.config),
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
                if trial.config.include_target_history
                != raw_winner.config.include_target_history
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
                if trial.config.max_base_table_occurrences
                != raw_winner.config.max_base_table_occurrences
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


def diverse_fidelity_trials(
    successful: Sequence[Trial],
    *,
    top_k: int,
    complexity_recommendation: Trial | None = None,
    structural_first: bool = False,
    shallow_sparse_controls: int = 0,
) -> tuple[Trial, ...]:
    """Select candidates whose cheap ranking deserves higher-fidelity evidence.

    A recent-cutoff screen is deliberately inexpensive, but it is not a safe
    global elimination rule: a feature family can require several historical
    training frames before its value becomes visible.  Promotion therefore
    retains the raw and complexity-aware leaders, the highest-order family
    interactions under both annotation policies, and sparse representatives
    of every available family.  Orthogonal target-history and cyclic-topology
    choices are retained as well before the remaining capacity is filled by
    score.  With ``structural_first``, base-only controls and orthogonal graph
    axes are reserved before sparse family controls can exhaust a small budget.
    ``shallow_sparse_controls`` reserves the strongest distinct depth-minimal
    single-family controls before depth variants of the same base lineage can
    consume the remaining slots.
    Keeping an interaction control before its main effects matters:
    low-fidelity screens cannot assume that useful relational families obey
    strong heredity.

    The policy is entirely structural.  It never inspects table, dataset, task,
    or feature names, and its ordering is deterministic.
    """

    if top_k < 0:
        raise ValueError("top_k must be non-negative")
    if shallow_sparse_controls < 0:
        raise ValueError("shallow_sparse_controls must be non-negative")
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
            graph_config_sort_key(trial.config),
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

    def best_matching(predicate: Any) -> Trial | None:
        return next(
            (
                trial
                for trial in ranked
                if trial.config not in selected_configs and predicate(trial.config)
            ),
            None,
        )

    add(ranked[0])
    if not structural_first or shallow_sparse_controls == 0:
        add(complexity_recommendation)

    present_families = {
        family
        for trial in ranked
        for family in trial.config.feature_families
        if family != "base"
    }
    optional_families = tuple(
        family for family in _CANONICAL_FAMILY_ORDER if family in present_families
    )
    annotation_options = tuple(
        option
        for option in (False, True)
        if any(trial.config.auto_annotate_features == option for trial in ranked)
    )

    interaction_orders = sorted(
        {
            len(
                tuple(
                    family
                    for family in trial.config.feature_families
                    if family != "base"
                )
            )
            for trial in ranked
            if len(trial.config.feature_families) > 2
        },
        reverse=True,
    )
    if structural_first:
        # A low-ranked shallow base-only graph is an important full-fidelity
        # control.  A deeper base graph can screen better while changing the
        # topology and feature set; it cannot stand in for the shallow control.
        # Protect both annotation policies before the width quotas fill.
        for annotate in annotation_options:
            base_depths = (
                trial.config.depth
                for trial in ranked
                if trial.config.feature_families == ("base",)
                and trial.config.auto_annotate_features == annotate
            )
            shallowest_depth = min(base_depths, default=None)
            add(
                best_matching(
                    lambda config, annotate=annotate, depth=shallowest_depth: (
                        config.feature_families == ("base",)
                        and config.auto_annotate_features == annotate
                        and config.depth == depth
                    )
                )
            )
        if shallow_sparse_controls:
            sparse_controls: list[Trial] = []
            for family in optional_families:
                family_trials = [
                    trial
                    for trial in ranked
                    if trial.config.feature_families == ("base", family)
                ]
                if not family_trials:
                    continue
                shallowest_depth = min(trial.config.depth for trial in family_trials)
                sparse_controls.append(
                    next(
                        trial
                        for trial in family_trials
                        if trial.config.depth == shallowest_depth
                    )
                )
            sparse_controls.sort(
                key=lambda trial: (
                    -trial.objective_score,
                    trial.feature_count,
                    trial.config.complexity,
                    graph_config_sort_key(trial.config),
                )
            )
            for trial in sparse_controls[:shallow_sparse_controls]:
                add(trial)
            add(complexity_recommendation)
    # Reserve the widest interaction lineage first.  Otherwise a modest top-k
    # can be exhausted by individually attributable main-effect controls even
    # when the useful signal only appears when several families are combined.
    if interaction_orders:
        widest_order = interaction_orders[0]
        for annotate in annotation_options:
            add(
                best_matching(
                    lambda config, order=widest_order, annotate=annotate: (
                        len(
                            tuple(
                                family
                                for family in config.feature_families
                                if family != "base"
                            )
                        )
                        == order
                        and config.auto_annotate_features == annotate
                    )
                )
            )

    def add_structural_controls() -> None:
        raw = ranked[0].config
        add(
            best_matching(
                lambda config: (
                    config.include_target_history != raw.include_target_history
                )
            )
        )
        add(
            best_matching(
                lambda config: (
                    config.max_base_table_occurrences != raw.max_base_table_occurrences
                )
            )
        )
        depths = sorted({trial.config.depth for trial in ranked})
        # Cover the shallow and deepest alternatives first.  Intermediate
        # depths follow if the promotion budget has room.
        depth_order = tuple(dict.fromkeys((depths[0], depths[-1], *depths)))
        for depth in depth_order:
            if depth != raw.depth:
                add(best_matching(lambda config, depth=depth: config.depth == depth))

    if structural_first:
        add_structural_controls()
        for order in interaction_orders[1:]:
            for annotate in annotation_options:
                add(
                    best_matching(
                        lambda config, order=order, annotate=annotate: (
                            len(config.feature_families) - 1 == order
                            and config.auto_annotate_features == annotate
                        )
                    )
                )

    for family in optional_families:
        for annotate in annotation_options:
            # Prefer the attributable sparse control.  If capability pruning
            # made it unavailable, retain the strongest graph containing the
            # family rather than silently dropping that search dimension.
            sparse = best_matching(
                lambda config, family=family, annotate=annotate: (
                    config.feature_families == ("base", family)
                    and config.auto_annotate_features == annotate
                )
            )
            add(
                sparse
                or best_matching(
                    lambda config, family=family, annotate=annotate: (
                        family in config.feature_families
                        and config.auto_annotate_features == annotate
                    )
                )
            )

    if not structural_first:
        for order in interaction_orders[1:]:
            for annotate in annotation_options:
                add(
                    best_matching(
                        lambda config, order=order, annotate=annotate: (
                            len(
                                tuple(
                                    family
                                    for family in config.feature_families
                                    if family != "base"
                                )
                            )
                            == order
                            and config.auto_annotate_features == annotate
                        )
                    )
                )
        raw = ranked[0].config
        add(
            best_matching(
                lambda config: (
                    config.include_target_history != raw.include_target_history
                )
            )
        )
        add(
            best_matching(
                lambda config: (
                    config.max_base_table_occurrences != raw.max_base_table_occurrences
                )
            )
        )
        for depth in sorted({trial.config.depth for trial in ranked}):
            if depth != raw.depth:
                add(best_matching(lambda config, depth=depth: config.depth == depth))

    for trial in ranked:
        add(trial)
        if len(selected) == top_k:
            break
    return tuple(selected)


def confidence_frontier_trials(
    successful: Sequence[Trial],
    *,
    top_k: int,
    uncertainty_multiplier: float = 1.0,
    classification_loss_margin: float = 0.01,
    regression_relative_loss_margin: float = 0.01,
    fill_budget: bool = True,
    extra_controls: int | None = None,
) -> tuple[Trial, ...]:
    """Promote every statistically plausible graph before using score rank.

    Cheap scores may invert at higher fidelity. A graph remains plausible when
    its loss to the cheap leader is smaller than either a practical margin or
    their combined marginal standard error. Distinct configurations are kept
    once; up to ``extra_controls`` spare slots are filled by structural
    diversity (all spare slots when it is ``None``). ``top_k`` is still
    a hard resource ceiling, so callers should report when the frontier is
    larger than the budget instead of treating the result as exhaustive.
    """

    if top_k < 0:
        raise ValueError("top_k must be non-negative")
    if uncertainty_multiplier < 0:
        raise ValueError("uncertainty_multiplier must be non-negative")
    if classification_loss_margin < 0 or regression_relative_loss_margin < 0:
        raise ValueError("practical loss margins must be non-negative")
    if extra_controls is not None and extra_controls < 0:
        raise ValueError("extra_controls must be non-negative or None")
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
            graph_config_sort_key(trial.config),
        ),
    )
    unique: dict[GraphConfig, Trial] = {}
    for trial in ranked:
        unique.setdefault(trial.config, trial)
    ranked = list(unique.values())
    if not ranked:
        return ()
    best = ranked[0]
    margin = (
        classification_loss_margin
        if best.metric in {"roc_auc", "accuracy"}
        else abs(best.validation_score) * regression_relative_loss_margin
    )
    plausible = [
        trial
        for trial in ranked
        if best.objective_score - trial.objective_score
        <= max(
            margin,
            uncertainty_multiplier
            * hypot(best.validation_standard_error, trial.validation_standard_error),
        )
    ]
    if len(plausible) >= top_k:
        return diverse_fidelity_trials(plausible, top_k=top_k)
    if not fill_budget:
        return tuple(plausible)
    selected = list(plausible)
    seen = {trial.config for trial in selected}
    target_count = (
        top_k
        if extra_controls is None
        else min(top_k, len(plausible) + extra_controls)
    )
    if len(selected) == target_count:
        return tuple(selected)
    for trial in diverse_fidelity_trials(ranked, top_k=top_k):
        if trial.config not in seen:
            selected.append(trial)
            seen.add(trial.config)
            if len(selected) == target_count:
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
    return min(
        completed,
        key=lambda trial: (
            -trial.objective_score,
            trial.feature_count,
            trial.config.complexity,
            graph_config_sort_key(trial.config),
        ),
    )


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
    target_history: pd.DataFrame | None = None
    target_history_key: Any = None
    peer_context: Any = None
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
