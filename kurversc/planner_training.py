"""Baseline training and evaluation tools for the KurvePlanner ranker."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .planner_profiles import PLANNER_PROFILE_VERSION, Scalar
from .planner_traces import (
    CandidateObservation,
    PlannerEpisode,
    read_episodes_jsonl,
    split_episodes_by_database,
)


_META_COLUMNS = {
    "_candidate_id",
    "_completed",
    "_database_id",
    "_episode_id",
    "_group_weight",
    "_metric",
    "_objective_score",
    "_relevance",
    "_stage",
    "_supervision",
    "_task_type",
}


def _flatten_plan_value(output: dict[str, Scalar], prefix: str, value: Any) -> None:
    if value is None or isinstance(value, (str, int, float, bool)):
        output[prefix] = value
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            _flatten_plan_value(output, f"{prefix}__{key}", item)
        return
    if isinstance(value, (list, tuple, set)):
        items = list(value)
        output[f"{prefix}__count"] = len(items)
        if all(isinstance(item, str) for item in items):
            for item in items:
                output[f"{prefix}__has__{item}"] = True
        elif all(isinstance(item, (int, float)) for item in items):
            numeric = [float(item) for item in items]
            if numeric:
                output[f"{prefix}__min"] = min(numeric)
                output[f"{prefix}__max"] = max(numeric)
                output[f"{prefix}__mean"] = sum(numeric) / len(numeric)


def _candidate_features(observation: CandidateObservation) -> dict[str, Scalar]:
    output: dict[str, Scalar] = {}
    for key, value in observation.plan.items():
        _flatten_plan_value(output, f"plan__{key}", value)
    # This estimate is produced by the preflight audit before materialization,
    # unlike observed width, runtime, and validation score.
    output["plan__estimated_feature_count"] = observation.estimated_feature_count
    return output


def _episode_relevance(episode: PlannerEpisode) -> dict[str, float]:
    completed = [
        item.objective_score
        for item in episode.observations
        if item.completed and item.objective_score is not None
    ]
    if not completed:
        return {item.candidate_id: 0.0 for item in episode.observations}
    lowest = min(completed)
    highest = max(completed)
    span = highest - lowest
    relevance = {}
    for item in episode.observations:
        if not item.supervised:
            relevance[item.candidate_id] = float("nan")
        elif not item.completed or item.objective_score is None:
            relevance[item.candidate_id] = 0.0
        elif span <= 1e-12:
            relevance[item.candidate_id] = 1.0
        else:
            relevance[item.candidate_id] = (
                1.0 + 99.0 * (item.objective_score - lowest) / span
            )
    return relevance


def candidate_frame(episodes: Sequence[PlannerEpisode]) -> pd.DataFrame:
    """Flatten structured episodes for the first CatBoost ranking baseline."""

    rows: list[dict[str, Any]] = []
    for episode in episodes:
        context = {
            **episode.database_profile.aggregate_features(),
            **episode.task_profile.aggregate_features(),
            **episode.runtime_profile.aggregate_features(),
        }
        relevance = _episode_relevance(episode)
        for observation in episode.observations:
            rows.append(
                {
                    **context,
                    **_candidate_features(observation),
                    "_episode_id": episode.episode_id,
                    "_database_id": episode.database_id,
                    "_candidate_id": observation.candidate_id,
                    "_stage": episode.stage,
                    "_supervision": observation.supervision,
                    "_group_weight": episode.stage_weight,
                    "_task_type": episode.task_profile.task_type,
                    "_metric": observation.metric,
                    "_completed": observation.completed,
                    "_objective_score": observation.objective_score,
                    "_relevance": relevance[observation.candidate_id],
                }
            )
    return pd.DataFrame(rows)


def _feature_columns(frame: pd.DataFrame) -> tuple[tuple[str, ...], tuple[str, ...]]:
    features = tuple(sorted(set(frame.columns) - _META_COLUMNS))
    categorical = []
    for column in features:
        values = frame[column].dropna()
        if any(isinstance(value, str) for value in values.head(100)):
            categorical.append(column)
    return features, tuple(categorical)


def _prepare_features(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    categorical_columns: Sequence[str],
) -> pd.DataFrame:
    prepared = pd.DataFrame(index=frame.index)
    categorical = set(categorical_columns)
    for column in feature_columns:
        if column not in frame:
            prepared[column] = "__missing__" if column in categorical else np.nan
        elif column in categorical:
            prepared[column] = frame[column].fillna("__missing__").astype(str)
        else:
            prepared[column] = pd.to_numeric(frame[column], errors="coerce")
    return prepared


def _ranking_frame(episodes: Sequence[PlannerEpisode]) -> pd.DataFrame:
    frame = candidate_frame(episodes)
    if frame.empty:
        return frame
    frame = frame[frame["_supervision"] != "censored"].reset_index(drop=True)
    group_sizes = frame.groupby("_episode_id", sort=False).size()
    completed = frame.groupby("_episode_id", sort=False)["_completed"].sum()
    eligible = set(group_sizes[group_sizes >= 2].index) & set(
        completed[completed >= 1].index
    )
    return frame[frame["_episode_id"].isin(eligible)].reset_index(drop=True)


@dataclass(frozen=True)
class PlannerTrainingConfig:
    """Reproducible settings for the initial gradient-boosted meta-ranker."""

    iterations: int = 500
    depth: int = 7
    learning_rate: float = 0.05
    loss_function: str = "YetiRank"
    random_state: int = 42
    thread_count: int = -1

    def __post_init__(self) -> None:
        if self.iterations < 1 or self.depth < 1 or self.learning_rate <= 0:
            raise ValueError("iterations, depth, and learning_rate must be positive")


class KurvePlannerRanker:
    """CatBoost candidate ranker used to validate the planner training signal."""

    def __init__(
        self,
        *,
        model: Any,
        feature_columns: Sequence[str],
        categorical_columns: Sequence[str],
        training_config: PlannerTrainingConfig,
    ) -> None:
        self.model = model
        self.feature_columns = tuple(feature_columns)
        self.categorical_columns = tuple(categorical_columns)
        self.training_config = training_config

    @classmethod
    def fit(
        cls,
        train_episodes: Sequence[PlannerEpisode],
        *,
        validation_episodes: Sequence[PlannerEpisode] = (),
        config: PlannerTrainingConfig | None = None,
    ) -> "KurvePlannerRanker":
        from catboost import CatBoostRanker, Pool

        settings = config or PlannerTrainingConfig()
        train = _ranking_frame(train_episodes)
        if train.empty:
            raise ValueError("No eligible ranking episodes were found for training")
        feature_columns, categorical_columns = _feature_columns(train)
        train_x = _prepare_features(train, feature_columns, categorical_columns)
        train_pool = Pool(
            train_x,
            label=train["_relevance"],
            group_id=train["_episode_id"],
            group_weight=train["_group_weight"],
            cat_features=list(categorical_columns),
        )

        validation_pool = None
        validation = _ranking_frame(validation_episodes)
        if not validation.empty:
            validation_x = _prepare_features(
                validation, feature_columns, categorical_columns
            )
            validation_pool = Pool(
                validation_x,
                label=validation["_relevance"],
                group_id=validation["_episode_id"],
                group_weight=validation["_group_weight"],
                cat_features=list(categorical_columns),
            )

        model = CatBoostRanker(
            iterations=settings.iterations,
            depth=settings.depth,
            learning_rate=settings.learning_rate,
            loss_function=settings.loss_function,
            eval_metric="NDCG:top=1",
            random_seed=settings.random_state,
            thread_count=settings.thread_count,
            allow_writing_files=False,
            verbose=False,
        )
        model.fit(
            train_pool,
            eval_set=validation_pool,
            use_best_model=validation_pool is not None,
        )
        return cls(
            model=model,
            feature_columns=feature_columns,
            categorical_columns=categorical_columns,
            training_config=settings,
        )

    def score_episode(self, episode: PlannerEpisode) -> np.ndarray:
        frame = candidate_frame((episode,))
        features = _prepare_features(
            frame, self.feature_columns, self.categorical_columns
        )
        return np.asarray(self.model.predict(features), dtype=float)

    def save(self, directory: str | Path) -> None:
        destination = Path(directory)
        destination.mkdir(parents=True, exist_ok=True)
        self.model.save_model(str(destination / "ranker.cbm"))
        manifest = {
            "artifact_type": "kurveplanner_catboost_ranker",
            "artifact_version": "1.0",
            "profile_version": PLANNER_PROFILE_VERSION,
            "feature_columns": list(self.feature_columns),
            "categorical_columns": list(self.categorical_columns),
            "training_config": asdict(self.training_config),
        }
        (destination / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, directory: str | Path) -> "KurvePlannerRanker":
        from catboost import CatBoostRanker

        source = Path(directory)
        manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("profile_version") != PLANNER_PROFILE_VERSION:
            raise ValueError("Planner artifact uses an unsupported profile version")
        model = CatBoostRanker()
        model.load_model(str(source / "ranker.cbm"))
        return cls(
            model=model,
            feature_columns=manifest["feature_columns"],
            categorical_columns=manifest["categorical_columns"],
            training_config=PlannerTrainingConfig(**manifest["training_config"]),
        )


def summarize_episodes(episodes: Sequence[PlannerEpisode]) -> dict[str, Any]:
    """Return corpus coverage and supervision counts before a costly run."""

    stage_counts: dict[str, int] = {}
    task_type_counts: dict[str, int] = {}
    status_counts: dict[str, int] = {}
    supervision_counts: dict[str, int] = {}
    candidate_count = 0
    for episode in episodes:
        stage_counts[episode.stage] = stage_counts.get(episode.stage, 0) + 1
        task_type = episode.task_profile.task_type
        task_type_counts[task_type] = task_type_counts.get(task_type, 0) + 1
        for observation in episode.observations:
            candidate_count += 1
            status_counts[observation.status] = (
                status_counts.get(observation.status, 0) + 1
            )
            supervision_counts[observation.supervision] = (
                supervision_counts.get(observation.supervision, 0) + 1
            )
    return {
        "database_count": len({episode.database_id for episode in episodes}),
        "task_count": len(
            {
                (episode.database_id, episode.task_profile.task_id)
                for episode in episodes
            }
        ),
        "episode_count": len(episodes),
        "candidate_count": candidate_count,
        "stage_counts": stage_counts,
        "task_type_episode_counts": task_type_counts,
        "candidate_status_counts": status_counts,
        "candidate_supervision_counts": supervision_counts,
    }


def evaluate_ranker(
    ranker: KurvePlannerRanker,
    episodes: Sequence[PlannerEpisode],
    *,
    top_k: int = 3,
    normalized_regret_tolerance: float = 0.01,
) -> dict[str, Any]:
    """Measure unseen-database selection regret and top-k oracle recall."""

    if top_k < 1:
        raise ValueError("top_k must be positive")
    records = []
    for episode in episodes:
        evaluable_indices = [
            index for index, item in enumerate(episode.observations) if item.supervised
        ]
        completed = [
            item for item in episode.observations if item.supervised and item.completed
        ]
        if not completed:
            continue
        scores = ranker.score_episode(episode)
        order = np.asarray(evaluable_indices)[
            np.argsort(-scores[np.asarray(evaluable_indices)])
        ]
        selected = episode.observations[int(order[0])]
        best = max(completed, key=lambda item: float(item.objective_score))
        worst = min(completed, key=lambda item: float(item.objective_score))
        best_score = float(best.objective_score)
        worst_score = float(worst.objective_score)
        span = best_score - worst_score
        if selected.completed:
            selected_score = float(selected.objective_score)
            raw_regret = best_score - selected_score
            normalized_regret = raw_regret / span if span > 1e-12 else 0.0
        else:
            selected_score = None
            raw_regret = None
            normalized_regret = 1.0
        top_candidates = {
            episode.observations[int(index)].candidate_id
            for index in order[: min(top_k, len(order))]
        }
        records.append(
            {
                "episode_id": episode.episode_id,
                "database_id": episode.database_id,
                "task_id": episode.task_profile.task_id,
                "task_type": episode.task_profile.task_type,
                "stage": episode.stage,
                "candidate_count": len(episode.observations),
                "best_candidate_id": best.candidate_id,
                "selected_candidate_id": selected.candidate_id,
                "best_objective_score": best_score,
                "selected_objective_score": selected_score,
                "raw_regret": raw_regret,
                "normalized_regret": normalized_regret,
                "within_tolerance": (normalized_regret <= normalized_regret_tolerance),
                "top_k_oracle_recall": best.candidate_id in top_candidates,
                "selected_failed": not selected.completed,
            }
        )

    summary: dict[str, Any] = {
        "episode_count": len(records),
        "normalized_regret_tolerance": normalized_regret_tolerance,
        "top_k": top_k,
    }
    if records:
        regrets = [float(record["normalized_regret"]) for record in records]
        summary.update(
            {
                "mean_normalized_regret": float(np.mean(regrets)),
                "median_normalized_regret": float(np.median(regrets)),
                "within_tolerance_fraction": float(
                    np.mean([record["within_tolerance"] for record in records])
                ),
                "top_k_oracle_recall": float(
                    np.mean([record["top_k_oracle_recall"] for record in records])
                ),
                "failed_selection_fraction": float(
                    np.mean([record["selected_failed"] for record in records])
                ),
            }
        )
    by_task_type = {}
    for task_type in sorted({record["task_type"] for record in records}):
        selected = [record for record in records if record["task_type"] == task_type]
        regrets = [float(record["normalized_regret"]) for record in selected]
        by_task_type[task_type] = {
            "episode_count": len(selected),
            "mean_normalized_regret": float(np.mean(regrets)),
            "median_normalized_regret": float(np.median(regrets)),
            "within_tolerance_fraction": float(
                np.mean([record["within_tolerance"] for record in selected])
            ),
            "top_k_oracle_recall": float(
                np.mean([record["top_k_oracle_recall"] for record in selected])
            ),
            "failed_selection_fraction": float(
                np.mean([record["selected_failed"] for record in selected])
            ),
        }
    return {"summary": summary, "by_task_type": by_task_type, "episodes": records}


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the initial KurvePlanner candidate ranker"
    )
    parser.add_argument("traces", nargs="+", help="Planner JSONL trace files")
    parser.add_argument("--output", required=True, help="Artifact output directory")
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--validation-size", type=int)
    parser.add_argument("--test-size", type=int, default=2)
    parser.add_argument("--validation-databases", nargs="*")
    parser.add_argument("--test-databases", nargs="*")
    parser.add_argument(
        "--required-test-task-types",
        nargs="*",
        default=["classification", "regression"],
        help="Task types that the final held-out database set must cover",
    )
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--depth", type=int, default=7)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--thread-count", type=int, default=-1)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--evaluate-test",
        action="store_true",
        help="Evaluate the final held-out databases (omit during model selection)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate traces and write the database split without training",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    episodes = read_episodes_jsonl(args.traces)
    split = split_episodes_by_database(
        episodes,
        train_fraction=args.train_fraction,
        validation_size=args.validation_size,
        test_size=args.test_size,
        validation_databases=args.validation_databases,
        test_databases=args.test_databases,
        required_test_task_types=args.required_test_task_types,
        random_state=args.random_state,
    )
    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=True)
    corpus_summary = summarize_episodes(episodes)
    _write_json(destination / "dataset_summary.json", corpus_summary)
    _write_json(destination / "split.json", split.to_dict())
    if args.dry_run:
        print(
            json.dumps(
                {
                    "artifact": str(destination),
                    "dataset": corpus_summary,
                    "split": split.to_dict(),
                    "trained": False,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    config = PlannerTrainingConfig(
        iterations=args.iterations,
        depth=args.depth,
        learning_rate=args.learning_rate,
        random_state=args.random_state,
        thread_count=args.thread_count,
    )
    ranker = KurvePlannerRanker.fit(
        split.train,
        validation_episodes=split.validation,
        config=config,
    )
    ranker.save(destination)
    validation = evaluate_ranker(ranker, split.validation)
    _write_json(destination / "validation_metrics.json", validation)
    output = {
        "artifact": str(destination),
        "dataset": corpus_summary,
        "split": split.to_dict(),
        "validation": validation["summary"],
    }
    if args.evaluate_test:
        test = evaluate_ranker(ranker, split.test)
        _write_json(destination / "test_metrics.json", test)
        output["test"] = test["summary"]
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
