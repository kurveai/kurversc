"""Explicit, schema-generic execution profile from the scored DNF recovery.

This is not a registered-default change or a saved-winner lookup. Every task
searches the same reference lattice using its own inner validation evidence.
The effective low-cardinality policy is recorded explicitly: the recovered
run used top-k, while a separate diagnostic tests preserving every level.
"""

import copy
from dataclasses import replace

import kurversc
from kurversc.staged_adapter import KurveRSCSystem, _runtime_config


def recovered_reference_config():
    return {
        **_runtime_config(),
        "full_training_frames": 1,
        "search_training_frames": 1,
        "temporal_estimator_mode": "frame_ensemble",
        "estimator_train_rows": None,
        "categorical_low_cardinality_policy": "top_k",
    }


def recovered_reference_candidates():
    # This is the policy actually executed by the 0.7321 recovery. Make it
    # explicit instead of depending on the previously coupled fit overrides.
    return tuple(
        replace(config, categorical_low_cardinality_policy="top_k")
        for config in kurversc.published_compatible_configs()
    )


class RecoveredReferenceSystem(KurveRSCSystem):
    """Fresh validation selection and frozen-plan outer replay for every task."""

    schema_depth = 3

    def _fit_arm(self, task, db, train_table, val_table, *, seed, phase,
                 selection=None):
        if phase not in {"inner", "outer"}:
            raise ValueError(f"Unknown phase: {phase!r}")
        if phase == "outer" and selection is None:
            raise RuntimeError("Reference reporting requires the inner selection")
        identity = self.run_identity
        dataset_name = identity.dataset if identity is not None else "relbench"
        task_name = identity.task if identity is not None else type(task).__name__
        config = recovered_reference_config()
        problem = kurversc.relbench_problem_from_objects(
            task, db, train_table, val_table,
            dataset_name=dataset_name, task_name=task_name,
            sample_rows=config["sample_rows"], search_full_data=True,
            search_training_frames=1,
            max_train_timestamps=3 if phase == "inner" else 1,
            schema_depth=self.schema_depth, max_base_table_occurrences=1,
            include_target_history=False, random_state=seed,
        )
        kwargs = {**problem.fit_kwargs(), **config, "random_state": seed}
        if phase == "inner":
            from pathlib import Path
            import os

            root = os.environ.get("KURVERSC_SEARCH_CHECKPOINT_DIR")
            if not root:
                raise ValueError("A fresh run-scoped checkpoint directory is required")
            kwargs.update(
                graph_configs=recovered_reference_candidates(),
                search_checkpoint_path=Path(root),
                search_checkpoint_key=f"{dataset_name}/{task_name}/reference/seed={seed}",
            )
        else:
            kwargs.update(preselected_config=selection.config,
                preselected_execution_plan=copy.deepcopy(selection.execution_plan))
        result = kurversc.fit(**kwargs)
        chosen = result.recommended_config
        if (chosen.categorical_low_cardinality_policy != "top_k"
                or chosen.include_target_history
                or not chosen.published_cutoff_compatibility):
            raise RuntimeError("The recovered representation profile was altered")
        return result, problem


def execution_profile(name):
    if name == "registered":
        return KurveRSCSystem, _runtime_config()
    if name == "recovered-reference":
        return RecoveredReferenceSystem, recovered_reference_config()
    raise ValueError(f"Unknown execution profile: {name!r}")
