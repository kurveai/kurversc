"""Execution boundary shared by the benchmark and explicit-context runners.

The benchmark branch reexports the original adapter objects unchanged. The
portable branch needs neither RelArena nor a task registry: its caller owns
the censored phases, and held-out answers are deliberately not representable.
The search, materialization, fitted operation plan and downstream stages are
the same implementations, not a benchmark-specific recipe lookup.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from .task_source import CONTEXT_ENV


if not os.environ.get(CONTEXT_ENV):
    # Keep the registered benchmark lifecycle/defaults byte-for-byte at its
    # boundary. Importing the portable API itself never enters this branch.
    from relarena.cache import resolve_cache_config
    from relarena.dataset import _copy_timeless_tables, _drop_dangling_seeds
    from relarena.models.kurversc.model import KURVERSC_DEFAULTS, KurveRSCSystem, _SelectedGraph, _runtime_config
    from relarena.results import summary_to_dataframe
    from relarena.runner import run_system_experiment
else:
    KURVERSC_DEFAULTS = dict(
        full_training_frames=12,full_validation_frames=None,search_training_frames=12,
        estimator_train_rows=100_000,temporal_estimator_mode="pooled",sample_rows=50_000,
        screening_rows=10_000,confirmation_top_k=8,search_parallelism=1,search_full_data=True,
        rerank_top_k=3,rerank_cutoff_frames=3,rerank_stability_penalty=.25,
        feature_family_max_columns=4,intelligent_column_selection=True,
        adaptive_depth_promotion=True,capability_pruning=True,search_max_features=8_000,
        infer_ts_periods=False,auto_text_features=False,auto_annotate_max_text_columns=None,
        high_cardinality_policy="top_k",text_categorical_fallback=True,
        duckdb_memory_limit="64GB",duckdb_max_temp_directory_size="128GB",
    )

    def _runtime_config():
        result = dict(KURVERSC_DEFAULTS)
        result.update(model_backend="catboost",complexity_selector=True)
        integers = {"sample_rows","screening_rows","search_training_frames","estimator_train_rows",
            "rerank_top_k","search_parallelism","search_max_features","full_training_frames"}
        floats = {"rerank_stability_penalty"}
        booleans = {"search_full_data","intelligent_column_selection","complexity_selector","text_categorical_fallback"}
        nullable = {"full_validation_frames","feature_family_max_columns"}
        strings = {"model_backend","temporal_estimator_mode","high_cardinality_policy"}
        limits = {"duckdb_memory_limit","duckdb_max_temp_directory_size"}
        for name in integers|floats|booleans|nullable|strings|limits:
            value = os.environ.get("KURVERSC_"+name.upper())
            if value is None:
                continue
            if name in integers:
                result[name] = int(value)
            elif name in floats:
                result[name] = float(value)
            elif name in booleans:
                if value.lower() not in {"1","true","yes","0","false","no"}:
                    raise ValueError(f"KURVERSC_{name.upper()} must be boolean")
                result[name] = value.lower() in {"1","true","yes"}
            elif name in nullable|limits:
                if not value.strip():
                    raise ValueError(f"KURVERSC_{name.upper()} cannot be empty")
                result[name] = None if value.lower() in {"none","null","unlimited"} else (int(value) if name in nullable else value)
                if name in nullable and result[name] is not None and result[name]<1:
                    raise ValueError(f"KURVERSC_{name.upper()} must be positive")
            else:
                result[name] = value
        return result

    @dataclass(frozen=True)
    class _SelectedGraph:
        config: object
        execution_plan: dict
        training_frames: int | None = None

    def resolve_cache_config(cache_dir, *, on_miss):
        # Staged search owns its content-addressed checkpoints. This is only
        # lifecycle metadata, not a substitute for a benchmark feature cache.
        return SimpleNamespace(directory=Path(cache_dir).expanduser() if cache_dir else None,on_miss=on_miss)

    def _copy_timeless_tables(db):
        for table in db.table_dict.values():
            if table.time_col is None:
                table.df = table.df.copy(deep=True)

    def _drop_dangling_seeds(*args,**kwargs):
        raise ValueError("Portable entity membership must use declared keys, not dense ID ranges")

    class KurveRSCSystem:
        name = "kurversc"
        schema_depth = 4
        rerank_practical_recent_ties = False

        def __init__(self, *, cache=None,run_identity=None):
            self.cache,self.run_identity = cache,run_identity

        def run(self,task,*,inner_split,outer_split,seed,time_limit=None):
            selected,_ = self._fit_arm(task,inner_split.db_state,inner_split.train_table,
                inner_split.eval_table,seed=seed,phase="inner")
            selection = _SelectedGraph(selected.recommended_config,copy.deepcopy(selected.execution_plan),
                getattr(getattr(selected,"fitted_model",None),"training_frames",None))
            fitted,problem = self._fit_arm(task,outer_split.db_state,outer_split.train_table,
                outer_split.val_table,seed=seed,phase="outer",selection=selection)
            return self._predict(task,outer_split.eval_table,result=fitted,problem=problem)

        def _fit_arm(self,task,db,train_table,val_table,*,seed,phase,selection=None):
            # The staged subclass owns inner search. A no-peer outer replay
            # still uses its exact frozen graph/operation plan.
            import kurversc
            if phase!="outer" or selection is None:
                raise ValueError("Staged inner selection is required before a portable outer replay")
            config = _runtime_config()
            config["full_training_frames"] = selection.training_frames or config["full_training_frames"]
            problem = kurversc.relbench_problem_from_objects(task,db,train_table,val_table,
                dataset_name=self.run_identity.dataset,task_name=self.run_identity.task,
                sample_rows=config["sample_rows"],search_full_data=config["search_full_data"],
                search_training_frames=config["search_training_frames"],
                max_train_timestamps=config["full_training_frames"],schema_depth=self.schema_depth,
                max_base_table_occurrences=int(os.environ.get("KURVERSC_MAX_BASE_TABLE_OCCURRENCES","1")),
                include_target_history=os.environ.get("KURVERSC_INCLUDE_TARGET_HISTORY","1").lower() not in {"0","false","no"},
                random_state=seed)
            kwargs = {**problem.fit_kwargs(),**config,"random_state":seed,
                "preselected_config":selection.config,"preselected_execution_plan":copy.deepcopy(selection.execution_plan)}
            if os.environ.get("KURVERSC_CATBOOST_ITERATIONS"):
                kwargs["model_params"] = dict(iterations=int(os.environ["KURVERSC_CATBOOST_ITERATIONS"]))
            if os.environ.get("KURVERSC_FINAL_MODEL_BACKEND"):
                kwargs["final_model_backend"] = os.environ["KURVERSC_FINAL_MODEL_BACKEND"]
            return kurversc.fit(**kwargs),problem

        def _predict(self,task,table,*,result,problem):
            import kurversc
            frame = table.df.copy()
            clock = getattr(problem,"timestamp_column",task.time_col)
            if clock!=task.time_col:
                frame[clock] = frame[task.time_col].dt.normalize()+np.timedelta64(1,"D")
            config = _runtime_config()
            output = kurversc.predict(result,parent_node=problem.parent_node,
                prediction_node=kurversc.Labels(frame,key=task.entity_col,timestamp=clock),
                tables=problem.tables,relationships=problem.relationships,
                duckdb_memory_limit=config["duckdb_memory_limit"],
                duckdb_max_temp_directory_size=config["duckdb_max_temp_directory_size"],use_validation_model=False)
            return output.prediction.to_numpy(dtype=float)

    def run_system_experiment(*args,**kwargs):
        raise ValueError("Portable execution must supply its context, not a benchmark scorer")

    def summary_to_dataframe(summary):
        return pd.DataFrame([dict(model=summary.system_name,kind="system",dataset=summary.dataset,
            task=summary.task_name,task_type=str(summary.task_type),seed=summary.seed,
            metric=summary.metric_name,selected=True,test_score=None,time_total=summary.result.time_total)])
