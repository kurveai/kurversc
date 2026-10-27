"""One full retrospective feature-parity check using bounded source storage.

No search, altered graph, reduced feature set or official scoring. A one-tree
driver materializes the same frozen plan; all driver scores are discarded.
Only complete cell-for-cell prepared-feature parity counts as success.
"""

import argparse
import copy
from contextlib import contextmanager
import gc
import hashlib
import json
from pathlib import Path
import pickle
import time

import numpy as np
import pandas as pd

import kurversc.core as core
from scripts.arrow_relbench_storage import arrow_relbench_storage
from scripts.benchmark_signal_challengers import prepared
from scripts.engagement_local_comparison import freeze
from scripts.run_gap_followup_queue import dump
from scripts.select_activity_readout import cache_locations


def file_hash(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            hasher.update(block)
    return hasher.hexdigest()


@contextmanager
def persist_frames(output, schema, *, target, entity, root):
    original = core._materialize_at_cutoff
    records = []

    def materialize(*args, **kwargs):
        frame = original(*args, **kwargs)
        key = entity if entity in frame else f"{root}_{entity}"
        if key not in frame or target not in frame or set(schema["columns"]) - set(frame):
            raise ValueError("Frozen full feature schema or routing is missing")
        selected = [*schema["columns"], target]
        saved = pd.DataFrame({name: frame[name] for name in selected}, copy=False)
        saved["__event_entity"] = frame[key].to_numpy()
        saved["__event_cutoff"] = str(pd.Timestamp(kwargs["cut_date"]))
        path = output / f"frame{len(records)}.parquet"
        saved.to_parquet(path, index=False)
        records.append(dict(path=str(path), cutoff=str(pd.Timestamp(kwargs["cut_date"])),
                            rows=len(saved), sha256=file_hash(path)))
        dump(output / "captured-frames.json", records)
        del saved
        gc.collect()
        return frame

    core._materialize_at_cutoff = materialize
    try:
        yield records
    finally:
        core._materialize_at_cutoff = original


def compare_prepared_frames(candidate, reference, schema, *, target):
    keys = ["__event_entity", "__event_cutoff"]
    left = pd.read_parquet(reference, columns=[*keys, target])
    right = pd.read_parquet(candidate, columns=[*keys, target])
    for frame in (left, right):
        frame["__event_cutoff"] = pd.to_datetime(frame.__event_cutoff)
        if frame.duplicated(keys).any() or frame[keys].isna().any().any():
            raise ValueError("Unique complete entity/time identity required")
    expected = pd.MultiIndex.from_frame(left[keys])
    actual = pd.MultiIndex.from_frame(right[keys])
    positions = actual.get_indexer(expected)
    if len(left) != len(right) or (positions < 0).any():
        raise ValueError("Storage replay changed the root cohort")
    np.testing.assert_array_equal(left[target], right.iloc[positions][target])
    for start in range(0, len(schema["columns"]), 32):
        columns = schema["columns"][start:start + 32]
        small_schema = {**schema, "columns": columns}
        a = prepared(pd.read_parquet(reference, columns=columns), small_schema).reset_index(drop=True)
        b = prepared(pd.read_parquet(candidate, columns=columns).iloc[positions], small_schema).reset_index(drop=True)
        pd.testing.assert_frame_equal(a, b, check_exact=True, check_dtype=True)
        del a, b
        gc.collect()
    return dict(rows=len(left), columns=len(schema["columns"]), ordered_entity_time_identity=True,
                prepared_features_exact=True, target_values_exact=True)


def run(args):
    from relarena.cache import resolve_cache_config
    from relarena.dataset import RelBenchDatasetTask, _copy_timeless_tables, _drop_dangling_seeds
    from relarena.models.kurversc.model import KURVERSC_DEFAULTS, _SelectedGraph
    from kurversc.plan_dates import atomic_execution_plan
    from scripts.run_kurversc_selected_relarena import AuditWinnerSystem

    started = time.monotonic()
    raw = json.loads((args.task_root / "raw-audit/manifest.json").read_text())
    meta, (_, snapshots) = raw["metadata"], cache_locations(args.task_root)
    fold = len(raw["folds"]) - 1
    snapshot_path = snapshots / f"fold{fold}/manifest.json"
    snapshot = json.loads(snapshot_path.read_text())
    boundary = pd.Timestamp(snapshot["snapshot_cutoff"])
    checkpoint = Path(meta["checkpoint"])
    if file_hash(checkpoint) != meta["checkpoint_sha256"]:
        raise ValueError("Frozen checkpoint bytes changed")
    reference = snapshots / f"fold{fold}/validation.parquet"
    sources = [__file__, "scripts/arrow_relbench_storage.py", "scripts/materialize_policy_folds.py",
               "kurversc/core.py", "kurversc/modeling.py", "kurversc/relbench.py"]
    identity = dict(metadata=meta, origin=snapshot["origin"], knowledge_boundary=str(boundary),
        sources={p: file_hash(p) for p in sources}, inputs={str(p): file_hash(p) for p in
            (checkpoint, snapshot_path, reference, args.task_root / "raw-audit/manifest.json")},
        scope="Full retrospective storage-equivalence probe only; no official scoring",
        default_changed=False, official_test_accessed=False)
    args.output.mkdir(parents=True, exist_ok=True)
    freeze(args.output / "manifest.json", identity)
    trial = pickle.loads(checkpoint.read_bytes())["trial"]
    trial.execution_plan = atomic_execution_plan(trial.execution_plan)
    defaults, storage = copy.deepcopy(KURVERSC_DEFAULTS), []
    try:
        with arrow_relbench_storage(storage):
            print(json.dumps(dict(phase="load_arrow_database")), flush=True)
            source = RelBenchDatasetTask(meta["dataset"], meta["task"], download=False)
            task = source.task
            db = source._db.upto(boundary)
            _copy_timeless_tables(db)
            source._dataset.validate_and_correct_db(db)
            source._db = db
            # Release only this process's cached larger DB; the requested phase
            # and its exact rows are retained, with no on-disk dataset edits.
            source._dataset.get_db.cache_clear()
            labels = pd.concat([source._tables[n].df for n in ("train", "val")], ignore_index=True)
            clocks = pd.to_datetime(labels[task.time_col])
            train, held = copy.copy(source._tables["train"]), copy.copy(source._tables["val"])
            train.df = labels.loc[(clocks < boundary) & (clocks + task.timedelta <= boundary)].copy()
            held.df = labels.loc[clocks.isin(pd.to_datetime(snapshot["cutoffs"]))].copy()
            count = len(db.table_dict[task.entity_table])
            train = _drop_dangling_seeds(train, task.entity_col, count)
            held = _drop_dangling_seeds(held, task.entity_col, count)
            if len(train.df) != snapshot["observed_training_label_rows"] or len(held.df) != snapshot["heldout_rows"]:
                raise ValueError("Retrospective source cohorts do not match the frozen phase")
            dump(args.output / "storage-audit.json", storage)
            del labels, clocks
            gc.collect()
            KURVERSC_DEFAULTS.update(full_training_frames=1, full_validation_frames=None,
                base_distribution_features=True, temporal_estimator_mode="frame_ensemble",
                duckdb_memory_limit="1GB", duckdb_max_temp_directory_size="40GB",
                model_params=dict(iterations=1, thread_count=1))
            system = AuditWinnerSystem(cache=resolve_cache_config(None, on_miss="raise"), run_identity=source.run_identity())
            system.peer_context_mode, system.peer_max_history_rows = "both", 50_000
            print(json.dumps(dict(phase="materialize_full_frozen_phase", rows=len(held.df),
                                  columns=len(meta["feature_schema"]["columns"]))), flush=True)
            with persist_frames(args.output, meta["feature_schema"], target=task.target_col,
                                entity=task.entity_col, root=task.entity_table) as records:
                fitted, problem = system._fit_arm(task, db, train, held, seed=0, phase="outer",
                    selection=_SelectedGraph(config=trial.config, execution_plan=copy.deepcopy(trial.execution_plan)))
            del fitted, problem, db, source, train, held
            gc.collect()
    finally:
        KURVERSC_DEFAULTS.clear()
        KURVERSC_DEFAULTS.update(defaults)
    matching = [r for r in records if pd.Timestamp(r["cutoff"]) == pd.Timestamp(snapshot["origin"])]
    if len(matching) != 1:
        raise ValueError("Expected one complete matching held-out phase")
    print(json.dumps(dict(phase="compare_all_frozen_features")), flush=True)
    comparison = compare_prepared_frames(Path(matching[0]["path"]), reference, meta["feature_schema"], target=meta["target"])
    for section in ("sources", "inputs"):
        if any(file_hash(p) != h for p, h in identity[section].items()):
            raise ValueError("Frozen storage-probe evidence changed")
    dump(args.output / "results.json", dict(comparison=comparison, records=records,
         storage=storage, elapsed_seconds=time.monotonic()-started,
         official_test_accessed=False, default_changed=False, numerical_equivalence_qualified=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("task-root", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    run(parser.parse_args())
