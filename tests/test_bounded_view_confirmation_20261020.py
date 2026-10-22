import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from scripts import confirm_bounded_view_recovery_20261020 as confirm
from scripts import capture_owned_pooled_native_20261020 as capture
from scripts.run_gap_followup_queue import digest


def test_full_admission_replay_checks_hashes_and_locked_graph_before_any_final_fit(tmp_path, monkeypatch):
    folder = tmp_path / "consensus"
    folder.mkdir()
    meta = dict(dataset="arbitrary", task="unseen", feature_schema=dict(columns=["x"]))
    gate = dict(selected=confirm.RECIPES[1], reason="eligible")
    rows = []
    for fold in (0, 1):
        (folder / f"fold{fold}").mkdir()
        for seed in range(3):
            path = folder / f"fold{fold}/seed{seed}.npz"
            np.savez_compressed(path, truth=[0, 1], groups=[1, 2], control=[.3, .7],
                **{n: [.2, .8] for n in confirm.RECIPES})
            rows.append(dict(fold=fold, seed=seed, origin=str(fold), vectors_sha256=digest(path)))
    m = dict(metadata=meta, allowance=1024, sources={}, inputs={}, component_headers=[], folds=[0, 1])
    r = dict(complete_grid=True, official_test_accessed=False, selected=confirm.RECIPES[1], records=rows, gate=gate)
    (folder / "manifest.json").write_text(json.dumps(m))
    (folder / "results.json").write_text(json.dumps(r))
    monkeypatch.setattr(confirm, "matched_gate", lambda *args, **kwargs: gate)
    assert confirm.admission(tmp_path, meta)[1]["selected"] == confirm.RECIPES[1]
    with pytest.raises(ValueError, match="admission"):
        confirm.admission(tmp_path, dict(meta, task="other_graph"))
    (folder / "fold0/seed0.npz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="vectors changed"):
        confirm.admission(tmp_path, meta)


def test_owned_native_harness_is_stopped_before_scorer_and_restores_all_hooks(tmp_path, monkeypatch):
    import kurversc.core as core
    import kurversc.task_source as sources
    from kurversc.prediction_provenance import key_arrays, masked_query_keys
    from scripts import run_official_cloud as cloud
    from scripts import benchmark_supported_official as harness

    root = tmp_path / "fresh"
    task_root, output = root / "tasks/task", tmp_path / "output"
    task_root.mkdir(parents=True)
    output.mkdir()
    task = object.__new__(sources.ContextTask)
    task.entity_col, task.entity_table, task.target_col = "id", "actors", "target"
    outer = SimpleNamespace(eval_table=SimpleNamespace(df=pd.DataFrame(dict(id=[2, 1]))))
    source = SimpleNamespace(task=task, outer_split=lambda: outer)
    monkeypatch.setattr(sources, "load_task_source", lambda *args, **kwargs: source)
    monkeypatch.setattr(capture, "load_owned_trial", lambda *args: (root / "own.pkl", SimpleNamespace()))
    training = pd.DataFrame(dict(id=[1, 2], x=[3., 4.], target=[0, 1]))
    forecast = pd.DataFrame(dict(id=[2, 1], x=[4., 3.], __kurversc_prediction_order__0=[0, 1]))
    current = [training]
    original_core = lambda *args, **kwargs: current[0]
    monkeypatch.setattr(core, "_materialize_at_cutoff", original_core)
    original_predict = lambda *a, **kw: np.array([.8, .2])
    monkeypatch.setattr(harness.AuditWinnerSystem, "_predict", original_predict)
    scorer_calls = []
    original_scorer = lambda *args, **kwargs: scorer_calls.append(1)
    monkeypatch.setattr(sources.ContextTask, "evaluate", original_scorer)
    checkpoint_function = harness._checkpoint

    def fake_harness():
        core._materialize_at_cutoff(cut_date="2001-01-01")
        current[0] = forecast
        core._materialize_at_cutoff(cut_date="2001-02-01")
        system = object.__new__(harness.AuditWinnerSystem)
        system._predict()
        # Different native arms may legitimately reduce floating aggregates
        # differently. Capture the actual candidate, not an earlier control.
        current[0] = training.assign(x=[3.000000000000001, 4.])
        core._materialize_at_cutoff(cut_date="2001-01-01")
        current[0] = forecast
        core._materialize_at_cutoff(cut_date="2001-02-01")
        system._predict()
        folder = output / "native-capture"
        folder.mkdir()
        keys = masked_query_keys(pd.DataFrame(dict(entity=[2, 1], at=pd.Timestamp("2001-02-01"))),
            entity="entity", timestamp="at", target="target")
        np.savez_compressed(folder / "predictions.npz", candidate=[.8, .2], **key_arrays(keys))
        task.evaluate(np.array([.8, .2]), target_table=None)
        pytest.fail("Hidden-target interception did not stop the harness")

    monkeypatch.setattr(cloud, "main", fake_harness)
    meta = dict(dataset="not_registered", task="new_task", target="target", feature_schema=dict(columns=["x"]))
    result = capture.materialize(SimpleNamespace(run_root=root, task_root=task_root, output=output), meta)
    assert scorer_calls == [] and sources.ContextTask.evaluate is original_scorer
    assert core._materialize_at_cutoff is original_core and harness._checkpoint is checkpoint_function
    assert harness.AuditWinnerSystem._predict is original_predict
    np.testing.assert_array_equal(result[-1], [.8, .2])
    assert len(result[2]) == len(result[3]) == 2
    assert result[2].x.iloc[0] == 3.000000000000001
