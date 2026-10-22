import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import run_sequence_recovery_cloud as runner
from scripts.run_gap_followup_queue import dump


def setup(tmp_path, monkeypatch, *, admitted=False, budget=300):
    from scripts import run_kurversc_local_campaign as guard
    baseline = dict(selected="incumbent", official_test_scores=dict(candidate=.8), query_keys_sha256="same-keys")
    staged = SimpleNamespace(CONTRACT=dict(task_hours=23.75), verify_stages=lambda *a: (baseline, dict(previous=True)))
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        output = Path(command[command.index("--output")+1])
        if command[command.index("-m")+1] == "scripts.benchmark_production_sequence":
            dump(output / "selection.json", dict(selected="sequence_weight_paired_0.5" if admitted else "incumbent",
                coefficients=[.3828125, .3125, .3046875] if admitted else [1., 0., 0.],
                complete_grid=admitted, fallback=None if admitted else "preserve_already_admitted_signal_or_readout",
                official_targets_used_for_selection=False))
        else:
            dump(output / "results.json", dict(selected="sequence_weight_paired_0.5", native_exact=True,
                deployment_seed=0, query_keys_sha256="same-keys", official_test_scores=dict(candidate=.81)))
        return 0, dict(test=True)
    monkeypatch.setattr(guard, "_run_guarded", run)
    monkeypatch.setattr(runner, "phase_source_root", lambda *a, **kw: tmp_path / "unseen-source")
    fresh = SimpleNamespace(ROOT=tmp_path, remaining_task_seconds=lambda *a: budget,
        task_main=lambda args: 0, official_result=lambda *a: baseline,
        official_path=lambda *a: tmp_path / "old-results.json")
    runner.extend_recipe(staged, fresh)
    args = SimpleNamespace(dataset="new-schema", task="new-problem", output=tmp_path)
    return staged, fresh, args, calls, baseline


def test_existing_readout_fallback_is_exact_and_does_not_run_final_fit(tmp_path, monkeypatch):
    staged, fresh, args, calls, baseline = setup(tmp_path, monkeypatch)
    assert fresh.task_main(args) == 0
    result, stages = staged.verify_stages(tmp_path, args.dataset, args.task)
    assert all(result[k] == v for k, v in baseline.items())
    assert len(calls) == 1
    assert stages["production_sequence"]["admitted"] is False
    assert staged.CONTRACT["task_hours"] < 24
    assert "--workers" in calls[0] and calls[0][calls[0].index("--workers")+1] == "1"


def test_only_complete_admission_can_run_official_fit(tmp_path, monkeypatch):
    staged, fresh, args, calls, _ = setup(tmp_path, monkeypatch, admitted=True)
    assert fresh.task_main(args) == 0
    result, stages = staged.verify_stages(tmp_path, args.dataset, args.task)
    assert len(calls) == 2 and result["native_exact"]
    assert stages["production_sequence"]["admitted"]
    path = runner.sequence_root(tmp_path, f"{args.dataset}/{args.task}") / "validation/selection.json"
    value = json.loads(path.read_text())
    value["coefficients"] = [1., 0., 0.]
    dump(path, value)
    with pytest.raises(ValueError, match="admission changed"):
        staged.verify_stages(tmp_path, args.dataset, args.task)


def test_original_task_deadline_is_not_reset(tmp_path, monkeypatch):
    _, fresh, args, calls, _ = setup(tmp_path, monkeypatch, budget=0)
    with pytest.raises(TimeoutError, match="Original complete-task deadline"):
        fresh.task_main(args)
    assert calls == []
