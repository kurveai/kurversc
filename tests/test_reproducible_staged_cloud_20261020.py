import json
from pathlib import Path
from types import SimpleNamespace

import psutil
import pytest

from scripts import run_reproducible_staged_cloud as runner
from scripts.run_gap_followup_queue import dump


def setup(tmp_path, monkeypatch, *, admitted=False, budget=300, complete=True):
    from scripts import run_kurversc_local_campaign as guard
    baseline = dict(selected="incumbent", query_keys_sha256="same-keys",
        official_test_scores=dict(candidate=.8))
    staged = SimpleNamespace(CONTRACT=dict(task_hours=23.75),
        verify_stages=lambda *a: (baseline, dict(previous=True)))
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        output = Path(command[command.index("--output")+1])
        if command[command.index("-m")+1] == "scripts.run_bounded_view_validation_20261020":
            dump(output / "results.json", dict(selected="bounded_view_mean_0.5" if admitted else "incumbent",
                complete_grid=complete, official_test_accessed=False,
                fallback=None if admitted else "preserve_already_admitted_procedure"))
        else:
            dump(output / "results.json", dict(selected="bounded_view_mean_0.5", native_exact=True,
                deployment_seed=0, query_keys_sha256="same-keys", official_test_scores=dict(candidate=.81)))
        return 0, dict(test=True)
    monkeypatch.setattr(guard, "_run_guarded", run)
    monkeypatch.setattr(runner, "phase_source_root", lambda *a, **kw: tmp_path / "source")
    monkeypatch.setattr(runner, "source_identity", lambda *a: dict(source_sha256={}, profile=runner.PROFILE))
    monkeypatch.setattr(psutil, "virtual_memory", lambda: SimpleNamespace(total=512*2**30))
    monkeypatch.setenv("KURVERSC_DUCKDB_MEMORY_LIMIT", "none")
    fresh = SimpleNamespace(ROOT=tmp_path, remaining_task_seconds=lambda *a: budget,
        task_main=lambda args: 0, official_result=lambda *a: baseline,
        official_path=lambda *a: tmp_path / "previous-result.json")
    runner.extend_recipe(staged, fresh)
    return staged, fresh, SimpleNamespace(dataset="unknown_schema", task="unregistered", output=tmp_path), calls, baseline


def test_optional_fallback_preserves_entire_previous_procedure_exactly(tmp_path, monkeypatch):
    staged, fresh, args, calls, baseline = setup(tmp_path, monkeypatch)
    assert fresh.task_main(args) == 0
    result, stages = staged.verify_stages(tmp_path, args.dataset, args.task)
    assert {k: result[k] for k in baseline} == baseline
    assert len(calls) == 1 and not stages["bounded_views"]["admitted"]
    allocation = json.loads((tmp_path / "execution-allocation.json").read_text())
    assert allocation["task_rss_cap"] is None and allocation["duckdb_spill_threshold"] == "128GiB"
    assert not allocation["source_sampling_changed"] and not allocation["model_budget_changed"]


def test_complete_admission_replays_only_the_new_run_native_path(tmp_path, monkeypatch):
    staged, fresh, args, calls, _ = setup(tmp_path, monkeypatch, admitted=True)
    assert fresh.task_main(args) == 0
    result, stages = staged.verify_stages(tmp_path, args.dataset, args.task)
    assert result["native_exact"] and stages["bounded_views"]["admitted"]
    assert len(calls) == 2
    assert calls[1][calls[1].index("--run-root")+1] == str(tmp_path)
    path = runner.bounded_root(tmp_path, f"{args.dataset}/{args.task}") / "validation/results.json"
    changed = json.loads(path.read_text())
    changed["selected"] = "incumbent"
    dump(path, changed)
    with pytest.raises(ValueError, match="admission changed"):
        staged.verify_stages(tmp_path, args.dataset, args.task)


def test_incomplete_or_expired_procedure_never_calls_official_confirmation(tmp_path, monkeypatch):
    _, fresh, args, calls, _ = setup(tmp_path, monkeypatch, admitted=True, complete=False)
    with pytest.raises(ValueError, match="Incomplete"):
        fresh.task_main(args)
    assert len(calls) == 1
    _, fresh, args, calls, _ = setup(tmp_path / "expired", monkeypatch, budget=0)
    with pytest.raises(TimeoutError, match="deadline exhausted"):
        fresh.task_main(args)
    assert calls == []


def test_engine_spill_policy_is_host_generic_and_never_an_rss_cap():
    assert runner.spill_threshold(48*2**30) == "12GiB"
    assert runner.spill_threshold(512*2**30) == "128GiB"
    assert runner.spill_threshold(1536*2**30) == "128GiB"
    assert runner.STORAGE_POLICY["task_rss_cap"] is None
