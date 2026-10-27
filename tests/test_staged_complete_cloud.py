import json
from pathlib import Path

import pandas as pd
import pytest

from scripts import run_staged_complete_cloud as staged
from scripts.run_gap_followup_queue import dump


def test_full_recipe_cannot_omit_the_validation_selected_distribution_head():
    contract = staged.CONTRACT
    assert contract["graph_profile"] == "staged-retention"
    assert contract["base_cdf"] is True and contract["peer_context"] == "both"
    assert contract["temporal_selection_required"] is True
    assert contract["ordered_median_challenger"] is True
    assert contract["optional_signal_selection_required"] is True
    assert contract["force_graph"] is False
    assert contract["reuse_previous_scores"] is False
    assert contract["task_hours"] < 24


def test_attendance_is_the_only_benchmark_admitted_before_recovery_gate():
    manifest = staged.build_manifest(Path("/fresh"), staged.base.task_pairs(), [], 20, 64)
    jobs = [j for j in manifest["jobs"] if j["name"].startswith("rel-")]
    assert len(jobs) == 21
    assert manifest["max_workers"] == 8
    for job in jobs:
        assert ("attendance-recovery-gate" in job["depends"]) == (
            job["name"] != "rel-event_user-attendance")
        assert "--fixed-graph" not in job["command"]
        assert job["threads"] == 8
    gate = next(j for j in manifest["jobs"] if j["name"] == "attendance-recovery-gate")
    assert set(gate["depends"]) == {"rel-event_user-attendance", "leaderboard-snapshot"}


def test_operational_gate_requires_recovered_win_not_just_completed_pipeline():
    stages = {"temporal_policy": {"name": "any_validation_selected_policy"}}
    assert staged.recovery_passes(.2365, .24165, stages)
    assert not staged.recovery_passes(.2553, .24165, stages)
    assert not staged.recovery_passes(.24165, .24165, stages)
    with pytest.raises(ValueError, match="learner stage"):
        staged.recovery_passes(.2365, .24165, {})
    with pytest.raises(ValueError, match="Finite"):
        staged.recovery_passes(float("nan"), .24165, stages)


def test_official_rows_survive_the_upstream_selected_row_filter():
    from relarena.evaluation.leaderboard import to_bencheval_frame

    row = staged.official_score_row("rel-event", "user-attendance",
        dict(metric="mae", official_test_scores=dict(candidate=.2365)), elapsed_seconds=123)
    opponent = dict(model="opponent", dataset="rel-event", task="user-attendance",
        metric="mae", test_score=.24165, selected=True, time_total=100)
    frame = to_bencheval_frame(pd.DataFrame([opponent, row]))
    assert set(frame.method) == {"opponent", "kurversc"}
    assert frame.loc[frame.method.eq("kurversc"), "time_train_s"].iloc[0] == 123


def evidence(root):
    slug = "rel-event_user-attendance"
    folder = root / "workers" / slug / "temporal/tasks" / slug
    temporal = dict(selected_policy=dict(name="ordered12"),
        decision=dict(selected="ordered12", reason="paired_recent_gain_with_temporal_veto",
            candidates={"ordered12": dict(reason="eligible")}))
    signals = dict(selected="incumbent", decision=dict(selected="incumbent"))
    graph = dict(profile="staged-retention", base_distribution_features=True,
        peer_context_mode="both", config="a freshly validation-selected graph")
    official = dict(dataset="rel-event", task="user-attendance", metric="mae",
        arm="ordered12", official_test_scores=dict(control=.255, candidate=.2365))
    dump(root / "campaign-status.json", dict(identity=dict(signal_challengers=True)))
    dump(folder / "selection/results.json", temporal)
    dump(folder / "signal-challengers/selection.json", signals)
    dump(folder / "signal-challengers/results.json", official)
    dump(root / "graphs/results" / (slug + ".json"), graph)
    return folder, temporal, graph


def test_complete_procedure_accepts_only_a_validation_admitted_policy(tmp_path):
    folder, temporal, graph = evidence(tmp_path)
    official, stages = staged.verify_stages(tmp_path, "rel-event", "user-attendance")
    assert official["official_test_scores"]["candidate"] == .2365
    assert stages["temporal_policy"]["name"] == "ordered12"
    temporal["decision"]["candidates"]["ordered12"]["reason"] = "failed_confidence"
    dump(folder / "selection/results.json", temporal)
    with pytest.raises(ValueError, match="validation admission"):
        staged.verify_stages(tmp_path, "rel-event", "user-attendance")


@pytest.mark.parametrize("field,value", [("base_distribution_features", False),
    ("profile", "recovered-reference"), ("peer_context_mode", "off")])
def test_omitted_stronger_graph_stages_fail_closed(tmp_path, field, value):
    folder, temporal, graph = evidence(tmp_path)
    graph[field] = value
    dump(tmp_path / "graphs/results/rel-event_user-attendance.json", graph)
    with pytest.raises(ValueError, match="omitted"):
        staged.verify_stages(tmp_path, "rel-event", "user-attendance")


def test_missing_temporal_selection_cannot_be_misreported_as_a_complete_task(tmp_path):
    evidence(tmp_path)
    path = tmp_path / "workers/rel-event_user-attendance/temporal/tasks/rel-event_user-attendance/selection/results.json"
    path.unlink()
    with pytest.raises(FileNotFoundError):
        staged.verify_stages(tmp_path, "rel-event", "user-attendance")


@pytest.mark.parametrize("recovered", [True, False])
def test_real_queue_unblocks_all_21_only_after_recovery(tmp_path, monkeypatch, recovered):
    """Exercise the actual scheduler, not just the manifest's dependency list."""
    from argparse import Namespace
    from types import SimpleNamespace
    from scripts import run_gap_followup_queue as queue

    manifest = staged.build_manifest(tmp_path, staged.base.task_pairs(), [], 8, 64)
    manifest_path = tmp_path / "manifest.json"
    dump(manifest_path, manifest)
    jobs = {j["name"]: j for j in manifest["jobs"]}
    started, live = [], set()
    gate_completed = False
    peak_workers = 0

    class Worker:
        def __init__(self, command, **kwargs):
            nonlocal peak_workers
            mode = command[command.index("--mode") + 1]
            if mode == "prepare":
                name = "prepare-" + command[command.index("--dataset") + 1]
            elif mode == "task":
                dataset = command[command.index("--dataset") + 1]
                task = command[command.index("--task") + 1]
                name = f"{dataset}_{task}"
                if name != "rel-event_user-attendance":
                    assert gate_completed and recovered
            else:
                name = {"snapshot": "leaderboard-snapshot",
                    "gate": "attendance-recovery-gate",
                    "collect": "final-official-elo"}[mode]
            self.name, self.pid, self.returncode = name, 10000 + len(started), None
            self.polls = 0
            started.append(name)
            live.add(name)
            peak_workers = max(peak_workers, len(live))

        def poll(self):
            nonlocal gate_completed
            self.polls += 1
            if self.polls >= 2 and self.returncode is None:
                self.returncode = 0
                payload = {"status": "completed"}
                if self.name == "attendance-recovery-gate":
                    gate_completed = True
                    payload["status"] = "passed" if recovered else "failed"
                    self.returncode = 0 if recovered else 1
                dump(jobs[self.name]["result"], payload)
                live.remove(self.name)
            return self.returncode

    monkeypatch.setattr(queue.subprocess, "Popen", Worker)
    monkeypatch.setattr(queue.psutil, "Process", lambda pid: SimpleNamespace(create_time=lambda: 1.0))
    monkeypatch.setattr(queue.psutil, "virtual_memory", lambda: SimpleNamespace(
        total=1536 * 2**30, available=1500 * 2**30))
    monkeypatch.setattr(queue.shutil, "disk_usage", lambda path: SimpleNamespace(free=2000 * 2**30))
    monkeypatch.setattr(queue, "tree_rss", lambda pid: 1.0)
    monkeypatch.setattr(queue.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(queue.signal, "signal", lambda *args: None)
    output = tmp_path / "queue"
    assert queue.run(Namespace(manifest=manifest_path, output=output, detach=False)) == (0 if recovered else 1)
    state = json.loads((output / "state.json").read_text())
    benchmark = {k: v for k, v in state["jobs"].items() if k.startswith("rel-")}
    assert peak_workers <= 8
    assert len(started) == len(set(started))
    if recovered:
        assert all(v["status"] == "completed" for v in benchmark.values())
        assert state["jobs"]["final-official-elo"]["status"] == "completed"
    else:
        assert benchmark["rel-event_user-attendance"]["status"] == "completed"
        assert all(v["status"] == "blocked" for k, v in benchmark.items()
            if k != "rel-event_user-attendance")
        assert state["jobs"]["final-official-elo"]["status"] == "blocked"
