import json
from pathlib import Path

import pytest

from scripts.cloud_task_metrics import dedicated_cgroup
from scripts.run_measured_staged_cloud import measured_manifest, resources, validate


def simple_manifest():
    return dict(reserve_gib=128, task_hours=23.75, max_workers=8, max_threads=64, jobs=[
        dict(name="prepare-rel-f1", command=["{python}", "prepare"], ram_gib=2, threads=1),
        dict(name="rel-f1_driver-dnf", task="rel-f1/driver-dnf", command=["{python}", "task"],
             ram_gib=8, threads=8, depends=["prepare-rel-f1", "readout-recovery-gate"]),
        dict(name="readout-recovery-gate", command=["{python}", "gate"], ram_gib=1, threads=1),
        dict(name="final-official-elo", command=["{python}", "collect"], ram_gib=4, threads=1,
             depends=["rel-f1_driver-dnf"])])


def test_resource_instrumentation_does_not_change_task_procedure():
    original = simple_manifest()
    measured = measured_manifest(original, Path("/fresh"), 64)
    validate(measured)
    assert original["reserve_gib"] == 128
    assert measured["reserve_gib"] == 64
    task = next(j for j in measured["jobs"] if j["name"] == "rel-f1_driver-dnf")
    assert task["command"][-2:] == ["{python}", "task"]
    assert task["depends"] == ["prepare-rel-f1"]
    assert task["ram_gib"] == 8 and task["threads"] == 8
    assert not any(j["name"] == "readout-recovery-gate" for j in measured["jobs"])
    assert "final-official-elo" in measured["jobs"][-1]["wait_for_terminal"]


def test_node_reserve_and_task_deadline_cannot_be_disabled():
    m = measured_manifest(simple_manifest(), Path("/fresh"), 63)
    with pytest.raises(ValueError, match="64 GiB"):
        validate(m)
    m["reserve_gib"] = 64
    m["task_hours"] = 24
    with pytest.raises(ValueError):
        validate(m)


def test_metrics_cannot_accidentally_report_the_whole_node(tmp_path):
    for entry in ["0::/", "0::/user.slice", "0::/somebody-else.scope", "0::/kurversc-task-x.scope/../../oops.scope"]:
        with pytest.raises(ValueError):
            dedicated_cgroup(entry, tmp_path)
    result = dedicated_cgroup("0::/user.slice/kurversc-task-123.scope", tmp_path)
    assert result == tmp_path / "user.slice/kurversc-task-123.scope"


def test_resource_summary_retains_peaks_and_times_across_retries(tmp_path):
    (tmp_path / "queue").mkdir()
    task = dict(status="completed", command=["python", "scripts.cloud_task_metrics"])
    (tmp_path / "queue/state.json").write_text(json.dumps(dict(jobs={"rel-f1_driver-dnf": task})))
    for i, (rss, charged, duration) in enumerate([(8, 10, 30), (6, 9, 20)]):
        folder = tmp_path / "metrics/rel-f1_driver-dnf" / f"attempt-{i}"
        folder.mkdir(parents=True)
        (folder / "resources.json").write_text(json.dumps(dict(elapsed_seconds=duration,
            peak_process_tree_rss_gib=rss, kernel_memory_peak_gib=charged, finished_at="done")))
    resources(tmp_path)
    report = json.loads((tmp_path / "resource-summary.json").read_text())["tasks"][0]
    assert report["elapsed_seconds"] == 50 and report["attempts"] == 2
    assert report["peak_process_tree_rss_gib"] == 8
    assert report["kernel_task_group_peak_gib"] == 10
    assert report["per_task_hard_memory_limit"] == "none"
