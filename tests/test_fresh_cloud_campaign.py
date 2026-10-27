from argparse import Namespace
from pathlib import Path
import json
from types import SimpleNamespace

import pytest

from scripts import run_fresh_cloud_campaign as cloud
from scripts.run_fresh_cloud_campaign import dispatch, graph_arguments, task_pairs


def task_metadata_fixture(tmp_path, monkeypatch):
    """Legacy scheduler tests need task names, never historical scores/plans."""
    import csv
    from scripts.run_default_relarena_cloud import task_pairs as official_pairs

    path = tmp_path / "metadata/docs/checkpoints/20261007/frozen-graph-campaign-21.csv"
    path.parent.mkdir(parents=True)
    with path.open("w") as stream:
        writer = csv.writer(stream)
        writer.writerow(["dataset", "task"])
        writer.writerows(official_pairs())
    monkeypatch.setattr(cloud, "ROOT", tmp_path / "metadata")


def test_all_21_tasks_are_unique_and_interleaved(tmp_path, monkeypatch):
    task_metadata_fixture(tmp_path, monkeypatch)
    pairs = task_pairs()
    assert len(pairs) == len(set(pairs)) == 21
    assert len({dataset for dataset, _ in pairs[:7]}) == 7


def test_fresh_search_preserves_stable_recipe_and_has_no_old_winner():
    args = Namespace(output=Path('/fresh'), dataset='rel-f1', task='driver-dnf', search_workers=2)
    values = graph_arguments(args)
    assert values[values.index('--profile') + 1] == 'staged-retention'
    assert values[values.index('--full-training-frames') + 1] == '1'
    assert values[values.index('--base-cdf') + 1] == 'on'
    assert values[values.index('--peer-context-mode') + 1] == 'both'
    assert values[values.index('--duckdb-memory-limit') + 1] == 'none'
    assert '--fixed-trial-checkpoint' not in values
    assert '--fixed-config-json' not in values


def test_stages_are_redirected_to_this_fresh_graph_root():
    original = ['python', '-u', '-m', 'scripts.benchmark_sparse_training', '--output', '/cache']
    actual = dispatch(original, Path('/fresh/graphs'))
    assert actual[3:10] == ['scripts.run_fresh_cloud_campaign', '--mode', 'stage', '--graph-root', '/fresh/graphs', '--stage-module', 'scripts.benchmark_sparse_training']
    assert actual[10:] == ['--', '--output', '/cache']
    assert original[3] == 'scripts.benchmark_sparse_training'


@pytest.mark.parametrize("seed_workers", [1, 2, 20])
def test_frozen_audit_keeps_veto_and_fidelity_without_parallel_state(seed_workers):
    args = ['--origins', '3', '--seeds', '0,1,2', '--threads', '4']
    actual = dispatch(['python', '-m', 'scripts.benchmark_temporal_policy', *args],
                      Path('/fresh/graphs'), seed_workers=seed_workers)
    assert 'scripts.benchmark_temporal_policy' in actual
    assert 'scripts.run_temporal_audit_parallel' not in actual
    assert actual[actual.index('--') + 1:] == [*args, '--prune-vetoed']


def test_isolated_engine_spill_limit_scales_with_machine_memory():
    assert cloud.duckdb_limit_for_task(1500 * 1024**3, isolated=True) == '750GiB'
    assert cloud.duckdb_limit_for_task(1500 * 1024**3, isolated=False) == 'none'


def write_result(root, key, **overrides):
    dataset, task = key.split('/')
    result = dict(dataset=dataset, task=task, metric='roc_auc', arm='incumbent',
                  official_test_scores=dict(control=0.8, candidate=0.8))
    result.update(overrides)
    cloud.atomic_json(result, cloud.official_path(root, key))


def test_zero_exit_without_official_artifact_is_not_completed(tmp_path):
    assert cloud.completion_status(tmp_path, 'rel-hm/user-churn', 0) == 'failed'


def test_guarded_zero_exit_is_resource_deferred_even_with_result(tmp_path):
    write_result(tmp_path, 'rel-hm/user-churn')
    assert cloud.completion_status(tmp_path, 'rel-hm/user-churn', 0,
                                   'global_resource_pressure') == 'resource_deferred'


def test_valid_official_result_and_zero_exit_are_both_required(tmp_path):
    write_result(tmp_path, 'rel-f1/driver-dnf')
    assert cloud.completion_status(tmp_path, 'rel-f1/driver-dnf', 0) == 'completed'
    assert cloud.completion_status(tmp_path, 'rel-f1/driver-dnf', 1) == 'failed'
    assert cloud.completion_status(tmp_path, 'rel-f1/driver-dnf', 0,
                                   'total_task_time_budget') == 'time_budget_exceeded'


def test_opt_in_requires_signal_completion_not_only_baseline(tmp_path):
    cloud.atomic_json(dict(identity=dict(signal_challengers=True)), tmp_path/'campaign-status.json')
    write_result(tmp_path, 'rel-f1/driver-dnf')
    assert cloud.official_result(tmp_path, 'rel-f1/driver-dnf') is None
    baseline = cloud.official_path(tmp_path, 'rel-f1/driver-dnf')
    payload = json.loads(baseline.read_text())
    cloud.atomic_json(payload, baseline.parent.parent/'signal-challengers/results.json')
    assert cloud.official_result(tmp_path, 'rel-f1/driver-dnf') is not None


def test_protected_isolated_wave_blocks_admission(tmp_path):
    path = tmp_path/'protected.json'
    assert cloud.protected_campaign_busy(None) is False
    assert cloud.protected_campaign_busy(path) is True
    cloud.atomic_json(dict(tasks={'a':dict(status='running',isolated=False)}),path)
    assert cloud.protected_campaign_busy(path) is False
    assert cloud.protected_campaign_busy(path, isolated=True) is True
    cloud.atomic_json(dict(tasks={'a':dict(status='running',isolated=True)}),path)
    assert cloud.protected_campaign_busy(path) is True
    cloud.atomic_json(dict(tasks={'a':dict(status='completed',isolated=True)}),path)
    assert cloud.protected_campaign_busy(path, isolated=True) is False


@pytest.mark.parametrize('overrides', [
    dict(dataset='wrong'), dict(task='wrong'), dict(metric='wrong'),
    dict(official_test_scores=dict(control=0.8, candidate=float('nan'))),
    dict(official_test_scores=dict(control=0.8)),
])
def test_invalid_official_artifact_is_not_a_success(tmp_path, overrides):
    write_result(tmp_path, 'rel-f1/driver-dnf', **overrides)
    assert cloud.official_result(tmp_path, 'rel-f1/driver-dnf') is None


def test_summary_does_not_crash_if_stale_state_claims_missing_success(tmp_path):
    state = dict(tasks={'rel-hm/user-churn': dict(status='completed')})
    cloud.summarize(tmp_path, state)
    assert (tmp_path / 'campaign-results.csv').read_text().count('\n') == 1


def test_resume_keeps_results_and_accounts_for_interrupted_runtime(tmp_path):
    pairs = [('rel-event', 'user-ignore'), ('rel-hm', 'user-churn')]
    write_result(tmp_path, 'rel-event/user-ignore')
    state = dict(started_at='2026-10-07T10:00:00+00:00',
                 resources=dict(observed_at='2026-10-07T11:00:00+00:00'),
                 tasks={
                     'rel-event/user-ignore': dict(status='completed', elapsed_seconds=123),
                     'rel-hm/user-churn': dict(status='running', pid=999,
                                               started_at='2026-10-07T10:30:00+00:00',
                                               consumed_seconds=60),
                 })
    resumed = cloud.reconcile_resume(tmp_path, state, pairs)
    assert resumed['tasks']['rel-event/user-ignore']['elapsed_seconds'] == 123
    assert resumed['tasks']['rel-event/user-ignore']['status'] == 'completed'
    interrupted = resumed['tasks']['rel-hm/user-churn']
    assert interrupted['status'] == 'queued'
    assert interrupted['consumed_seconds'] == 1860
    assert 'pid' not in interrupted
    assert cloud.reconcile_resume(tmp_path, resumed, pairs)['tasks']['rel-hm/user-churn']['consumed_seconds'] == 1860


def test_18_task_resume_uses_ten_slots_then_isolates_hm(tmp_path, monkeypatch):
    task_metadata_fixture(tmp_path, monkeypatch)
    pairs = task_pairs()
    completed = {'rel-avito/ad-ctr', 'rel-event/user-ignore', 'rel-event/user-attendance'}
    identity = dict(kurversc_commit=cloud.RSC_COMMIT, graphreduce_commit=cloud.GR_COMMIT,
                    search_parallelism=2, seed_parallelism=2, task_hours=23,
                    profile='staged-retention', base_cdf=True, peer_context='both',
                    full_training_frames=1, temporal_origins=3, temporal_seeds=[0, 1, 2])
    state = dict(identity=identity, started_at='2026-10-07T10:00:00+00:00',
                 tasks={f'{d}/{t}': dict(status='completed' if f'{d}/{t}' in completed else 'queued',
                                        elapsed_seconds=10, peak_tree_rss_gib=1) for d, t in pairs})
    for key in completed:
        write_result(tmp_path, key)
    cloud.atomic_json(state, tmp_path / 'campaign-status.json')
    alive, launches, counts = set(), [], []

    class Worker:
        def __init__(self, key):
            self.key, self.pid, self.returncode, self.polls = key, 10000 + len(launches), None, 0
            alive.add(key)
            launches.append(key)
            counts.append(len(alive))
            if key == 'rel-hm/user-churn':
                assert alive == {key}

        def poll(self):
            self.polls += 1
            if self.polls >= 4:
                self.returncode = 0
                alive.discard(self.key)
                write_result(tmp_path, self.key)
            return self.returncode

    def popen(command, **kwargs):
        key = f"{command[command.index('--dataset') + 1]}/{command[command.index('--task') + 1]}"
        assert key not in completed
        return Worker(key)

    monkeypatch.setattr(cloud.subprocess, 'Popen', popen)
    monkeypatch.setattr(cloud, 'tree_memory', lambda pid: (1, []))
    monkeypatch.setattr(cloud.psutil, 'Process', lambda *a: SimpleNamespace(cpu_affinity=lambda *a: list(range(192))))
    monkeypatch.setattr(cloud.psutil, 'virtual_memory', lambda: SimpleNamespace(available=1500 * 1024**3, total=1500 * 1024**3))
    monkeypatch.setattr(cloud.shutil, 'disk_usage', lambda root: SimpleNamespace(free=2000 * 1024**3))
    monkeypatch.setattr(cloud.time, 'sleep', lambda seconds: None)
    monkeypatch.setattr(cloud, 'prepare_dataset', lambda *a: pytest.fail('Resume must not download again'))
    args = SimpleNamespace(output=tmp_path, resume=True, task_workers=10, search_workers=2,
                           seed_workers=2, task_hours=23, isolated_task=['rel-hm/user-churn'],
                           isolated_search_workers=1, admission_ram_gib=384, critical_ram_gib=320,
                           poll_seconds=2)
    assert cloud.campaign_main(args) == 0
    assert max(counts) == 10
    assert len(launches) == len(set(launches)) == 18
    assert launches[-1] == 'rel-hm/user-churn'
    final = json.loads((tmp_path / 'campaign-status.json').read_text())
    assert all(task['status'] == 'completed' for task in final['tasks'].values())
