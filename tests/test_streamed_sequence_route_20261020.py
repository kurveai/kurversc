from scripts.run_streamed_sequence_cloud import cache_route, install_execution_contract
from types import SimpleNamespace


def test_execution_wrapper_routes_direct_and_dispatched_capture_only():
    for command in (
        ["python", "-m", "scripts.benchmark_sparse_training", "--materialize-only"],
        ["python", "-m", "scripts.run_fresh_cloud_campaign", "--mode", "stage",
            "--stage-module", "scripts.benchmark_sparse_training", "--", "--materialize-only"],
    ):
        result = cache_route(command, previous=list)
        assert result == [*command, "--stream-feature-cache"]
        assert cache_route(result, previous=list) == result
        assert "--stream-feature-cache" not in command
    graph = ["python", "-m", "scripts.run_kurversc_selected_relarena", "--seed", "0"]
    assert cache_route(graph, previous=list) == graph
    unrelated = ["python", "-m", "scripts.benchmark_production_sequence"]
    assert cache_route(unrelated, previous=list) == unrelated


def test_storage_contract_leaves_predictive_contract_intact():
    staged = SimpleNamespace(PROFILE="old", CONTRACT=dict(seed=0, training_frames=1,
        temporal_estimator_mode="frame_ensemble", model_threads=8))
    old = staged.CONTRACT.copy()
    install_execution_contract(staged)
    assert {k: staged.CONTRACT[k] for k in old} == old
    assert staged.CONTRACT["historical_feature_cache_storage"]["changed_source_sampling"] is False


def test_portable_and_cloud_share_one_recipe_installer(monkeypatch):
    from scripts import run_streamed_sequence_cloud as streamed
    from scripts import run_production_finalist_cloud_v2 as base
    from scripts import run_sequence_recovery_cloud as sequence
    from scripts import run_fresh_cloud_campaign as fresh
    old_route = lambda c: [*c, "original_route"]
    monkeypatch.setattr(base, "route", old_route)
    calls = []
    staged = SimpleNamespace(CONTRACT=dict(task_hours=23.75))
    def extend(value, worker):
        assert value is staged and worker is fresh
        calls.append("sequence")
        value.CONTRACT["production_sequence_readout"] = dict(optional=True)
        return value
    monkeypatch.setattr(sequence, "extend_recipe", extend)
    result = streamed.install_recipe(install_base=lambda: staged)
    assert result is staged and calls == ["sequence"]
    assert result.CONTRACT["production_sequence_readout"]["optional"]
    assert result.CONTRACT["historical_feature_cache_storage"]["cohort_order_preserved"]
    command = ["python", "-m", "scripts.benchmark_sparse_training", "--materialize-only"]
    assert base.route(command) == [*command, "original_route", "--stream-feature-cache"]
