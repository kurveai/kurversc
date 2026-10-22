import copy
from contextlib import contextmanager
import json
import sys

import pytest

from scripts.run_official_cloud import learner_replay_parameters


def test_legacy_behavior_is_identical_without_opt_in():
    assert learner_replay_parameters(thread_budget=8)==dict(thread_count=8)
    assert learner_replay_parameters(thread_budget=48)==dict(thread_count=48)


def test_validation_parameters_are_preserved_not_rescaled_to_host_resources():
    manifest = dict(model_params=dict(thread_count=4,iterations=300,depth=6))
    before = copy.deepcopy(manifest)
    assert learner_replay_parameters(thread_budget=20,validation_manifest=manifest)==manifest["model_params"]
    assert learner_replay_parameters(thread_budget=4,validation_manifest=manifest)==manifest["model_params"]
    assert manifest==before
    with pytest.raises(ValueError,match="CPU budget"):
        learner_replay_parameters(thread_budget=2,validation_manifest=manifest)


@pytest.mark.parametrize("params",[{},dict(thread_count=-1),dict(thread_count=True)])
def test_strict_replay_cannot_guess_missing_or_unbounded_fit_parameters(params):
    with pytest.raises(ValueError,match="explicit validation"):
        learner_replay_parameters(thread_budget=8,validation_manifest=dict(model_params=params))


@pytest.mark.parametrize("strict,expected_threads",[(False,8),(True,4)])
def test_wrapper_forwards_and_restores_both_estimator_bindings(tmp_path,monkeypatch,strict,expected_threads):
    from scripts import run_official_cloud as runner
    output = tmp_path/"output"
    evidence = tmp_path/"validation/results.json"
    evidence.parent.mkdir()
    evidence.write_text(json.dumps(dict(selected_policy=dict(readout="direct"))))
    (evidence.parent/"manifest.json").write_text(json.dumps(dict(model_params=dict(thread_count=4,depth=6))))
    calls,statements = [],[]

    def core_factory(*args,**kwargs):
        calls.append(kwargs["model_params"])
        return "fitted"

    def separate_binding(*args,**kwargs):
        raise AssertionError("Only the wrapper's underlying core factory should be called")

    class Connection:
        def sql(self,statement):
            statements.append(statement)

    @contextmanager
    def scope(*args,**kwargs):
        yield Connection()

    monkeypatch.setattr(runner.core,"fit_final_catboost",core_factory)
    monkeypatch.setattr(runner.benchmark,"fit_final_catboost",separate_binding)
    monkeypatch.setattr(runner.core,"_connection_scope",scope)

    def benchmark():
        assert runner.core.fit_final_catboost(model_params=dict(iterations=300))=="fitted"
        assert runner.benchmark.fit_final_catboost(model_params=None)=="fitted"
        with runner.core._connection_scope():
            pass

    monkeypatch.setattr(runner.benchmark,"main",benchmark)
    argv = ["wrapper","--threads","8","--duckdb-threads","8","--policy-audit",str(evidence),"--output",str(output)]
    if strict:
        argv.append("--replay-validation-model-params")
    monkeypatch.setattr(sys,"argv",argv)
    runner.main()
    assert all(call["thread_count"]==expected_threads for call in calls)
    assert calls[0]["iterations"]==300
    assert ("depth" in calls[0])==strict
    assert statements==["SET threads=8"]
    assert runner.core.fit_final_catboost is core_factory
    assert runner.benchmark.fit_final_catboost is separate_binding
    assert runner.core._connection_scope is scope
    execution = json.loads((output/"cloud-execution.json").read_text())
    assert execution["catboost_threads"]==expected_threads
