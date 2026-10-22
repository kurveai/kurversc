import json
from types import SimpleNamespace

import pytest

from kurversc.staged import run_staged_task
from test_task_context import context


def setup(monkeypatch,tmp_path,*,running=False):
    from kurversc import staged as module
    bundle = context().export(tmp_path/"input")
    output = tmp_path/"run"
    output.mkdir()
    calls,signals = [],[]
    class Child:
        pid = 417
        returncode = None if running else 0
        def __init__(self,command,**kwargs):
            calls.append((command,kwargs))
            (output/"portable-results.json").write_text(json.dumps(dict(status="completed")))
        def poll(self):return self.returncode
        def wait(self,timeout=None):self.returncode=0;return 0
    monkeypatch.setattr(module,"_runtime_entry",lambda:("scripts.run_portable_staged",tmp_path))
    monkeypatch.setattr(module.subprocess,"Popen",Child)
    monkeypatch.setattr(module.os,"killpg",lambda pid,sig:signals.append((pid,sig)))
    monkeypatch.setattr(module.time,"time",lambda:1000.)
    return bundle,output,calls,signals


def test_supervisor_isolates_research_environment_and_owns_only_its_worker(monkeypatch,tmp_path):
    bundle,output,calls,signals = setup(monkeypatch,tmp_path)
    monkeypatch.setenv("KURVERSC_CATBOOST_ITERATIONS","1")
    monkeypatch.setenv("KURVERSC_INCLUDE_TARGET_HISTORY","0")
    result = run_staged_task(bundle,output=output,task_hours=1)
    assert result.report["status"]=="completed"
    _,kwargs = calls[0]
    assert kwargs["start_new_session"] is True
    assert kwargs["env"]["KURVERSC_TASK_CONTEXT"]==str(bundle)
    assert "KURVERSC_CATBOOST_ITERATIONS" not in kwargs["env"]
    assert "KURVERSC_INCLUDE_TARGET_HISTORY" not in kwargs["env"]
    assert signals and {pid for pid,_ in signals}=={417}


def test_resource_stop_preserves_checkpoint_and_does_not_invent_a_completed_result(monkeypatch,tmp_path):
    import psutil
    bundle,output,calls,signals = setup(monkeypatch,tmp_path,running=True)
    checkpoint = output/"important-checkpoint"
    checkpoint.write_bytes(b"preserved")
    monkeypatch.setattr(psutil,"virtual_memory",lambda:SimpleNamespace(available=0))
    with pytest.raises(MemoryError,match="only this task"):
        run_staged_task(bundle,output=output,task_hours=1)
    assert checkpoint.read_bytes()==b"preserved"
    assert json.loads((output/"resource-stop.json").read_text())["reason"]=="available_memory_reserve"
    assert {pid for pid,_ in signals}=={417}


def test_exhausted_original_deadline_cannot_restart_a_worker(monkeypatch,tmp_path):
    bundle,output,calls,_ = setup(monkeypatch,tmp_path)
    (output/"portable-state.json").write_text(json.dumps(dict(task_hours=1,
        first_started_epoch=0,deadline_epoch=999)))
    with pytest.raises(TimeoutError,match="original"):
        run_staged_task(bundle,output=output,task_hours=1)
    assert not calls
