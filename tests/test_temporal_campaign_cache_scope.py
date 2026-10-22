import json

import pytest

from scripts import run_temporal_policy_campaign as campaign


def test_legacy_cache_cannot_override_the_current_campaign_graph(tmp_path, monkeypatch):
    repository=tmp_path/"repository"
    legacy=repository/"results/lifecycle-policy-cache-20261005/user-clicks"
    legacy.mkdir(parents=True)
    (legacy/"materialization.json").write_text(json.dumps(dict(dataset="rel-avito",task="user-clicks",
        checkpoint="an internally valid but unrelated old graph")))
    monkeypatch.setattr(campaign,"ROOT",repository)
    current=tmp_path/"new-run"
    assert campaign.current_run_cache(current,"rel-avito","user-clicks")==current/"tasks/rel-avito_user-clicks/cache"
    assert campaign.current_run_cache(current,"rel-avito","user-clicks")!=legacy


def test_same_run_resume_keeps_its_cache_and_different_runs_never_share_by_task(tmp_path):
    a=campaign.current_run_cache(tmp_path/"a","generic","task")
    a.mkdir(parents=True)
    (a/"materialization.json").write_text("{}")
    assert campaign.current_run_cache(tmp_path/"a","generic","task")==a
    assert campaign.current_run_cache(tmp_path/"b","generic","task")!=a


@pytest.mark.parametrize("dataset,task", [("../other","task"),("dataset","/task"),("","task")])
def test_cache_scope_rejects_escaped_task_paths(tmp_path,dataset,task):
    with pytest.raises(ValueError):campaign.current_run_cache(tmp_path,dataset,task)
