import json
import os
from pathlib import Path
import subprocess
import sys

from scripts.run_production_finalist_cloud_v2 import build_manifest, resource_estimate, route, MODULE, PROFILE


def test_graph_and_audit_routing_preserves_stage_arguments():
    old=["python","-u","-m","scripts.run_kurversc_selected_relarena","--seed","0"]
    assert route(old)==["python","-u","-m",MODULE,"--mode","graph","--","--seed","0"]
    audit=["python","-m","scripts.benchmark_production_finalists","--hours","4"]
    assert route(audit)==["python","-m","scripts.benchmark_production_finalists_v2","--hours","4"]
    assert route(["python","-m","scripts.benchmark_recovered_readouts"])==["python","-m","scripts.benchmark_recovered_readouts"]


def test_all_tasks_get_whole_task_metrics_same_deadline_and_no_old_scores(tmp_path):
    from scripts.run_default_relarena_cloud import task_pairs
    pairs=task_pairs()
    manifest=build_manifest(tmp_path,pairs,["source.py"],cpus=64,max_workers=8,reserve_gib=128,all_tasks=True)
    assert len(pairs)==21
    tasks=[j for j in manifest["jobs"] if "/" in j.get("task","")]
    assert len(tasks)==21 and len({j["task"] for j in tasks})==21
    for j in tasks:
        assert j["command"][3]=="scripts.cloud_task_metrics"
        assert MODULE in j["command"]
        assert j["depends"]==["prepare-"+j["task"].split("/")[0]]
    assert manifest["task_hours"]==23.75 and manifest["per_task_hard_memory_limit"]=="none"
    assert manifest["run_profile"]==PROFILE and not manifest["score_dependent_task_admission"]
    assert manifest["max_workers"]==8
    assert not any("recovery-gate" in j["name"] for j in manifest["jobs"])
    collector=next(j for j in manifest["jobs"] if j["name"]=="final-official-elo")
    assert len(collector["depends"])==22
    reporter=manifest["jobs"][-1]
    assert reporter["name"]=="resource-report" and len(reporter["wait_for_terminal"])==len(manifest["jobs"])-1


def test_resource_estimates_use_only_completed_peak_memory_not_scores(tmp_path):
    path=tmp_path/"metrics/rel-stack_post-votes/attempt-1/resources.json"
    path.parent.mkdir(parents=True)
    report=dict(status="completed",returncode=0,peak_process_tree_rss_gib=100,kernel_memory_peak_gib=110,
        official_test_score="This is not an input to resource scheduling")
    path.write_text(json.dumps(report))
    assert resource_estimate("rel-stack","post-votes",tmp_path)==181
    report.update(status="failed",returncode=1,peak_process_tree_rss_gib=1,kernel_memory_peak_gib=1)
    path.write_text(json.dumps(report))
    assert resource_estimate("rel-stack","post-votes",tmp_path)==180


def test_v2_profile_keeps_all_previous_stage_contracts_and_central_resource_ownership():
    program='''
from scripts import run_staged_complete_cloud as staged
from scripts.run_recovered_staged_cloud import READOUT_CONTRACT
from scripts.run_count_challenger_cloud import COUNT_CONTRACT
from scripts.run_production_finalist_cloud_v2 import install_recipe, PROFILE
from scripts.benchmark_production_finalists_v2 import CONTRACT
original=dict(staged.CONTRACT)
install_recipe()
assert staged.PROFILE==PROFILE
assert staged.CONTRACT=={**original,"recovered_readouts":READOUT_CONTRACT,
    "conditional_count_challenger":COUNT_CONTRACT,"production_finalist_check":CONTRACT}
assert staged.CONTRACT["initial_production_training_frames"]==1
assert staged.CONTRACT["search_parallelism"]==1
assert staged.CONTRACT["temporal_seeds"]==[0,1,2]
print("same complete staged recipe, optional generalized native finalist check")
'''
    result=subprocess.run([sys.executable,"-c",program],cwd=Path(__file__).resolve().parents[1],
        env=dict(os.environ),capture_output=True,text=True)
    assert result.returncode==0,result.stderr
