"""Fresh single-task or all-21 reproduction of the bounded finalist recipe.

Full current staged search, all later selectors and official scoring remain
required. The optional native finalist check supports arbitrary validation
block lengths. Resource profiles influence scheduling only, never models.
"""

import argparse
from datetime import datetime, timezone
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import subprocess
import sys

from scripts.benchmark_production_finalists_v2 import CONTRACT as FINALIST_CONTRACT
from scripts.run_gap_followup_queue import digest, dump


PROFILE="staged-complete-production-finalists-count-v2"
MODULE="scripts.run_production_finalist_cloud_v2"


def route(command):
    command=list(command)
    if "-m" not in command:return command
    i=command.index("-m")+1
    if command[i]=="scripts.run_kurversc_selected_relarena":
        command[i:i+1]=[MODULE,"--mode","graph","--"]
    elif command[i]=="scripts.benchmark_production_finalists":
        command[i]="scripts.benchmark_production_finalists_v2"
    return command


def install_guard():
    from scripts import run_kurversc_local_campaign as guards
    original=guards._run_guarded
    def guarded(command,**kwargs):
        # One resource coordinator owns host intervention, not independent
        # in-task guards which could simultaneously cancel multiple tasks.
        kwargs["minimum_available_memory_gb"]=0
        return original(route(command),**kwargs)
    guards._run_guarded=guarded


def install_recipe():
    from scripts import run_count_challenger_cloud as count
    install_guard()
    staged=count.install_recipe()
    staged.PROFILE=PROFILE
    staged.CONTRACT={**staged.CONTRACT,"production_finalist_check":FINALIST_CONTRACT}
    return staged


def resource_estimate(dataset,task,reference=None):
    from scripts import run_staged_complete_cloud as staged
    baseline=staged.ram_estimate(dataset,task)
    if reference is None:return baseline
    paths=sorted((Path(reference)/f"metrics/{dataset}_{task}").glob("attempt-*/resources.json"))
    reports=[json.loads(p.read_text()) for p in paths]
    complete=[r for r in reports if r.get("status")=="completed" and r.get("returncode")==0]
    if not complete:return baseline
    peak=max(max(r["peak_process_tree_rss_gib"],r.get("kernel_memory_peak_gib",0)) for r in complete)
    # Additional fits are sequential. Keep a material margin for the optional
    # check, semantic runtime and unsampled source-frame materialization.
    return max(8 if dataset=="rel-f1" else 64,round(1.5*peak+16,1))


def build_manifest(output,pairs,sources,*,cpus,max_workers,reserve_gib,all_tasks,resource_reference=None):
    from scripts import run_default_relarena_cloud as base
    def command(mode,*rest):
        return ["{python}","-u","-m",MODULE,"--mode",mode,"--output",str(output),*rest]
    datasets=sorted({d for d,_ in pairs})
    jobs=[dict(name="prepare-"+d,command=command("prepare","--dataset",d),ram_gib=2,threads=1,
        result=str(output/"datasets"/d/"results.json")) for d in datasets]
    if all_tasks:
        jobs.append(dict(name="leaderboard-snapshot",command=command("snapshot"),ram_gib=2,threads=1,
            result=str(output/"baseline/results.json")))
    for d,t in sorted(pairs,key=lambda p:(resource_estimate(*p,resource_reference),*p)):
        slug=f"{d}_{t}"
        jobs.append(dict(name=slug,task=f"{d}/{t}",ram_gib=resource_estimate(d,t,resource_reference),threads=8,
            command=["{python}","-u","-m","scripts.cloud_task_metrics","--output",str(output/"metrics"/slug),
                "--",*command("task","--dataset",d,"--task",t)],depends=["prepare-"+d],
            result=str(output/"tasks"/slug/"results.json")))
    if all_tasks:
        jobs.append(dict(name="final-official-elo",command=command("collect"),ram_gib=4,threads=1,
            depends=["leaderboard-snapshot",*[f"{d}_{t}" for d,t in pairs]],result=str(output/"elo/results.json")))
    jobs.append(dict(name="resource-report",command=command("resources"),ram_gib=1,threads=1,
        wait_for_terminal=[j["name"] for j in jobs],result=str(output/"resource-summary.json")))
    return dict(sources=list(sources),reserve_gib=reserve_gib,task_hours=23.75,
        max_workers=min(max_workers,max(1,cpus//8)),max_threads=cpus,jobs=jobs,run_profile=PROFILE,
        scope="Fresh full staged reproduction; no carried-forward scores, graphs or predictions",
        score_dependent_task_admission=False,per_task_hard_memory_limit="none")


def launch(args):
    from scripts import run_default_relarena_cloud as base
    from scripts.run_measured_staged_cloud import validate
    staged=install_recipe()
    pairs=(base.task_pairs() if args.all_tasks else
        [tuple(p.strip().split("/")) for p in args.task_subset.split(",")] if args.task_subset
        else [(args.dataset,args.task)])
    if len(pairs)!=len(set(pairs)):raise ValueError("Duplicate tasks are not permitted")
    if not set(pairs)<=set(base.task_pairs()):raise ValueError("Official task set required")
    identity,sources,cpus=base.initialize(args.output,pairs)
    sources=sorted(set(sources)|set((base.ROOT/"scripts").glob("*.py"))|
        set(Path(__file__).parent.glob("*.py"))|{base.ROOT/"docs/checkpoints/20261007/frozen-graph-campaign-21.csv"})
    runtime=None
    if args.runtime_artifacts:
        runtime=json.loads((args.runtime_artifacts/"semantic-smoke.json").read_text())
        if runtime["status"]!="passed" or runtime["maximum_independent_error"]!=0:
            raise ValueError("Complete semantic runtime has not reproduced")
        sources=sorted(set(sources)|{args.runtime_artifacts/name for name in
            ("environment.json","installed-packages.txt","semantic-smoke.json")}|
            {Path(p) for p in runtime["weights_and_tokenizer_sha256"]})
    identity.update(run_profile=PROFILE,effective_runtime_config=staged.CONTRACT,
        source_sha256={str(p):digest(p) for p in sources},complete_semantic_runtime=runtime,
        packages_complete={d.metadata["Name"]:d.version for d in metadata.distributions()},
        execution_environment={k:os.environ[k] for k in ("HF_HUB_CACHE","TOKENIZERS_PARALLELISM",
            "OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS") if k in os.environ},
        score_dependent_task_admission=False,
        resource_reference=str(args.resource_reference) if args.resource_reference else None,
        scope="Fresh full staged task(s), bounded production-faithful finalists; no reused model result")
    dump(args.output/"source-identity.json",identity)
    manifest=build_manifest(args.output,pairs,identity["source_sha256"],cpus=cpus,
        max_workers=args.max_workers,reserve_gib=args.reserve_gib,all_tasks=args.all_tasks,
        resource_reference=args.resource_reference)
    validate(manifest)
    dump(args.output/"manifest.json",manifest)
    queue=args.output/"queue"
    queue.mkdir()
    command=[sys.executable,"-u","-m",MODULE,"--mode","queue","--output",str(args.output)]
    with (queue/"coordinator.log").open("a") as log:
        child=subprocess.Popen(command,cwd=base.ROOT,env=dict(os.environ),stdout=log,
            stderr=subprocess.STDOUT,start_new_session=True)
    dump(queue/"pid.json",dict(pid=child.pid,command=command))
    print(json.dumps(dict(pid=child.pid,output=str(args.output),profile=PROFILE,tasks=len(pairs),
        max_workers=manifest["max_workers"],reserve_gib=args.reserve_gib,
        launched_at=datetime.now(timezone.utc).isoformat())),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode",choices=("launch","queue","prepare","task","graph","snapshot","collect","resources"),default="launch")
    parser.add_argument("--dataset")
    parser.add_argument("--task")
    parser.add_argument("--all-tasks",action="store_true")
    parser.add_argument("--task-subset",help="Explicit comma-separated dataset/task pairs")
    parser.add_argument("--output",type=Path)
    parser.add_argument("--runtime-artifacts",type=Path)
    parser.add_argument("--resource-reference",type=Path)
    parser.add_argument("--max-workers",type=int,default=8)
    parser.add_argument("--reserve-gib",type=float,default=128.)
    args,remaining=parser.parse_known_args()
    if args.mode=="graph":
        from scripts.run_production_finalist_cloud import graph_stage
        install_guard()
        return graph_stage(remaining[1:] if remaining and remaining[0]=="--" else remaining)
    if remaining or args.output is None:parser.error("Output and no unrecognized arguments required")
    if args.max_workers<1 or args.reserve_gib<96:parser.error("Positive worker count and >=96 GiB reserve required")
    if args.mode=="launch" and not (args.all_tasks or args.task_subset or (args.dataset and args.task)):
        parser.error("Explicit single task or --all-tasks required")
    if sum(bool(v) for v in (args.all_tasks,args.task_subset,args.dataset or args.task))>1:
        parser.error("Single task, task subset and --all-tasks are mutually exclusive")
    args.output=args.output.resolve()
    if args.mode=="launch":return launch(args)
    if args.mode in {"queue","resources"}:
        from scripts.run_measured_staged_cloud import run_queue,resources
        return {"queue":run_queue,"resources":resources}[args.mode](args.output)
    staged=install_recipe()
    if args.mode=="snapshot":
        from scripts.run_default_relarena_cloud import snapshot
        return snapshot(args)
    return {"prepare":staged.prepare,"task":staged.task,"collect":staged.collect}[args.mode](args)


if __name__=="__main__":main()
