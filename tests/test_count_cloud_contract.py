import os
from pathlib import Path
import subprocess
import sys


def test_count_recipe_preserves_every_previous_staged_contract_field():
    program = '''
from scripts import run_staged_complete_cloud as staged
from scripts.run_recovered_staged_cloud import READOUT_CONTRACT
from scripts.run_count_challenger_cloud import install_recipe, COUNT_CONTRACT, PROFILE
previous = dict(staged.CONTRACT)
install_recipe()
expected = {**previous, "recovered_readouts":READOUT_CONTRACT,
            "conditional_count_challenger":COUNT_CONTRACT}
assert staged.CONTRACT == expected
assert staged.PROFILE == PROFILE
assert staged.CONTRACT["initial_production_training_frames"] == 1
assert staged.CONTRACT["search_parallelism"] == 1
assert staged.CONTRACT["temporal_origins"] == 3
assert staged.CONTRACT["task_hours"] < 24
assert not COUNT_CONTRACT["historical_winner_lookup"]
assert not COUNT_CONTRACT["graph_search_expanded"]
print("contract unchanged except optional count readout")
'''
    completed = subprocess.run([sys.executable, "-c", program], cwd=Path(__file__).resolve().parents[1],
        env=dict(os.environ), capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    assert "contract unchanged" in completed.stdout


def test_unsupported_task_retains_the_preserved_pipeline_result(tmp_path):
    program = '''
import argparse,json
from pathlib import Path
from scripts import run_fresh_cloud_campaign as fresh
from scripts import run_recovered_staged_cloud as recovered
from scripts.run_count_challenger_cloud import count_root, install_recipe
root = Path(__import__("sys").argv[1])
key = "new-schema/unknown-task"
folder = recovered.readout_root(root,key)
folder.mkdir(parents=True)
baseline = dict(selected="incumbent",metric="roc_auc",official_test_scores=dict(control=.9,candidate=.9))
(folder/"results.json").write_text(json.dumps(baseline))
# The upstream recovered stage is already complete in this focused test.
# Its real lifecycle checks this manifest before reusing the saved result.
(folder.parent/"raw-audit").mkdir()
(folder.parent/"raw-audit/manifest.json").write_text("{}")
(root/"campaign-status.json").write_text(json.dumps(dict(tasks={key:dict(consumed_seconds=0)})))
fresh.task_main = lambda args: 0
install_recipe()
assert fresh.task_main(argparse.Namespace(output=root,dataset="new-schema",task="unknown-task",task_hours=23.75)) == 0
output = count_root(root,key)
admission = json.loads((output/"admission.json").read_text())
result = json.loads((output/"results.json").read_text())
assert not admission["applicable"] and not admission["admitted"]
assert result["official_test_scores"] == baseline["official_test_scores"]
assert result["count_challenger_retained_incumbent"]
print("unsupported domain retained exactly")
'''
    completed = subprocess.run([sys.executable,"-c",program,str(tmp_path)],cwd=Path(__file__).resolve().parents[1],
        env=dict(os.environ),capture_output=True,text=True)
    assert completed.returncode == 0, completed.stderr
    assert "retained exactly" in completed.stdout
