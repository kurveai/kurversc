"""Refresh current README performance tables from the frozen public evidence.

Run from a source checkout after adding the dated score, panel and scorer JSON
supplements. This only updates documentation; it never executes or selects a
model. Earlier README results are retained in a dated archive.
"""

import argparse
import csv
import json
import math
from pathlib import Path


NAMES = {
    "kurversc": "KurveRSC 0.2.0 staged",
    "tabpfn-rel-client-2026-09-28": "TabPFN-Rel client (2026-09-28)",
    "tabpfn-rel-local-2026-09-28": "TabPFN-Rel local (2026-09-28)",
    "tabpfn-rel-client-2026-08-15": "TabPFN-Rel client (2026-08-15)",
    "tabpfn-rel-local-2026-08-15": "TabPFN-Rel local (2026-08-15)",
    "rt-j": "RT-J",
    "rt-plurel": "RT-PluRel",
    "nori-rel": "Nori-Rel",
    "graphsage": "GraphSAGE",
    "relgt": "RelGT",
    "rdblearn": "RDBLearn",
    "relgnn-es": "RelGNN-ES",
    "lightgbm": "LightGBM",
    "constant-per-entity": "Constant/entity",
    "constant-global": "Constant/global",
}
MATRIX_NAMES = {
    **NAMES,
    "kurversc": "KurveRSC",
    "tabpfn-rel-client-2026-09-28": "Client 09-28",
    "tabpfn-rel-local-2026-09-28": "Local 09-28",
    "tabpfn-rel-client-2026-08-15": "Client 08-15",
    "tabpfn-rel-local-2026-08-15": "Local 08-15",
}
METHODS = list(NAMES)
CLIENT = "tabpfn-rel-client-2026-09-28"


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def table(headers, rows):
    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
        ]
        + ["| " + " | ".join(map(str, row)) + " |" for row in rows]
    )


def score(value, metric):
    value = float(value)
    return f"{value:.8f}" if metric == "mae" and abs(value) < 1 else f"{value:.6f}"


def build(repo):
    assets = repo / "docs/assets"
    rows = read_csv(assets / "kurversc-staged-scores-20261027.csv")
    panel = read_csv(assets / "kurversc-staged-panel-20261027.csv")
    evidence = json.loads(
        (assets / "kurversc-staged-evidence-20261027.json").read_text()
    )
    additional = json.loads(
        (assets / "kurversc-staged-additional-version-panel-20261027.json").read_text()
    )
    assert len(rows) == 21 and all(
        r["fresh_completed_at_snapshot"] == "True" for r in rows
    )
    assert len(panel) == 315 and set(r["model"] for r in panel) == set(METHODS)
    lookup = {(r["model"], r["dataset"], r["task"]): r for r in panel}
    assert len(lookup) == 315
    kinds = {r["model"]: r["kind"] for r in panel}
    comparisons, wins, losses = [], 0, 0
    for r in rows:
        metric = r["metric"]
        value, comparator = float(r["staged_kurversc"]), float(r["tabpfn_client"])
        assert math.isfinite(value) and math.isfinite(comparator)
        key = (r["dataset"], r["task"])
        assert math.isclose(
            float(lookup[("kurversc", *key)]["test_score"]), value, abs_tol=1e-12
        )
        assert float(lookup[(CLIENT, *key)]["test_score"]) == comparator
        better = value > comparator if metric == "roc_auc" else value < comparator
        wins += better
        losses += value != comparator and not better
        comparisons.append(
            [
                "/".join(key),
                "AUROC ↑" if metric == "roc_auc" else "MAE ↓",
                score(r["published_kurversc"], metric),
                score(value, metric),
                score(comparator, metric),
                "KurveRSC" if better else "Client",
            ]
        )
    assert (wins, losses) == (11, 10)
    rankings = evidence["aggregate"]["rankings"]
    by_name = {r["method"]: r for r in rankings}
    assert len(rankings) == 15 and by_name["kurversc"]["elo"] == 1948.6
    aggregate = table(
        [
            "Elo rank",
            "Method",
            "Kind",
            "Elo",
            "Bootstrap + / -",
            "Mean rank",
            "Panel win rate",
            "Rescaled loss",
        ],
        [
            [
                r["elo_rank"],
                NAMES[r["method"]],
                kinds[r["method"]],
                f"{r['elo']:.1f}",
                f"+{r['elo+']:.1f} / -{r['elo-']:.1f}",
                f"{r['rank']:.3f}",
                f"{100 * r['winrate']:.2f}%",
                f"{r['loss_rescaled']:.6f}",
            ]
            for r in rankings
        ],
    )
    loss = table(
        ["Loss rank", "Method", "Rescaled loss", "Elo rank", "Elo"],
        [
            [
                n,
                NAMES[r["method"]],
                f"{r['loss_rescaled']:.6f}",
                r["elo_rank"],
                f"{r['elo']:.1f}",
            ]
            for n, r in enumerate(sorted(rankings, key=lambda x: x["loss_rescaled"]), 1)
        ],
    )
    matrices = []
    for metric, label in [
        ("roc_auc", "Classification AUROC higher is better"),
        ("mae", "Regression MAE lower is better"),
    ]:
        matrix = []
        for r in rows:
            if r["metric"] != metric:
                continue
            values = [
                float(lookup[(m, r["dataset"], r["task"])]["test_score"])
                for m in METHODS
            ]
            best = max(values) if metric == "roc_auc" else min(values)
            formatted = [
                f"**{score(v, metric)}**" if v == best else score(v, metric)
                for v in values
            ]
            winners = ", ".join(
                MATRIX_NAMES[m] for m, v in zip(METHODS, values) if v == best
            )
            matrix.append([r["dataset"] + "/" + r["task"], *formatted, winners])
        matrices.append(
            f"<details>\n<summary><strong>{label}</strong></summary>\n\n"
            + table(
                [
                    "Dataset / task",
                    *[MATRIX_NAMES[m] for m in METHODS],
                    "Overall winner",
                ],
                matrix,
            )
            + "\n\n</details>"
        )
    assert additional["methods"] == 16 and additional["tasks"] == 21
    newer = next(
        r for r in additional["rankings"] if r["method"] == "kurversc-2026-10-22"
    )
    other = next(r for r in additional["rankings"] if r["method"] == CLIENT)
    assert newer["elo_rank"] == 1 and newer["elo"] == 1954.2 and other["elo"] == 1940.1
    section = f"""## RelArena performance

The frozen complete staged procedure finished **all 21 official RelBench v1
tasks**, with **no failures or historical score substitutions**. The final
server scorer ran on **October 27, 2026**. The review versions are **KurveRSC
0.2.0 / GraphReduce 1.10.18**; the score-producing wheels used development
labels 0.1.1 / 1.10.17. Their Python payloads match before version stamping;
see the [source-parity receipt](docs/staged-source-parity.json).

These results use the complete `kurversc.run_staged_task` procedure, exposed in
RelArena as `kurversc-latest`. Native `fit` / `predict` alone run a narrower
procedure. See the [reproduction instructions](docs/staged-release.md) and
[technical report](docs/kurversc-staged-technical-report.md).

### Current per-task comparison

The comparator is **`tabpfn-rel-client-2026-09-28`**, captured with the other
RelArena results on October 25. Previous KurveRSC means its rows in that same
capture. It differs from the older September 2 README snapshot.

{table(["Dataset / task", "Metric", "Previous KurveRSC", "KurveRSC 0.2.0", "TabPFN-Rel client", "Winner"], comparisons)}

KurveRSC wins **11 of 21** direct comparisons with that client: **8 of 12
classification tasks** and **3 of 9 regression tasks**. Sixteen scores improve
over the captured previous KurveRSC and five decline. Winners use full precision;
post-votes remains a narrow loss. The [score CSV](docs/assets/kurversc-staged-scores-20261027.csv)
contains full precision and result/prediction hashes.

### Aggregate RelArena leaderboard

The official scorer replaces the captured old KurveRSC rows with these 21 fresh
scores, retaining the other 14 methods and the constant-global Elo anchor at
1000. KurveRSC ranks **first by point Elo: 1948.6 versus 1937.1** for the client.
The bootstrap intervals overlap. This is the completed author-run comparison
on a fixed panel, pending independent maintainer reproduction and public
leaderboard acceptance.

{aggregate}

Panel win rate averages comparisons against all other methods; it is different
from the direct 11–10 client comparison. Earlier and later TabPFN versions are
separate participants in this captured panel.

#### Ranking by rescaled loss

Lower mean per-task min-max rescaled error is better. This metric is distinct
from regression nMAE normalized by the training-target standard deviation.

{loss}

KurveRSC also ranks first by this panel's rescaled loss: **0.073390 versus
0.073884** for the client. Scoring-pool membership changes the ratings:

| Reporting scope | KurveRSC Elo | Client Elo | KurveRSC Elo rank |
|---|---:|---:|---:|
| Replace previous KurveRSC, 15 methods | 1948.6 | 1937.1 | 1 |
| Retain previous KurveRSC and add the dated staged system, 16 methods | 1954.2 | 1940.1 | 1 |

Both calculations use the same 21 fresh scores. The participant matrices below
and the main aggregate tables use the 15-method replacement panel. The
[retained-version scorer output](docs/assets/kurversc-staged-additional-version-panel-20261027.json)
records the separate 16-method calculation.

### Per-task results across all participants

Bold marks the best full-precision point score across all 15 methods, including
both dated client/local TabPFN versions. Matrix date suffixes refer to 2026.
The [complete score panel CSV](docs/assets/kurversc-staged-panel-20261027.csv)
contains all 315 method/task records.

{chr(10).join(matrices)}

### Runtime and resources

The complete tasks took **0.72–14.17 hours** on a CPU-only host with 495.77 GiB
RAM and 64 logical CPUs. The maximum task cgroup RAM peak was **333.51 GiB**
for user-badge. Every task completed below 24 hours. The
[resource report and CSV](docs/kurversc-staged-resources.md) contain every task's
runtime, peak RAM, CPU/storage coverage, SQL working-memory settings and
parallel-admission guidance for larger machines.

### Earlier results

The [September 2 README performance snapshot](docs/archive/relarena-performance-20260902.txt)
and [original technical report](docs/kurversc-technical-report.md) remain historical
references. They describe earlier pipelines and comparison panels. The
[current staged report](docs/kurversc-staged-technical-report.md) documents the
new implementation and final reproduction.

## Native API details

"""
    readme = repo / "README.md"
    text = readme.read_text()
    start = text.index("## RelArena performance")
    end = text.index("Both node arguments accept", start)
    archive = repo / "docs/archive/relarena-performance-20260902.txt"
    if not archive.exists():
        assert "September 2, 2026" in text[start:end]
        archive.parent.mkdir(parents=True, exist_ok=True)
        archive.write_text(
            "Historical README performance snapshot, September 2, 2026\n\n"
            + text[start:end]
        )
    text = text[:start] + section + text[end:]
    text = text.replace(
        'href="https://github.com/kurveai/kurversc/blob/main/docs/kurversc-technical-report.pdf">Technical Report',
        'href="docs/kurversc-staged-technical-report.pdf">Technical Report',
        1,
    )
    text = text.replace(
        "The [KurveRSC technical report]", "The [original KurveRSC technical report]"
    )
    text = text.replace(
        "costs. Its combined 21-task scorecard is labeled separately from the ongoing\nfresh reproduction.",
        "costs. All 21 tasks in the final fresh reproduction are complete.",
    )
    text = text.replace(
        "The high-level API is one function. Give `fit` an entity table, a label table,\ntheir join keys, the target, and an authoritative train/validation split:",
        "The native graph-fitting API is one function. The performance tables below\nuse the [complete staged procedure](docs/staged-release.md), which adds temporal\npolicy and readout selection. For the native API, give `fit` an entity table, a\nlabel table, their join keys, target, and authoritative train/validation split:",
    )
    text = text.replace(
        "> pipeline. Native-reference tables elsewhere below are historical.",
        "> pipeline. Earlier reference tables are preserved in the dated archive.",
    )
    note = "The deferred feature-ownership cleanup is recorded in [TECH_DEBT.txt](TECH_DEBT.txt);\nit does not block this release or change its frozen configuration.\n\n"
    if "[TECH_DEBT.txt]" not in text:
        marker = "The experimental [KurvePlanner training scaffold]"
        text = text.replace(marker, note + marker, 1)
    readme.write_text(text)
    return dict(
        tasks=len(rows),
        panel_rows=len(panel),
        methods=len(rankings),
        wins=wins,
        losses=losses,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo", type=Path, default=Path(__file__).resolve().parents[1]
    )
    print(json.dumps(build(parser.parse_args().repo.resolve())))
