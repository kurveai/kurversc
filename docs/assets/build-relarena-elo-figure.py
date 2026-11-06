"""Render the report's frozen published Elo panel without recomputing ratings."""

from pathlib import Path
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.collections import LineCollection
import numpy as np


ASSETS = Path(__file__).resolve().parent
SOURCE = ASSETS / "kurversc-relarena-reproduction-20261105.json"
OUTPUT = ASSETS / "kurversc-relarena-elo-20261105"
SNAPSHOT = "fcab8fb46b82299321c101b9a921d9a3f7c72263"

LABELS = {
    "tabpfn-rel-client-latest": "TabPFN-Rel API\nSep 28",
    "kurversc-latest": "KurveRSC\n0.2.0",
    "tabpfn-rel-local-latest": "TabPFN-Rel local\nSep 28",
    "rt-j": "RT-J",
    "tabpfn-rel-client-2026-08-15": "TabPFN-Rel API\nAug 15",
    "rt-plurel": "RT-PluRel",
    "kurversc-2026-09-03": "KurveRSC\nSep 03",
    "nori-rel": "Nori-Rel",
    "tabpfn-rel-local-2026-08-15": "TabPFN-Rel local\nAug 15",
    "graphsage": "GraphSAGE",
    "relgt": "RelGT",
    "rdblearn": "RDBLearn",
    "relgnn-es": "RelGNN",
    "lightgbm": "LightGBM",
    "constant-per-entity": "Constant\nper entity",
    "constant-global": "Constant\nglobal",
}
COLORS = {
    "current": "#16818a",
    "tabpfn": "#8273a8",
    "other": "#344b65",
    "constant": "#b7bec7",
}


def main():
    source = json.loads(SOURCE.read_text())
    assert source["source"]["snapshot_commit"] == SNAPSHOT
    assert source["summary"]["tasks"] == 21
    rows = source["leaderboard"]
    assert len(rows) == 16 and {r["method"] for r in rows} == set(LABELS)
    assert [r["rank"] for r in rows] == list(range(1, 17))
    assert all(r["low"] <= r["elo"] <= r["high"] for r in rows)

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 16,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.labelsize": 18,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "svg.fonttype": "path",
        "svg.hashsalt": "kurversc-reproduced-20261105",
        "pdf.fonttype": 42,
        "hatch.linewidth": 0.8,
    })
    fig, ax = plt.subplots(figsize=(14.6, 8.6))
    x = np.arange(len(rows))
    elo = np.array([r["elo"] for r in rows])
    lower = elo - np.array([r["low"] for r in rows])
    upper = np.array([r["high"] for r in rows]) - elo
    system_bars = []
    for i, row in enumerate(rows):
        name = row["method"]
        group = ("current" if name == "kurversc-latest" else
                 "tabpfn" if name.startswith("tabpfn-rel-") else
                 "constant" if name.startswith("constant-") else "other")
        bar = ax.bar(i, row["elo"] - 1000, bottom=1000, width=0.68,
                     color=COLORS[group], edgecolor="#26384a", linewidth=0.8,
                     zorder=3)
        if row["kind"] == "System":
            system_bars.append(bar[0])
        ax.text(i, row["high"] + 23, f'{row["elo"]:.0f}',
                ha="center", va="bottom", rotation=0, fontsize=16,
                fontweight="bold" if name == "kurversc-latest" else "normal",
                color=COLORS["current"] if name == "kurversc-latest" else "#172b42")
    ax.errorbar(x, elo, yerr=np.array([lower, upper]), fmt="none",
                ecolor="#3e464f", elinewidth=1.5, capsize=3, capthick=1.5, zorder=4)
    ax.set_xticks(x, [LABELS[r["method"]] for r in rows], rotation=48,
                  ha="right", rotation_mode="anchor")
    for tick, row in zip(ax.get_xticklabels(), rows):
        if row["method"] == "kurversc-latest":
            tick.set_color(COLORS["current"])
            tick.set_fontweight("bold")
    ax.set_ylabel("Elo (constant-global anchor = 1000)")
    ax.set_ylim(800, 2180)
    ax.set_yticks(np.arange(800, 2200, 200))
    ax.set_xlim(-0.7, len(rows) - 0.3)
    ax.grid(axis="y", linestyle="--", linewidth=0.7, color="#cbd1d8", zorder=0)
    ax.axhline(1000, color="#7c8794", linewidth=0.9, zorder=2)
    legend = [
        Patch(facecolor=COLORS["current"], label="KurveRSC 0.2.0"),
        Patch(facecolor=COLORS["tabpfn"], label="TabPFN-Rel"),
        Patch(facecolor=COLORS["other"], label="Other methods"),
        Patch(facecolor=COLORS["constant"], label="Constant predictors"),
        Patch(facecolor="white", edgecolor="#26384a", hatch="///", label="System"),
    ]
    fig.legend(handles=legend, loc="lower center", bbox_to_anchor=(0.53, 0.015),
               ncol=5, frameon=False, fontsize=15, handlelength=1.4, columnspacing=1.3)
    fig.subplots_adjust(left=0.09, right=0.99, top=0.96, bottom=0.34)
    # Explicit clipped lines avoid repeating-pattern seams in PDF renderers.
    fig.canvas.draw()
    origin = ax.transData.transform((0, 1000))
    step_x = ax.transData.transform((1, 1000))[0] - origin[0]
    step_y = ax.transData.transform((0, 1001))[1] - origin[1]
    slope = step_x / step_y
    spacing = (8 * fig.dpi / 72) / step_y
    for bar in system_bars:
        left, width = bar.get_x(), bar.get_width()
        bottom, height = bar.get_y(), bar.get_height()
        rise = slope * width
        segments = [((left, y), (left + width, y + rise))
                    for y in np.arange(bottom - rise, bottom + height, spacing)]
        lines = LineCollection(segments, colors="#26384a", linewidths=0.7,
                               zorder=3.5)
        ax.add_collection(lines)
        lines.set_clip_path(bar.get_path(), bar.get_transform())
    fig.savefig(OUTPUT.with_suffix(".svg"), metadata={"Date": None})
    fig.savefig(OUTPUT.with_suffix(".pdf"),
                metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)
    print(f"Rendered all {len(rows)} methods from RelArena {SNAPSHOT[:7]}.")


if __name__ == "__main__":
    main()
