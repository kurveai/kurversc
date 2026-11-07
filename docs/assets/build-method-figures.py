"""Build compact, editable method figures and vector publication PDFs.

Run from any directory. Optional PNG previews stay outside the source assets:
    python assets/build-method-figures.py --preview-dir /path/to/previews
"""
from argparse import ArgumentParser
from html import escape
from pathlib import Path
import json
import shutil
import subprocess
import xml.etree.ElementTree as ET

ASSETS = Path(__file__).resolve().parent
INK = "#172b42"
MUTED = "#53667d"
TEAL = "#087e8b"
BLUE = "#386c9c"
BORDER = "#b9c7d6"
PALE = "#edf4fc"
SOFT = "#eff8f6"
WHITE = "#ffffff"


class Figure:
    def __init__(self, width, height, title, description):
        self.width, self.height = width, height
        self.font_sizes = []
        self.parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
            f'<title id="title">{escape(title)}</title><desc id="desc">{escape(description)}</desc>',
            '<defs>',
            *[f'<marker id="arrow-{name}" markerWidth="9" markerHeight="9" refX="8" refY="4.5" orient="auto" markerUnits="userSpaceOnUse"><path d="M0 0 L9 4.5 L0 9 Z" fill="{color}"/></marker>'
              for name, color in (("flow", MUTED), ("teal", TEAL), ("blue", BLUE))],
            '</defs>',
            f'<rect width="{width}" height="{height}" fill="{WHITE}"/>',
        ]

    def rect(self, x, y, width, height, *, fill=WHITE, stroke=BORDER, radius=9, weight=1.8):
        self.parts.append(f'<rect x="{x}" y="{y}" width="{width}" height="{height}" rx="{radius}" fill="{fill}" stroke="{stroke}" stroke-width="{weight}"/>')

    def text(self, x, y, value, *, size=26, bold=False, color=INK, anchor="start"):
        self.font_sizes.append(size)
        self.parts.append(f'<text x="{x}" y="{y}" fill="{color}" font-family="Arial, Helvetica, sans-serif" font-size="{size}" font-weight="{600 if bold else 400}" text-anchor="{anchor}">{escape(str(value))}</text>')

    def lines(self, x, y, values, *, leading=33, **kwargs):
        for n, value in enumerate(values):
            self.text(x, y+n*leading, value, **kwargs)

    def line(self, x1, y1, x2, y2, *, color=BORDER, weight=1.5):
        self.parts.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{weight}"/>')

    def path(self, path, *, kind="flow", arrow=True, weight=2.8):
        color = {"flow": MUTED, "teal": TEAL, "blue": BLUE}[kind]
        end = f' marker-end="url(#arrow-{kind})"' if arrow else ""
        self.parts.append(f'<path d="{path}" fill="none" stroke="{color}" stroke-width="{weight}"{end}/>')

    def arrow(self, x1, y1, x2, y2, *, kind="teal"):
        self.path(f'M{x1},{y1} L{x2},{y2}', kind=kind)

    def node(self, x, y, width, height, title, body, *, fill=PALE):
        self.rect(x, y, width, height, fill=fill)
        self.text(x+20, y+34, title, size=28, bold=True)
        self.lines(x+20, y+70, body, size=26, color=MUTED, leading=30)

    def number(self, x, y, value):
        self.parts.append(f'<circle cx="{x}" cy="{y}" r="20" fill="{TEAL}"/>')
        self.text(x, y+9, value, size=26, bold=True, color=WHITE, anchor="middle")

    def table(self, x, y, widths, headers, rows, fills):
        width, row_height = sum(widths), 44
        self.line(x, y, x+width, y, color=INK, weight=2)
        cursor = x
        for cell_width, value in zip(widths, headers):
            self.text(cursor+12, y+31, value, bold=True)
            cursor += cell_width
        self.line(x, y+row_height, x+width, y+row_height)
        for row_number, row in enumerate(rows):
            top = y+row_height*(row_number+1)
            self.rect(x, top, width, row_height, fill=fills[row_number], stroke="none", radius=0)
            cursor = x
            for cell_width, value in zip(widths, row):
                self.text(cursor+12, top+31, value)
                cursor += cell_width
            self.line(x, top+row_height, x+width, top+row_height)
        self.line(x, y+row_height*(len(rows)+1), x+width, y+row_height*(len(rows)+1), color=INK, weight=2)

    def save(self, name, *, renderer, preview_dir=None):
        source = ASSETS / f"{name}.svg"
        source.write_text("\n".join(self.parts+["</svg>"])+"\n")
        ET.parse(source)
        subprocess.run([renderer, "--format", "pdf", "--output", str(source.with_suffix(".pdf")), str(source)], check=True)
        if preview_dir:
            preview_dir.mkdir(parents=True, exist_ok=True)
            subprocess.run([renderer, "--format", "png", "--output", str(preview_dir/f"{name}.png"), str(source)], check=True)
        # Report text width: letter paper less two 0.78-inch margins.
        print_width_pt = (8.5-2*0.78)*72
        return dict(name=name, width=self.width, height=self.height,
                    print_height_inches=(8.5-2*0.78)*self.height/self.width,
                    minimum_label_points=min(self.font_sizes)*print_width_pt/self.width)


def architecture():
    f = Figure(1440, 720, "KurveRSC 0.2.0 architecture", "Strict temporal inputs feed bounded, schema-derived relational search. GraphReduce creates a typed entity-level frame for a pluggable learner. Validation promotes candidates from 10K screening to 50K confirmation and full-source rerank. A production-faithful finalist check and temporal policy selection precede one generic optional menu and a shared gate. The selected procedure is frozen and refitted before keyed prediction.")
    f.node(20, 12, 660, 90, "Relational database", ["Tables, keys and source clocks"])
    f.node(760, 12, 660, 90, "Task contract", ["Root, horizon, labels and masked requests"])
    f.arrow(350, 106, 350, 115, kind="flow")
    f.arrow(1090, 106, 1090, 115, kind="flow")
    f.rect(20, 119, 1400, 58, fill=PALE, stroke="#779ac4")
    f.text(720, 156, "Source time < query cutoff; phase and label-availability checks", size=27, bold=True, anchor="middle")
    f.arrow(720, 181, 720, 203, kind="flow")

    f.rect(20, 207, 1400, 250, fill="#f6f9fd", stroke="#779ac4")
    f.text(44, 241, "1. Staged relational search (S1)", size=30, bold=True)
    f.node(44, 260, 404, 116, "Schema-derived programs", ["Depth 1-4; bounded repeats", "Families, CDF, annotations"])
    f.node(518, 260, 404, 116, "GraphReduce", ["Filter, annotate, reduce, join", "Entity-level feature frame"])
    f.node(992, 260, 404, 116, "Pluggable learner", ["CatBoost: evaluated reference", "TFMs via compatible adapters"], fill=SOFT)
    f.arrow(452, 315, 514, 315, kind="flow")
    f.arrow(926, 315, 988, 315, kind="flow")
    f.path("M1194,380 V402 H246 V379", kind="blue")
    f.rect(526, 388, 388, 30, fill="#f6f9fd", stroke="none", radius=0)
    f.text(720, 410, "Validation-driven promotion", color=BLUE, anchor="middle")
    f.text(720, 444, "10K screening → 50K confirmation → full-source rerank", size=27, bold=True, anchor="middle")

    f.arrow(225, 461, 225, 479, kind="flow")
    f.node(20, 484, 410, 116, "2. Finalist check (S2)", ["Production-faithful comparison", "Classification and regression"])
    f.node(495, 484, 400, 116, "3. Training policy (S3)", ["Recent / pooled / blends", "Matched origins and seeds"])
    f.node(960, 484, 460, 116, "4. Generic optional menu", ["Text, ensembles and readouts", "One incumbent; shared gate"], fill=SOFT)
    f.arrow(434, 539, 490, 539, kind="flow")
    f.arrow(899, 539, 955, 539, kind="flow")
    f.path("M1190,604 V620 H520 V639", kind="flow")
    f.rect(20, 643, 1000, 64, fill=INK, stroke="none")
    f.text(44, 684, "Freeze and refit", size=29, bold=True, color=WHITE)
    f.text(310, 684, "Plan + schema + policy + learner", size=27, color=WHITE)
    f.arrow(1024, 675, 1085, 675, kind="flow")
    f.rect(1090, 643, 330, 64, fill=SOFT)
    f.text(1110, 670, "Keyed predictions", size=27, bold=True)
    f.text(1110, 698, "Entity / cutoff order", size=26, color=MUTED)
    return f


def reduction():
    f = Figure(1400, 650, "Keyed reduction and propagation", "Arrows carry features from order lines to orders and then to customers. Each reducing edge groups rows by its parent key, then left joins the summaries at the receiving grain. A store dimension enriches orders by a unique-key lookup without aggregation. Root extraction preserves one row per requested customer.")
    f.node(20, 18, 400, 125, "Customers (root)", ["1 row/customer", "customer_id, country, ..."], fill=SOFT)
    f.node(20, 263, 400, 125, "Orders + child features", ["1 row/order", "total, line summaries, region"])
    f.node(20, 508, 400, 120, "Order lines (deepest)", ["Many rows/order", "order_id, price, ..."])
    f.node(950, 263, 430, 125, "Stores (keyed dimension)", ["1 row/store", "store_id, region, ..."])

    f.arrow(220, 503, 220, 393)
    f.number(268, 419, "1")
    f.text(310, 428, "Reduce by order_id", size=28, bold=True)
    f.lines(310, 463, ["max(price), mean(price), ...", "LEFT JOIN into orders"], color=MUTED)
    f.arrow(220, 258, 220, 148)
    f.number(268, 175, "2")
    f.text(310, 184, "Reduce by customer_id", size=28, bold=True)
    f.lines(310, 219, ["Aggregate enriched orders", "LEFT JOIN into customers"], color=MUTED)
    f.arrow(945, 325, 425, 325, kind="blue")
    f.text(685, 295, "LEFT JOIN on store_id", bold=True, color=BLUE, anchor="middle")
    f.text(685, 363, "Lookup; no aggregation", color=MUTED, anchor="middle")

    f.arrow(425, 81, 945, 81)
    f.text(685, 54, "Root extraction", size=28, bold=True, anchor="middle")
    f.rect(950, 18, 430, 125, fill=SOFT)
    f.text(970, 56, "Root feature frame", size=28, bold=True)
    f.text(970, 93, "1 row/requested root entity", color=MUTED)
    for k in range(7):
        f.line(970+60*k, 108, 970+60*k, 128)
    for k in range(3):
        f.line(970, 108+k*10, 1330, 108+k*10)
    return f


def worked_example():
    f = Figure(1400, 610, "Worked keyed-reduction example", "Customer A has orders O1 and O2, with totals 20 and 100. Lines priced 10, 10 and 100 reduce to per-order maxima and means 10 and 100. Those summaries join to orders before reduction to customer grain. The selected output is customer A, order count 2, total sum 120, total mean 60, line maximum 100 and mean of order-level means 55.")
    f.text(20, 36, "Visible order lines", size=30, bold=True)
    f.table(20, 65, [80,100,140], ["Line", "Order", "Price"], [("L1","O1","10"),("L2","O1","10"),("L3","O2","100")], [SOFT,SOFT,PALE])
    f.text(500, 36, "Line summaries", size=30, bold=True)
    f.table(500, 65, [110,110,120], ["Order", "Max", "Mean"], [("O1","10","10"),("O2","100","100")], [SOFT,PALE])
    f.text(970, 36, "Orders + line summaries", size=30, bold=True)
    f.table(970, 65, [90,90,115,115], ["Order", "Total", "Max", "Mean"], [("O1","20","10","10"),("O2","100","100","100")], [SOFT,PALE])
    f.arrow(354, 174, 486, 174)
    f.lines(420, 105, ["reduce", "order_id"], anchor="middle", color=TEAL)
    f.arrow(854, 174, 956, 174)
    f.lines(905, 105, ["left join", "order_id"], anchor="middle", color=TEAL)
    f.text(20, 283, "Many rows/order", color=MUTED)
    f.text(500, 239, "1 summary/order", color=MUTED)
    f.text(970, 239, "1 enriched row/order", color=MUTED)
    f.text(970, 282, "Original totals retained", color=MUTED)
    f.arrow(1175, 297, 1175, 418)
    f.lines(1125, 338, ["Reduce by customer_id", "then left join customers"], leading=36, anchor="end", color=TEAL)
    f.text(20, 396, "Customer features", size=30, bold=True)
    f.rect(20, 429, 1360, 143, fill=SOFT)
    widths = [150,190,235,235,235,315]
    headers = [["Customer"],["Order", "count"],["Order total", "sum"],["Order total", "mean"],["Line-price", "max"],["Line-price", "mean of means"]]
    values = ["A","2","120","60","100","55"]
    cursor = 20
    for width, header, value in zip(widths, headers, values):
        f.lines(cursor+width/2, 461, header, leading=32, anchor="middle", color=MUTED)
        f.text(cursor+width/2, 550, value, size=36, bold=True, color=TEAL, anchor="middle")
        cursor += width
    return f


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--preview-dir", type=Path)
    args = parser.parse_args()
    renderer = shutil.which("rsvg-convert")
    if not renderer:
        raise SystemExit("Install librsvg (rsvg-convert) to build the vector PDFs")
    figures = [("kurversc-staged-architecture", architecture()),
               ("kurversc-keyed-reduction", reduction()),
               ("kurversc-reduction-example", worked_example())]
    receipts = [figure.save(name, renderer=renderer, preview_dir=args.preview_dir) for name,figure in figures]
    if args.preview_dir:
        (args.preview_dir/"figure-dimensions.json").write_text(json.dumps(receipts, indent=2)+"\n")
    print(json.dumps(receipts, indent=2))


if __name__ == "__main__":
    main()
