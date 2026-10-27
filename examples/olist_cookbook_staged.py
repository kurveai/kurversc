"""Reproduce the cookbook's Olist seller-churn task with the full staged recipe.

Use the frozen candidate wheels on PYTHONPATH. Held-out labels remain outside
the exported model context and are joined to final predictions by entity/time.
No notebook code, previously searched graph or earlier predictions are run.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import numpy as np
import pandas as pd


WINDOW = pd.Timedelta("30d")
VALIDATION = pd.Timestamp("2018-05-15")
TEST = pd.Timestamp("2018-06-15")
RAW = ("olist_sellers_dataset.csv", "olist_customers_dataset.csv",
    "product_category_name_translation.csv", "olist_products_dataset.csv",
    "olist_orders_dataset.csv", "olist_order_items_dataset.csv", "olist_order_reviews_dataset.csv")


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str) + "\n")
    temporary.replace(path)


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def prepare(args):
    from kurversc.task_context import PredictionTask, SourceTable, TaskContext

    if (args.output / "inputs/context.json").exists():
        audit = json.loads((args.output / "data-audit.json").read_text())
        if any(digest(Path(p)) != sha for p, sha in audit["raw_csv_sha256"].items()):
            raise ValueError("The raw data changed after context preparation")
        return
    started = time.monotonic()
    source = {name: pd.read_csv(args.data_dir / name) for name in RAW}
    orders = source["olist_orders_dataset.csv"][["order_id", "customer_id", "order_purchase_timestamp"]].copy()
    orders["order_purchase_timestamp"] = pd.to_datetime(orders["order_purchase_timestamp"])
    items = source["olist_order_items_dataset.csv"].merge(orders[["order_id", "order_purchase_timestamp"]],
        on="order_id", how="left", validate="many_to_one").rename(columns={"order_purchase_timestamp": "purchase_ts"})
    items = items[["order_id", "order_item_id", "product_id", "seller_id", "price", "freight_value", "purchase_ts"]]
    if items.purchase_ts.isna().any():
        raise ValueError("Every order item must have a known purchase time")
    reviews = source["olist_order_reviews_dataset.csv"][["order_id", "review_score", "review_answer_timestamp"]].copy()
    reviews["review_answer_timestamp"] = pd.to_datetime(reviews.review_answer_timestamp)
    products = source["olist_products_dataset.csv"][["product_id", "product_category_name", "product_weight_g",
        "product_length_cm", "product_height_cm", "product_width_cm", "product_photos_qty"]].copy()
    categories = source["product_category_name_translation.csv"]
    unknown = products.product_category_name.notna() & ~products.product_category_name.isin(categories.product_category_name)
    unknown_counts = products.loc[unknown, "product_category_name"].value_counts().to_dict()
    # The cookbook's ingest maps unknown foreign keys to missing values via a
    # left join. Retain those product rows and match that missing-key behavior.
    products.loc[unknown, "product_category_name"] = None
    sellers = source["olist_sellers_dataset.csv"][["seller_id", "seller_city", "seller_state"]].copy()
    customers = source["olist_customers_dataset.csv"][["customer_id", "customer_state", "customer_city"]].copy()
    tables = dict(
        sellers=SourceTable(sellers, primary_key="seller_id", timeless=True),
        customers=SourceTable(customers, primary_key="customer_id", timeless=True),
        product_category=SourceTable(categories, primary_key="product_category_name", timeless=True),
        products=SourceTable(products, primary_key="product_id", foreign_keys={"product_category_name": "product_category"}, timeless=True),
        orders=SourceTable(orders, primary_key="order_id", foreign_keys={"customer_id": "customers"},
            timestamp="order_purchase_timestamp", cutoff_boundary="inclusive"),
        order_items=SourceTable(items, foreign_keys={"order_id": "orders", "product_id": "products", "seller_id": "sellers"},
            timestamp="purchase_ts", cutoff_boundary="inclusive"),
        order_reviews=SourceTable(reviews, foreign_keys={"order_id": "orders"},
            timestamp="review_answer_timestamp", cutoff_boundary="inclusive"))

    def labels(cutoff):
        past = items.loc[(items.purchase_ts > cutoff - WINDOW) & (items.purchase_ts <= cutoff), "seller_id"].drop_duplicates()
        future = set(items.loc[(items.purchase_ts > cutoff) & (items.purchase_ts <= cutoff + WINDOW), "seller_id"])
        frame = pd.DataFrame({"seller_id": past.sort_values().to_numpy(), "timestamp": cutoff})
        frame["churn"] = (~frame.seller_id.isin(future)).astype("int8")
        return frame

    train = pd.concat([labels(cutoff) for cutoff in pd.date_range(VALIDATION - WINDOW, items.purchase_ts.min(), freq=-WINDOW)], ignore_index=True)
    validation, test = labels(VALIDATION), labels(TEST)
    counts = {name: dict(rows=len(frame), anchors=frame.timestamp.nunique(), churn_rate=float(frame.churn.mean()))
        for name, frame in (("train", train), ("validation", validation), ("test", test))}
    if (len(train), len(validation), len(test)) != (11135, 1159, 1088):
        raise ValueError("Olist cohort sizes differ from the inspected public data")
    private = args.output / "evaluation"
    private.mkdir(parents=True, exist_ok=True)
    test.to_parquet(private / "heldout-labels.parquet", index=False)
    context = TaskContext.from_tables(tables=tables,
        task=PredictionTask("sellers", "seller_id", "timestamp", "churn", WINDOW, name="seller-churn"),
        training=train, validation=validation, requests=test.drop(columns="churn"),
        inner_cutoff=VALIDATION, outer_cutoff=TEST, dataset_name="olist-cookbook")
    context.export(args.output / "inputs")
    raw_hashes = {str((args.data_dir / name).resolve()): digest(args.data_dir / name) for name in RAW}
    dump(args.output / "data-audit.json", dict(status="prepared", raw_csv_sha256=raw_hashes,
        table_rows={name: len(spec.frame) for name, spec in tables.items()}, cohort=counts,
        validation_timestamp=str(VALIDATION), test_timestamp=str(TEST), horizon=str(WINDOW),
        event_time_boundary="inclusive <= anchor; label window (anchor, anchor+30 days]",
        unknown_category_foreign_keys_mapped_to_missing=unknown_counts,
        heldout_targets_exported_to_model=False, structural_keys_are_predictors=False,
        notebook_reported_tabpfn_roc_auc_rounded=.784, preparation_seconds=time.monotonic() - started))
    print(json.dumps(dict(status="prepared", cohorts=counts, context=str(args.output / "inputs/context.json"))), flush=True)


def score(args):
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

    truth = pd.read_parquet(args.output / "evaluation/heldout-labels.parquet")
    predictions = pd.read_parquet(args.output / "run/predictions.parquet")
    joined = truth.merge(predictions, on=["seller_id", "timestamp"], how="outer", validate="one_to_one", indicator=True)
    if not joined._merge.eq("both").all() or not np.isfinite(joined.prediction).all():
        raise ValueError("Every prediction must exactly match a held-out seller/anchor")
    procedure = json.loads((args.output / "run/portable-results.json").read_text())
    if procedure["status"] != "completed" or procedure["heldout_targets_used"] or procedure["test_score"] is not None:
        raise ValueError("A completed target-unseen full procedure is required")
    resources = json.loads((args.output / "resources.json").read_text())
    versions = {}
    for name in ("kurversc", "graphreduce", "numpy", "pandas", "catboost", "duckdb", "torch", "transformers"):
        versions[name] = metadata.version(name)
    result = dict(status="completed", task="Olist seller churn in the next 30 days", seed=0,
        test_rows=len(joined), test_roc_auc=float(roc_auc_score(joined.churn, joined.prediction)),
        test_average_precision=float(average_precision_score(joined.churn, joined.prediction)),
        test_brier_score=float(brier_score_loss(joined.churn, joined.prediction)),
        notebook_tabpfn_test_auc_rounded=.784, tabpfn_rerun=False,
        full_staged_profile=procedure["profile"], heldout_targets_used_for_selection=False,
        validation_timestamp=str(VALIDATION), test_timestamp=str(TEST), horizon=str(WINDOW),
        elapsed_hours=resources["elapsed_seconds"] / 3600, peak_process_tree_rss_gib=resources["peak_process_tree_rss_gib"],
        python=sys.version, versions=versions, stages=procedure["stages"],
        prediction_sha256=digest(args.output / "run/predictions.parquet"),
        cookbook_source=json.loads((args.output / "source-identity.json").read_text()))
    joined.drop(columns="_merge").to_csv(args.output / "heldout-predictions.csv", index=False)
    dump(args.output / "results.json", result)
    text = ("OLIST COOKBOOK: FULL STAGED KURVERSC\n\n"
        f"Held-out ROC-AUC: {result['test_roc_auc']:.6f} ({len(joined)} sellers)\n"
        "Notebook's recorded TabPFN-Rel ROC-AUC: 0.784 (rounded; API not rerun)\n"
        f"Runtime: {result['elapsed_hours']:.3f} hours; sampled process-tree RSS peak: {result['peak_process_tree_rss_gib']:.2f} GiB\n"
        "Validation: 2018-05-15; test: 2018-06-15; 30-day forward churn.\n"
        "Same curated seven-table input; test answers remain outside model context.\n"
        "Frozen full staged candidate wheels; resource spill budget only is locally adjusted.\n")
    (args.output / "report.txt").write_text(text)
    print(text, flush=True)


def campaign(args):
    import psutil

    if (args.output / "resources.json").exists():
        raise ValueError("This driver cannot reset/restart an existing task attempt")
    prepare(args)
    started = time.monotonic()
    command = [sys.executable, "-u", str(Path(__file__).resolve()), "--mode", "execute",
        "--data-dir", str(args.data_dir), "--output", str(args.output)]
    state = dict(status="running", started_at=datetime.now(timezone.utc).isoformat(),
        command=command, sample_interval_seconds=1, peak_process_tree_rss_gib=0., memory_reserve_gib=4,
        original_task_budget_hours=23.75, measurement="One-second sampled sum of RSS across task descendants; shared pages may be counted more than once")
    with (args.output / "launcher.log").open("w", buffering=1) as log:
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        state["pid"] = child.pid
        try:
            while child.poll() is None:
                root = psutil.Process(child.pid)
                rss = 0
                for p in [root, *root.children(recursive=True)]:
                    try:
                        rss += p.memory_info().rss
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                state.update(elapsed_seconds=time.monotonic() - started, process_tree_rss_gib=rss / 2**30,
                    host_available_gib=psutil.virtual_memory().available / 2**30)
                state["peak_process_tree_rss_gib"] = max(state["peak_process_tree_rss_gib"], rss / 2**30)
                dump(args.output / "resources.json", state)
                time.sleep(1)
            state.update(status="completed" if child.returncode == 0 else "failed", returncode=child.returncode,
                elapsed_seconds=time.monotonic() - started, finished_at=datetime.now(timezone.utc).isoformat())
            dump(args.output / "resources.json", state)
            if child.returncode:
                raise RuntimeError("Full Olist task did not complete; see launcher.log and run/execution.log")
        except BaseException:
            if child.poll() is None:
                child.send_signal(signal.SIGTERM)
                try:
                    child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
            raise
    score(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("prepare", "campaign", "execute", "score"), default="campaign")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.data_dir, args.output = args.data_dir.resolve(), args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.mode == "execute":
        from kurversc.staged import run_staged_task
        run_staged_task(args.output / "inputs/context.json", output=args.output / "run",
            task_hours=23.75, minimum_available_memory_gib=4, poll_seconds=1)
    else:
        {"prepare": prepare, "campaign": campaign, "score": score}[args.mode](args)
