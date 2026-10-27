"""Run the installed complete pipeline on a small, unregistered relational task.

Use manually after installing the staged extra. Pass --harness to also exercise
the current RelArena registered system and actual core split types. Every run
owns a fresh output directory; no benchmark data or hidden labels are loaded.
"""

import argparse
from dataclasses import fields
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from kurversc import run_staged_task
from kurversc.task_context import PredictionTask, SourceTable, TaskContext
from kurversc.task_source import ContextTaskSource


def make_context(*, regression=False):
    count = 48
    entity = np.arange(count)
    dates = pd.date_range("2019-05-01", periods=8, freq="MS")

    def outcomes(cutoff):
        value = entity % 4 + cutoff.month % 3
        return value.astype(float) if regression else (value >= 3).astype(int)

    train = pd.concat(
        [
            pd.DataFrame(
                {"account": entity, "at": cutoff, "response": outcomes(cutoff)}
            )
            for cutoff in dates
        ],
        ignore_index=True,
    )
    val_time = pd.Timestamp("2020-01-01")
    validation = pd.DataFrame(
        {
            "account": entity,
            "at": val_time,
            "response": outcomes(val_time),
        }
    )
    requests = validation.drop(columns="response").assign(at=pd.Timestamp("2020-02-01"))
    events = pd.concat(
        [
            pd.DataFrame(
                {
                    "account": entity,
                    "event_time": cutoff - pd.Timedelta(days=2),
                    "amount": 1 + entity % 5,
                }
            )
            for cutoff in [*dates, val_time]
        ],
        ignore_index=True,
    )
    return TaskContext.from_tables(
        tables={
            "accounts": SourceTable(
                pd.DataFrame({"account": entity, "attribute": entity % 4}),
                primary_key="account",
                timeless=True,
            ),
            "activity": SourceTable(
                events, foreign_keys={"account": "accounts"}, timestamp="event_time"
            ),
        },
        task=PredictionTask(
            "accounts",
            "account",
            "at",
            "response",
            "28d",
            name="unseen-response",
            kind="regression" if regression else "classification",
        ),
        training=train,
        validation=validation,
        requests=requests,
        inner_cutoff="2020-01-01",
        outer_cutoff="2020-02-01",
        dataset_name="unregistered-release-smoke",
    )


def run(output, *, regression=False, harness=False):
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    bundle = make_context(regression=regression).export(output / "input")
    source = ContextTaskSource(bundle)
    if harness:
        from relarena.models.kurversc import KurveRSC20261022System
        from relarena_core import CacheConfig, InnerSplit, OuterSplit, RunIdentity

        def split(kind, values):
            return kind(
                **{field.name: getattr(values, field.name) for field in fields(kind)}
            )

        inner = split(InnerSplit, source.inner_split())
        outer = split(OuterSplit, source.outer_split())
        assert source.task.target_col not in outer.eval_table.df
        system = KurveRSC20261022System(
            cache=CacheConfig(directory=output / "cache", on_miss="compute"),
            run_identity=RunIdentity(
                "unregistered-release-smoke",
                source.identity_digest,
                "unseen-response",
                source.identity_digest,
            ),
        )
        values = system.run(
            source.task, inner_split=inner, outer_split=outer, seed=0, time_limit=1800
        )
        pipeline_output = system.complete_pipeline_output
        predictions = pd.read_parquet(pipeline_output / "predictions.parquet")
        np.testing.assert_array_equal(values, predictions["prediction"].to_numpy())
    else:
        result = run_staged_task(bundle, output=output / "pipeline", task_hours=0.5)
        pipeline_output, predictions = result.output, result.predictions
    assert len(predictions) == 48
    assert np.isfinite(predictions["prediction"]).all()
    receipt = dict(
        status="passed",
        rows=len(predictions),
        harness=harness,
        task_kind="regression" if regression else "classification",
        elapsed_seconds=time.monotonic() - started,
        pipeline_output=str(pipeline_output),
        hidden_targets_supplied=False,
        unregistered_schema=True,
    )
    predictions.to_parquet(output / "predictions.parquet", index=False)
    (output / "smoke.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--regression", action="store_true")
    parser.add_argument("--harness", action="store_true")
    args = parser.parse_args()
    run(args.output.resolve(), regression=args.regression, harness=args.harness)
