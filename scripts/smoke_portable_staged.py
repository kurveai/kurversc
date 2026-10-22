"""Build an unseen synthetic schema for a genuine complete-pipeline replay."""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from kurversc.task_context import PredictionTask,SourceTable,TaskContext


def make_context(*, renamed=False, delayed_labels=False, regression=False):
    count = 48
    root,child = ("warehouse/customer accounts","warehouse/purchases") if renamed else ("subscribers","interactions")
    entity,tenant = ("account identifier","region identifier") if renamed else ("external_id","tenant_id")
    clock,target = ("forecast moment","future response") if renamed else ("prediction_time","outcome")
    key = (tenant,entity)
    ids = [f"uuid-{i:04d}" for i in range(count)]
    parents = pd.DataFrame({entity:ids,tenant:["zone"]*count,"attribute":np.arange(count)%4})
    dates = pd.date_range("2019-05-01",periods=8,freq="MS")
    labels = pd.concat([pd.DataFrame({entity:ids,tenant:["zone"]*count,clock:cut,
        target:(np.arange(count)%4+cut.month%3).astype(float) if regression
            else ((np.arange(count)%4+cut.month%3)>=3).astype(int)}) for cut in dates],ignore_index=True)
    validation = labels.iloc[:count].copy()
    validation[clock] = pd.Timestamp("2020-01-01")
    validation[target] = (np.arange(count)%4+1).astype(float) if regression else ((np.arange(count)%4+1)>=3).astype(int)
    query = validation.drop(columns=target).assign(**{clock:pd.Timestamp("2020-02-01")})
    if delayed_labels:
        for frame in (labels,validation):
            # Some older cohort results arrive out of order. Latest results
            # remain mature by their declared inner/outer phase boundaries.
            delay = (np.arange(len(frame))%7).clip(max=2)
            frame["result_observed_at"] = frame[clock]+pd.Timedelta("28d")+pd.to_timedelta(delay,unit="d")
    events = pd.concat([pd.DataFrame({entity:ids,tenant:["zone"]*count,"event_time":cut-pd.Timedelta(days=2),
        "amount":1+np.arange(count)%5}) for cut in [*dates,pd.Timestamp("2020-01-01")]],ignore_index=True)
    return TaskContext.from_tables(tables={root:SourceTable(parents,primary_key=key,timeless=True),
        child:SourceTable(events,foreign_keys={key:root},timestamp="event_time")},
        task=PredictionTask(root,key,clock,target,"28d",name="unseen-problem",
            kind="regression" if regression else "classification",
            label_available_at="result_observed_at" if delayed_labels else None),
        training=labels,validation=validation,requests=query,
        inner_cutoff="2020-01-01",outer_cutoff="2020-02-01",dataset_name="unregistered-schema")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--renamed",action="store_true")
    parser.add_argument("--delayed-labels",action="store_true")
    parser.add_argument("--regression",action="store_true")
    args = parser.parse_args()
    print(make_context(renamed=args.renamed,delayed_labels=args.delayed_labels,regression=args.regression).export(args.output))


if __name__=="__main__":
    main()
