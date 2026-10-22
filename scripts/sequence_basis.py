"""Reconstruct existing GraphReduce sequence algebra from an authorized frame.

No labels, historical winner, relation scans, or newly searched family are
needed. Source counts and date extrema are already point-in-time features.
The lifetime denominator is recovered from GraphReduce's observed-duration
rate, never guessed from an arbitrary largest window or foreign-key count.
"""

import argparse
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from scripts.engagement_local_comparison import freeze
from scripts.run_gap_followup_queue import digest, dump
from scripts.select_activity_readout import cache_locations


COUNT = re.compile(r"^(.+)_num_events_([1-9][0-9]*)d$")


class SequenceBasis:
    def fit(self, schema, *, maximum_columns=256):
        columns = list(schema["columns"])
        dates = set(schema["datetime_columns"])
        groups = {}
        for column in columns:
            match = COUNT.fullmatch(column)
            if match:
                prefix, period = match.group(1), int(match.group(2))
                groups.setdefault(prefix, {})[period] = column
        self.groups, self.input_columns, self.columns = [], [], []
        for prefix, counts in groups.items():
            history = f"{prefix}_observed_history_seconds"
            rate = f"{prefix}_events_per_observed_day"
            if history not in columns or rate not in columns:
                continue
            periods = sorted(counts)
            pairs = [name[:-4] for name in columns if name in dates and name.startswith(prefix + "_")
                     and name.endswith("_min") and name[:-4] + "_max" in dates]
            date_pair = pairs[0] if len(pairs) == 1 else None
            output = [name for p in periods for name in
                (f"{prefix}_activity_rate_{p}d", f"{prefix}_activity_share_{p}d")]
            output += [f"{prefix}_activity_burst_{p}v{q}" for p, q in zip(periods, periods[1:])]
            if date_pair:
                output += [f"{prefix}_active_span_seconds", f"{prefix}_activities_per_active_day"]
            output = [c for c in output if c not in columns]
            if not output:
                continue
            needed = [history, rate, *[counts[p] for p in periods]]
            if date_pair:
                needed += [date_pair + "_min", date_pair + "_max"]
            self.input_columns.extend(c for c in needed if c not in self.input_columns)
            self.columns.extend(output)
            self.groups.append(dict(prefix=prefix, periods=periods, counts=counts,
                history=history, rate=rate, date_pair=date_pair, outputs=output))
        if len(self.columns) > maximum_columns:
            raise ValueError("Complete sequence block exceeds its declared feature budget")
        self.audit = dict(strategy="existing_sequence_algebra_from_authorized_frame",
            columns=len(self.columns), groups=len(self.groups), labels_used=False,
            alternative_graph_used=False, additional_source_rows_read=0,
            maximum_columns=maximum_columns, lifetime_rule="events_per_observed_day * (observed_history_seconds / 86400 + 1)")
        return self

    def transform(self, frame):
        result = {}
        for group in self.groups:
            prefix, periods = group["prefix"], group["periods"]
            history = pd.to_numeric(frame[group["history"]], errors="raise")
            rate = pd.to_numeric(frame[group["rate"]], errors="raise")
            lifetime = rate * (history / 86400 + 1)
            visible = lifetime.notna()
            if (not np.isfinite(lifetime[visible]).all() or (history.dropna() < 0).any()
                    or (lifetime[visible].round() < 1).any()
                    or not np.allclose(lifetime[visible], lifetime[visible].round(), atol=1e-7, rtol=1e-10)):
                raise ValueError("Source duration/rate does not reconstruct a valid integer event count")
            lifetime = lifetime.round()
            counts = {p: pd.to_numeric(frame[group["counts"][p]], errors="raise") for p in periods}
            for p, count in counts.items():
                if ((count.dropna() < 0).any() or not np.isfinite(count.dropna()).all()
                        or not np.equal(count.dropna(), count.dropna().round()).all()
                        or (count > lifetime).any()):
                    raise ValueError("Invalid source window counts")
                result[f"{prefix}_activity_rate_{p}d"] = count / p
                result[f"{prefix}_activity_share_{p}d"] = count / lifetime
            for p, q in zip(periods, periods[1:]):
                if (counts[p] > counts[q]).any():
                    raise ValueError("Source cumulative window counts are not monotone")
                result[f"{prefix}_activity_burst_{p}v{q}"] = counts[p] / counts[q].replace(0, np.nan)
            if group["date_pair"]:
                first, last = [pd.to_datetime(frame[group["date_pair"] + suffix]) for suffix in ("_min", "_max")]
                # DuckDB date_diff('second') counts crossed second boundaries.
                span = (last.dt.floor("s") - first.dt.floor("s")).dt.total_seconds()
                if (span.dropna() < 0).any():
                    raise ValueError("Source event date extrema are reversed")
                result[f"{prefix}_active_span_seconds"] = span
                result[f"{prefix}_activities_per_active_day"] = lifetime / (span / 86400 + 1)
        return pd.DataFrame(result, index=frame.index).reindex(columns=self.columns).reset_index(drop=True)


def build_view(args):
    raw_path = args.task_root / "raw-audit/manifest.json"
    raw = json.loads(raw_path.read_text())
    meta = raw["metadata"]
    if not meta["atomic_date_replay"] or meta["fold"] != "current":
        raise ValueError("Authorized phase-frozen source frames required")
    basis = SequenceBasis().fit(meta["feature_schema"])
    if not basis.columns:
        raise ValueError("No missing sequence algebra to reconstruct")
    cache, snapshots = cache_locations(args.task_root)
    folds = list(range(len(raw["folds"])))
    files = [raw_path, args.task_root / "selection/results.json", args.task_root / "selection/manifest.json",
        cache / "history.parquet", *[snapshots / f"fold{f}/{name}" for f in folds
                                    for name in ("validation.parquet", "manifest.json")]]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    identity = dict(task=meta["task"], dataset=meta["dataset"], basis=basis.audit,
        sources={__file__: digest(__file__)}, inputs={str(p): digest(p) for p in files},
        training_labels_not_used_for_transform=True, official_test_accessed=False, default_changed=False)
    freeze(output / "producer.json", identity)
    schema = dict(columns=basis.columns, categorical_columns=[], datetime_columns=[])
    alternate = {**raw, "metadata": {**meta, "feature_schema": schema}}
    dump(output / "raw-audit/manifest.json", alternate)
    dump(output / "selection/results.json", json.loads((args.task_root / "selection/results.json").read_text()))
    dump(output / "selection/manifest.json", dict(cache=str(output / "cache"),
        phase_frozen_cache=str(output / "snapshots"), producer=identity))
    keys = ["__event_entity", "__event_cutoff", meta["target"]]
    for source, destination in [(cache / "history.parquet", output / "cache/history.parquet"),
        *[(snapshots / f"fold{f}/validation.parquet", output / f"snapshots/fold{f}/validation.parquet") for f in folds]]:
        frame = pd.read_parquet(source, columns=[*basis.input_columns, *keys])
        derived = basis.transform(frame)
        value = pd.concat([frame[keys].reset_index(drop=True), derived], axis=1)
        destination.parent.mkdir(parents=True, exist_ok=True)
        value.to_parquet(destination, index=False)
    for fold in folds:
        phase = json.loads((snapshots / f"fold{fold}/manifest.json").read_text())
        phase["feature_cache_sha256"] = digest(output / f"snapshots/fold{fold}/validation.parquet")
        dump(output / f"snapshots/fold{fold}/manifest.json", phase)
    if identity["inputs"] != {p: digest(p) for p in identity["inputs"]}:
        raise ValueError("Source frames changed during sequence derivation")
    print(json.dumps(dict(output=str(output), **basis.audit)), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("task-root", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    build_view(parser.parse_args())
