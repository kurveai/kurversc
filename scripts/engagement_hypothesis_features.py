"""Isolated, bounded, metadata-driven challengers. No production integration."""

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from scripts.activity_timeline_challenger import literal, quote


def parent_text_columns(frame, schema, root, maximum=4):
    eligible = []
    for name in schema["categorical_columns"]:
        if not name.startswith(root+"_") or name not in frame:
            continue
        values = frame[name].astype("string").fillna("").replace("__missing__","")
        nonempty = values.loc[values.str.len()>0]
        if len(nonempty) and nonempty.str.len().mean() >= 12 and nonempty.nunique() >= 20:
            eligible.append((-float(nonempty.str.len().mean()),name))
    return [name for _,name in sorted(eligible)[:maximum]]


class ParentSemanticEncoder:
    """Training-only field-aware TF-IDF followed by at most 32 dense values."""

    def __init__(self, dimensions=32, maximum_per_field=4096):
        self.dimensions,self.maximum_per_field = dimensions,maximum_per_field

    @staticmethod
    def documents(frame,name):
        return (frame[name].astype("string").fillna("").replace("__missing__","")
            .str.slice(0,2048).str.replace(r"<[^>]*>"," ",regex=True).tolist())

    def fit(self,frame,columns):
        if not 1 <= len(columns) <= 4:
            raise ValueError("One to four training-selected parent text fields required")
        self.vectorizers = {}
        matrices = []
        for name in columns:
            vectorizer = TfidfVectorizer(max_features=self.maximum_per_field,min_df=2,max_df=.98,
                ngram_range=(1,2),sublinear_tf=True,strip_accents="unicode",dtype=np.float32)
            try:
                matrix = vectorizer.fit_transform(self.documents(frame,name))
            except ValueError as error:
                if "empty vocabulary" not in str(error) and "After pruning" not in str(error):
                    raise
                continue
            self.vectorizers[name] = vectorizer
            matrices.append(matrix)
        if not matrices:
            raise ValueError("No reusable training-only parent vocabulary")
        matrix = sparse.hstack(matrices,format="csr")
        count = min(self.dimensions,matrix.shape[1]-1,matrix.shape[0]-1)
        if count < 1:
            raise ValueError("Insufficient text variation for a bounded encoding")
        self.svd = TruncatedSVD(n_components=count,random_state=0)
        self.svd.fit(matrix)
        self.audit = dict(dimensions=count,training_rows=len(frame),
            vocabulary_sizes={n:len(v.vocabulary_) for n,v in self.vectorizers.items()},
            max_fields=4,max_characters_per_field=2048,fit_scope="Mature training rows only")
        return self

    def transform(self,frame):
        matrix = sparse.hstack([v.transform(self.documents(frame,n)) for n,v in self.vectorizers.items()],format="csr")
        return pd.DataFrame(self.svd.transform(matrix).astype("float32"),
            columns=[f"parent_semantic_{i}" for i in range(self.svd.n_components)])


class EventView:
    def __init__(self,channels,*,temporary,threads=2):
        if not 1 <= len(channels) <= 6:
            raise ValueError("Between one and six frozen channels required")
        self.channels = tuple(channels)
        Path(temporary).mkdir(parents=True,exist_ok=True)
        self.connection = duckdb.connect(config=dict(memory_limit="4GB",threads=threads,
            temp_directory=str(temporary),max_temp_directory_size="40GB"))
        pieces = [f"SELECT CAST({quote(c.entity)} AS VARCHAR) entity,{quote(c.timestamp)}::TIMESTAMP event_time,"
            f"{i} channel FROM read_parquet({literal(c.path)}) WHERE {quote(c.entity)} IS NOT NULL "
            f"AND {quote(c.timestamp)} IS NOT NULL" for i,c in enumerate(channels)]
        self.connection.execute("CREATE VIEW event_rows AS "+" UNION ALL ".join(pieces))

    @staticmethod
    def request_times(requests,phase_boundary):
        times = pd.to_datetime(requests["__event_cutoff"])
        if requests.empty or times.isna().any() or times.max()>pd.Timestamp(phase_boundary) or requests["__event_entity"].isna().any():
            raise ValueError("Nonempty complete requests within the authorized phase required")
        return times

    def close(self):
        self.connection.close()


class ChannelTransitions(EventView):
    """Preserve transitions between channel sets at distinct event timestamps.

    Simultaneous events are a set, not an arbitrary ordering. Their transition
    mass is shared across pairs, preventing tie/order and join-fanout artifacts.
    """

    def transform(self,requests,*,phase_boundary,horizon):
        times = self.request_times(requests,phase_boundary)
        days = pd.Timedelta(horizon).total_seconds()/86400
        if days <= 0:
            raise ValueError("Positive horizon required")
        output = []
        for cutoff in sorted(times.unique()):
            positions = np.flatnonzero(times.eq(cutoff))
            ids = requests.iloc[positions]["__event_entity"].astype(str).to_numpy()
            self.connection.register("roots",pd.DataFrame({"entity":ids}).drop_duplicates())
            stamp = f"TIMESTAMP {literal(pd.Timestamp(cutoff))}"
            fields = []
            for w,window in enumerate((days,4*days)):
                for previous in range(len(self.channels)):
                    for current in range(len(self.channels)):
                        condition = f"previous={previous} AND current={current} AND age<{window!r}"
                        prefix = f"ct_w{w}_p{previous}_c{current}"
                        fields += [f"SUM(mass) FILTER (WHERE {condition}) {prefix}_count",
                            f"SUM(mass*gap) FILTER (WHERE {condition})/NULLIF(SUM(mass) FILTER (WHERE {condition}),0) {prefix}_gap"]
            query = f"""WITH instants AS (
                SELECT entity,event_time,LIST(DISTINCT channel ORDER BY channel) channels
                FROM event_rows JOIN roots USING(entity) WHERE event_time<{stamp}
                GROUP BY entity,event_time
            ), ordered AS (
                SELECT *,LAG(channels) OVER(PARTITION BY entity ORDER BY event_time) previous_channels,
                    LAG(event_time) OVER(PARTITION BY entity ORDER BY event_time) previous_time
                FROM instants
            ), edges AS (
                SELECT entity,p.previous,c.current,
                    1.0/(LEN(channels)*LEN(previous_channels)) mass,
                    EPOCH(event_time-previous_time)/86400.0 gap,
                    EPOCH({stamp}-event_time)/86400.0 age
                FROM ordered CROSS JOIN UNNEST(previous_channels) p(previous)
                    CROSS JOIN UNNEST(channels) c(current)
                WHERE previous_time IS NOT NULL
            ) SELECT entity,{','.join(fields)} FROM edges GROUP BY entity"""
            block = self.connection.execute(query).df().set_index("entity").reindex(ids).reset_index(drop=True)
            count_columns = [c for c in block if c.endswith("_count")]
            block[count_columns] = block[count_columns].fillna(0)
            block["__position"] = positions
            output.append(block)
        return pd.concat(output).sort_values("__position").drop(columns="__position").reset_index(drop=True).astype("float32")


def km_curve(closed,censored):
    """Grouped Kaplan-Meier; events precede censoring within a time bin."""
    closed,censored = np.asarray(closed,float),np.asarray(censored,float)
    if closed.shape != censored.shape or np.any(closed<0) or np.any(censored<0):
        raise ValueError("Aligned nonnegative interval counts required")
    risk = np.cumsum((closed+censored)[::-1])[::-1]
    hazard = np.divide(closed,risk,out=np.zeros_like(closed),where=risk>0)
    return np.cumprod(1-hazard),risk


class CensoredPeerRecurrence(EventView):
    """Bounded peer return curves, including ongoing silent intervals.

    Cohorts use log2 lifetime active-day counts, never labels or task names.
    Unlike closed-gap sketches, still-dormant peers remain in the risk sets.
    All ordinary events used for each curve are strictly before its query.
    This is an approximate renewal diagnostic, not a causal survival estimate.
    """

    def transform(self,requests,*,phase_boundary,horizon):
        times = self.request_times(requests,phase_boundary)
        h = pd.Timedelta(horizon).total_seconds()/86400
        if h <= 0:
            raise ValueError("Positive horizon required")
        edges = np.r_[0,np.geomspace(h/128,h*128,64),np.inf]
        output = []
        for cutoff in sorted(times.unique()):
            positions = np.flatnonzero(times.eq(cutoff))
            ids = requests.iloc[positions]["__event_entity"].astype(str).to_numpy()
            stamp = f"TIMESTAMP {literal(pd.Timestamp(cutoff))}"
            self.connection.execute(f"""CREATE OR REPLACE TEMP TABLE km_instants AS
                SELECT entity,event_time,EPOCH(event_time-LAG(event_time) OVER
                    (PARTITION BY entity ORDER BY event_time))/86400.0 gap
                FROM (SELECT DISTINCT entity,event_time FROM event_rows WHERE event_time<{stamp})""")
            summary = self.connection.execute(f"""SELECT entity,
                COUNT(DISTINCT CAST(event_time AS DATE)) active_days,
                EPOCH({stamp}-MAX(event_time))/86400.0 age
                FROM km_instants GROUP BY entity""").df().set_index("entity")
            summary["cohort"] = np.clip(np.floor(np.log2(summary.active_days)),0,6).astype(int)
            self.connection.register("km_cohorts",summary.reset_index()[["entity","cohort","age"]])
            intervals = self.connection.execute("""SELECT cohort,gap duration,1 closed FROM km_instants
                JOIN km_cohorts USING(entity) WHERE gap IS NOT NULL
                UNION ALL SELECT cohort,age duration,0 closed FROM km_cohorts""").df()
            bucket = np.clip(np.searchsorted(edges,intervals.duration.to_numpy(),side="right")-1,0,len(edges)-2)
            histograms = {}
            for cohort in (-1,*range(7)):
                mask = np.ones(len(intervals),bool) if cohort==-1 else intervals.cohort.eq(cohort).to_numpy()
                deaths = np.bincount(bucket[mask & intervals.closed.eq(1).to_numpy()],minlength=len(edges)-1)
                censors = np.bincount(bucket[mask & intervals.closed.eq(0).to_numpy()],minlength=len(edges)-1)
                survival,risk = km_curve(deaths,censors)
                histograms[cohort] = (survival,risk,deaths,censors)
            requested = summary.reindex(ids).reset_index(drop=True)
            block = pd.DataFrame(index=range(len(ids)))
            block["km_log_active_days"] = requested.cohort
            for name in ("global","peer"):
                columns = {k:[] for k in ("return_next_horizon","tail_survival","at_risk","entities","closed_intervals","censored_intervals")}
                for age,cohort in zip(requested.age,requested.cohort):
                    if pd.isna(age) or pd.isna(cohort):
                        for values in columns.values():
                            values.append(np.nan)
                        continue
                    s,r,d,c = histograms[-1 if name=="global" else int(cohort)]
                    before = np.clip(np.searchsorted(edges,age,side="right")-1,0,len(s)-1)
                    after = np.clip(np.searchsorted(edges,age+h,side="right")-1,0,len(s)-1)
                    # Use survival at each bin's LEFT boundary: no event
                    # later than the conditioning time is assumed observed.
                    sa = 1. if before==0 else s[before-1]
                    sb = 1. if after==0 else s[after-1]
                    columns["return_next_horizon"].append(np.clip(1-sb/sa,0,1) if sa>0 else np.nan)
                    columns["tail_survival"].append(sa)
                    columns["at_risk"].append(r[before])
                    columns["entities"].append(c.sum())
                    columns["closed_intervals"].append(d.sum())
                    columns["censored_intervals"].append(c.sum())
                for key,values in columns.items():
                    block[f"km_{name}_{key}"] = values
            weight = block.km_peer_entities/(block.km_peer_entities+50)
            block["km_shrunk_return"] = (weight*block.km_peer_return_next_horizon+
                (1-weight)*block.km_global_return_next_horizon)
            block["__position"] = positions
            output.append(block)
        return pd.concat(output).sort_values("__position").drop(columns="__position").reset_index(drop=True).astype("float32")
