"""Quoted source names must survive annotation, reduction and plan replay."""

from datetime import datetime

import duckdb
import numpy as np
import pandas as pd
import pytest

from graphreduce.enum import ComputeLayerEnum, PeriodUnit
from graphreduce.node import DatabricksNode, DuckdbNode


@pytest.mark.parametrize("cdf", [False, True])
def test_annotation_and_base_state_reduce_keep_quoted_source_names(cdf):
    raw = pd.DataFrame(
        {
            "id": range(24),
            "user_id": np.repeat([1, 2], 12),
            "date": pd.date_range("2024-01-01", periods=24),
            "Unnamed: 0": np.arange(24) + 0.125,
            "status: raw": ["yes", "no"] * 12,
            'notes "raw"': ["Visit https://example.com with 42 details."] * 24,
        }
    )
    with duckdb.connect(config={"threads": "1"}) as con:
        con.register("source_rows", raw)
        node = DuckdbNode(
            fpath="source_rows",
            prefix="evt",
            pk="id",
            date_key="date",
            client=con,
            compute_layer=ComputeLayerEnum.duckdb,
            columns=list(raw),
            auto_annotate_features=True,
            auto_text_features=True,
            cut_date=datetime(2024, 2, 1),
            compute_period_val=365,
            compute_period_unit=PeriodUnit.day,
            feature_families=("base", "state"),
            base_distribution_features=cdf,
        )
        con.sql(f"CREATE TEMP TABLE prefixed AS {node.build_query(node.do_data())}")
        node._cur_data_ref = "prefixed"
        sample = con.sql("SELECT * FROM prefixed").df()
        ops = node.sql_auto_annotate(sample)
        annotation = node.build_query(ops)
        assert 'THEN "evt_Unnamed: 0" END' in annotation
        assert '"evt_status: raw"' in annotation
        assert '"evt_notes ""raw"""' in annotation
        con.sql(f"CREATE TEMP TABLE annotated AS {annotation}")
        node._cur_data_ref = "annotated"
        frame = con.sql("SELECT * FROM annotated").df()
        pd.testing.assert_frame_equal(frame[sample.columns], sample)
        reduction = node.sql_auto_features(
            frame,
            "user_id",
            type_func_map={
                "numerical": ["min", "max", "mean"],
                "categorical": ["count"],
            },
        )
        query = node.build_query(reduction)
        result = con.sql(query).df().sort_values("evt_user_id").reset_index(drop=True)
        assert len(result) == 2
        np.testing.assert_allclose(result["evt_Unnamed: 0_min"], [0.125, 12.125])
        np.testing.assert_allclose(
            result["evt_unnamed_0_state_latest"], [11.125, 23.125]
        )
        assert bool([c for c in result if "__cdf_" in c]) == cdf
        pd.testing.assert_frame_equal(
            result,
            con.sql(query).df().sort_values("evt_user_id").reset_index(drop=True),
        )


def test_annotation_uses_dialect_quoting_and_preserves_value_literals():
    sample = pd.DataFrame(
        {
            "evt_Unnamed: 0": np.arange(24) + 0.125,
            "evt_status`field": ["O'Reilly: status", "ordinary"] * 12,
        }
    )
    node = DatabricksNode(
        fpath="events", prefix="evt", compute_layer=ComputeLayerEnum.databricks
    )
    operations = node.sql_auto_annotate(sample)
    sql = node.build_query(operations)
    assert "`evt_status``field` = 'O''Reilly: status'" in sql
    assert "THEN `evt_Unnamed: 0` END" in sql
