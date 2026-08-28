from types import SimpleNamespace

import pandas as pd

from kurversc.relbench import (
    _edge_instances,
    _sample_timestamps,
    relbench_problem_from_objects,
)


def test_timestamp_sampling_keeps_latest_cutoff() -> None:
    timestamps = list(pd.date_range("2020-01-01", periods=10, freq="D"))

    selected = _sample_timestamps(timestamps, 3, seed=42)

    assert len(selected) == 3
    assert selected[-1] == timestamps[-1]
    assert selected == sorted(selected)


def test_edge_instances_use_metadata_and_skip_cycles() -> None:
    users = SimpleNamespace(
        df=pd.DataFrame(
            {
                "Id": [1, 2],
                "CreationDate": pd.to_datetime(["2020-01-01", "2020-01-02"]),
            }
        ),
        pkey_col="Id",
        time_col="CreationDate",
        fkey_col_to_pkey_table={},
    )
    posts = SimpleNamespace(
        df=pd.DataFrame(
            {
                "Id": [10, 11, 12],
                "OwnerUserId": [1, 1, 3],
                "ParentId": [None, 10, None],
                "CreationDate": pd.to_datetime(
                    ["2020-02-01", "2020-02-02", "2020-02-03"]
                ),
            }
        ),
        pkey_col="Id",
        time_col="CreationDate",
        fkey_col_to_pkey_table={"OwnerUserId": "users", "ParentId": "posts"},
    )
    comments = SimpleNamespace(
        df=pd.DataFrame(
            {
                "Id": [20, 21],
                "PostId": [10, 12],
                "CreationDate": pd.to_datetime(["2020-03-01", "2020-03-02"]),
            }
        ),
        pkey_col=None,
        time_col="CreationDate",
        fkey_col_to_pkey_table={"PostId": "posts"},
    )
    database = SimpleNamespace(
        table_dict={"users": users, "posts": posts, "comments": comments}
    )

    root, tables, relationships = _edge_instances(
        database,
        root_table="users",
        root_frame=users.df,
        search_root_frame=users.df,
        schema_depth=3,
        sample_rows=100,
        random_state=42,
    )

    assert root.key == "Id"
    assert root.date == "CreationDate"
    assert len(tables) == 2
    assert [table.date for table in tables] == ["CreationDate", "CreationDate"]
    assert [(edge.parent_key, edge.child_key) for edge in relationships] == [
        ("Id", "OwnerUserId"),
        ("Id", "PostId"),
    ]
    assert all("ParentId" not in edge.child for edge in relationships)
    assert len(tables[0].source) == 3
    assert len(tables[0].search_source) == 2
    assert tables[0].context_keys == ("ParentId",)
    assert tables[1].context_keys == ()


def test_problem_samples_latest_training_frame_before_rows() -> None:
    train = pd.DataFrame(
        {
            "uid": range(8),
            "timestamp": pd.to_datetime(["2020-01-01"] * 4 + ["2020-02-01"] * 4),
            "target": [0, 1] * 4,
        }
    )
    validation = pd.DataFrame(
        {
            "uid": range(8, 12),
            "timestamp": pd.to_datetime(["2020-03-01"] * 4),
            "target": [0, 1, 0, 1],
        }
    )
    users = SimpleNamespace(
        df=pd.DataFrame({"uid": range(12)}),
        pkey_col="uid",
        time_col=None,
        fkey_col_to_pkey_table={},
    )
    task = SimpleNamespace(
        entity_table="users",
        entity_col="uid",
        time_col="timestamp",
        target_col="target",
        task_type="binary_classification",
    )

    problem = relbench_problem_from_objects(
        task,
        SimpleNamespace(table_dict={"users": users}),
        SimpleNamespace(df=train),
        SimpleNamespace(df=validation),
        sample_rows=6,
        max_train_timestamps=2,
        schema_depth=1,
    )

    search_labels = problem.label_node.search_source
    search_train = search_labels.loc[
        search_labels["__kurversc_relbench_split__"] == "train"
    ]
    assert len(search_train) == 3
    assert search_train["timestamp"].nunique() == 1
    assert search_train["timestamp"].iloc[0] == pd.Timestamp("2020-02-01")
    assert problem.train_timestamps == (
        pd.Timestamp("2020-01-01"),
        pd.Timestamp("2020-02-01"),
    )
    assert problem.parent_node.timeless is True
