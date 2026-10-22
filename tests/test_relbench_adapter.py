from types import SimpleNamespace

import pandas as pd
import pytest
import numpy as np

from kurversc.relbench import (
    _edge_instances,
    _enrichment_columns,
    _recently_spaced_timestamps,
    _sample_timestamps,
    relbench_problem_from_objects,
)
from kurversc.core import (
    _materialize_at_cutoff,
    _normalize_tables,
    _prepare_target_history,
    _table_workspace,
    _topology_for_config,
)
from kurversc.search import GraphConfig
from kurversc.specs import (
    TARGET_HISTORY_TABLE,
    TARGET_HISTORY_TIME,
    TARGET_HISTORY_VALUE,
)


def test_enrichment_columns_accept_nested_array_cells() -> None:
    table = SimpleNamespace(
        df=pd.DataFrame(
            {
                "id": [1, 2, 3],
                "embedding": [np.array([1, 2]), np.array([1, 2]), np.array([3, 4])],
                "category": ["a", "b", "a"],
            }
        ),
        pkey_col="id",
        time_col=None,
        fkey_col_to_pkey_table={},
    )

    assert _enrichment_columns(table, 2) == ("embedding", "category")


def test_enrichment_can_retain_bounded_dictionary_descriptors() -> None:
    count = 1_101
    table = SimpleNamespace(
        df=pd.DataFrame(
            {
                "id": range(count),
                "mesh_term": [f"Condition {index}" for index in range(count)],
                "narrative": [f"{index} " + "x" * 90 for index in range(count)],
            }
        ),
        pkey_col="id",
        time_col=None,
        fkey_col_to_pkey_table={},
    )

    assert _enrichment_columns(table, 2) == ()
    assert _enrichment_columns(
        table, 1, high_cardinality_text="fallback"
    ) == ("mesh_term",)

    table.df["compact_class"] = ["A", "B"] * (count // 2) + ["A"]
    assert _enrichment_columns(
        table, 2, high_cardinality_text="fallback"
    ) == ("compact_class",)
    assert _enrichment_columns(
        table, 2, high_cardinality_text="short"
    ) == ("compact_class", "mesh_term")


def test_timestamp_sampling_keeps_latest_cutoff() -> None:
    timestamps = list(pd.date_range("2020-01-01", periods=10, freq="D"))

    selected = _sample_timestamps(timestamps, 3, seed=42)

    assert len(selected) == 3
    assert selected[-1] == timestamps[-1]
    assert selected == sorted(selected)


def test_recent_training_policy_keeps_the_newest_cohorts_and_unsampled_history() -> None:
    dates = pd.date_range("2000-01-01", periods=8, freq="YS")
    train = pd.DataFrame({"uid": [0, 1] * 8,
        "timestamp": dates.repeat(2), "target": [0, 1] * 8})
    validation = pd.DataFrame({"uid": [0, 1],
        "timestamp": pd.to_datetime(["2008-01-01"] * 2), "target": [1, 0]})
    users = SimpleNamespace(df=pd.DataFrame({"uid": [0, 1]}), pkey_col="uid",
        time_col=None, fkey_col_to_pkey_table={})
    task = SimpleNamespace(entity_table="users", entity_col="uid", time_col="timestamp",
        target_col="target", task_type="binary_classification")
    arguments = dict(sample_rows=100, max_train_timestamps=3, search_training_frames=3,
        schema_depth=1, include_target_history=False)
    problem = relbench_problem_from_objects(task, SimpleNamespace(table_dict={"users": users}),
        SimpleNamespace(df=train), SimpleNamespace(df=validation),
        training_timestamp_policy="recent", **arguments)
    assert problem.train_timestamps == tuple(dates[-3:])
    assert len(problem.label_node.history_source) == len(train) + len(validation)
    search = problem.label_node.search_source
    assert set(search.loc[search["__kurversc_relbench_split__"].eq("train"), "timestamp"]) == set(dates[-3:])
    for invalid in ("unknown",):
        with pytest.raises(ValueError, match="training_timestamp_policy"):
            relbench_problem_from_objects(task, SimpleNamespace(table_dict={"users": users}),
                SimpleNamespace(df=train), SimpleNamespace(df=validation),
                training_timestamp_policy=invalid, **arguments)


def test_recent_timestamp_sampling_is_evenly_spaced_near_latest_cutoff() -> None:
    timestamps = list(pd.date_range("2017-06-08", periods=5, freq="91D"))

    selected = _recently_spaced_timestamps(timestamps, 3, window_days=365)

    assert selected == [timestamps[0], timestamps[2], timestamps[4]]


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


def test_edge_instances_ignore_metadata_insertion_order() -> None:
    users = SimpleNamespace(
        df=pd.DataFrame({"user_id": list(range(50))}),
        pkey_col="user_id",
        time_col=None,
        fkey_col_to_pkey_table={},
    )
    alpha = SimpleNamespace(
        df=pd.DataFrame({"user_id": list(range(50)) * 2, "value": range(100)}),
        pkey_col=None,
        time_col=None,
        fkey_col_to_pkey_table={"user_id": "users"},
    )
    beta = SimpleNamespace(
        df=pd.DataFrame({"user_id": list(range(50)) * 2, "value": range(100)}),
        pkey_col=None,
        time_col=None,
        fkey_col_to_pkey_table={"user_id": "users"},
    )

    def expand(table_dict):
        _root, tables, relationships = _edge_instances(
            SimpleNamespace(table_dict=table_dict),
            root_table="users",
            root_frame=users.df,
            search_root_frame=users.df,
            schema_depth=1,
            sample_rows=20,
            random_state=0,
        )
        return (
            [(table.name, table.prefix, table.search_source.copy()) for table in tables],
            relationships,
        )

    left_tables, left_relationships = expand(
        {"users": users, "beta": beta, "alpha": alpha}
    )
    right_tables, right_relationships = expand(
        {"alpha": alpha, "users": users, "beta": beta}
    )

    assert left_relationships == right_relationships
    assert [(name, prefix) for name, prefix, _ in left_tables] == [
        (name, prefix) for name, prefix, _ in right_tables
    ]
    for (_name, _prefix, left), (_other_name, _other_prefix, right) in zip(
        left_tables, right_tables
    ):
        pd.testing.assert_frame_equal(left, right)


def test_edge_instances_traverse_association_to_dimension() -> None:
    customer = SimpleNamespace(
        df=pd.DataFrame({"customer_id": [1, 2]}),
        pkey_col="customer_id",
        time_col=None,
        fkey_col_to_pkey_table={},
    )
    transactions = SimpleNamespace(
        df=pd.DataFrame({"customer_id": [1, 1, 2], "article_id": [10, 11, 10]}),
        pkey_col=None,
        time_col="t_dat",
        fkey_col_to_pkey_table={
            "customer_id": "customer",
            "article_id": "article",
        },
    )
    article_columns = {
        "article_id": [10, 11],
        "department_name": ["shoes", "shirts"],
        **{f"attribute_{index}": [index, index + 1] for index in range(10)},
        "detail_desc": ["x" * 100, "y" * 100],
    }
    article = SimpleNamespace(
        df=pd.DataFrame(article_columns),
        pkey_col="article_id",
        time_col=None,
        fkey_col_to_pkey_table={},
    )
    database = SimpleNamespace(
        table_dict={
            "customer": customer,
            "transactions": transactions,
            "article": article,
        }
    )

    _root, tables, relationships = _edge_instances(
        database,
        root_table="customer",
        root_frame=customer.df,
        search_root_frame=customer.df,
        schema_depth=2,
        sample_rows=100,
        random_state=42,
    )

    assert len(tables) == 2
    assert relationships[0].parent_key == "customer_id"
    assert relationships[0].child_key == "customer_id"
    assert relationships[1].parent.endswith("transactions__customer_id")
    assert relationships[1].parent_key == "article_id"
    assert relationships[1].child_key == "article_id"
    assert relationships[0].reduce is True
    assert relationships[1].reduce is False
    assert set(tables[1].search_source["article_id"]) == {10, 11}
    assert len(tables[1].columns) == 8
    assert "department_name" in tables[1].columns
    assert "detail_desc" not in tables[1].columns


def test_edge_instances_can_unroll_a_bounded_self_relational_path() -> None:
    users = SimpleNamespace(
        df=pd.DataFrame({"user_id": [1, 2, 3]}),
        pkey_col="user_id",
        time_col=None,
        fkey_col_to_pkey_table={},
    )
    friendships = SimpleNamespace(
        df=pd.DataFrame({"user_id": [1, 2], "friend_id": [2, 3], "created_at": [1, 2]}),
        pkey_col=None,
        time_col="created_at",
        fkey_col_to_pkey_table={"user_id": "users", "friend_id": "users"},
    )
    database = SimpleNamespace(table_dict={"users": users, "friendships": friendships})

    _root, tables, relationships = _edge_instances(
        database,
        root_table="users",
        root_frame=users.df,
        search_root_frame=users.df,
        schema_depth=3,
        sample_rows=100,
        random_state=42,
        max_base_table_occurrences=2,
    )

    friendship = next(
        table for table in tables if table.name.startswith("users__friendships")
    )
    peer_edge = next(
        edge
        for edge in relationships
        if edge.parent == friendship.name
        and edge.parent_key == "friend_id"
        and edge.reduce is False
    )
    peer_user = next(table for table in tables if table.name == peer_edge.child)
    assert friendship.name != peer_user.name
    assert peer_user.path_base_tables == ("users", "friendships", "users")

    table_map = {"users": _root, **{table.name: table for table in tables}}
    acyclic_tables, acyclic_edges = _topology_for_config(
        table_map,
        relationships,
        GraphConfig(max_base_table_occurrences=1),
    )
    cyclic_tables, cyclic_edges = _topology_for_config(
        table_map,
        relationships,
        GraphConfig(max_base_table_occurrences=2),
    )

    assert peer_user.name not in acyclic_tables
    assert peer_edge not in acyclic_edges
    assert peer_user.name in cyclic_tables
    assert peer_edge in cyclic_edges


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


def test_problem_can_keep_multiple_full_search_frames() -> None:
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
        search_full_data=True,
        search_training_frames=2,
        max_train_timestamps=2,
        schema_depth=1,
    )

    search_labels = problem.label_node.search_source
    search_train = search_labels.loc[
        search_labels["__kurversc_relbench_split__"] == "train"
    ]
    assert len(search_labels) == 12
    assert len(search_train) == 8
    assert search_train["timestamp"].nunique() == 2
    assert len(problem.parent_node.search_source) == 12


def test_problem_separates_plan_search_from_temporal_fidelity_cohort() -> None:
    train = pd.DataFrame(
        {
            "uid": range(12),
            "timestamp": pd.to_datetime(
                ["2020-01-01"] * 4 + ["2020-02-01"] * 4 + ["2020-03-01"] * 4
            ),
            "target": [0, 1] * 6,
        }
    )
    validation = pd.DataFrame(
        {
            "uid": range(12, 16),
            "timestamp": pd.to_datetime(["2020-04-01"] * 4),
            "target": [0, 1, 0, 1],
        }
    )
    users = SimpleNamespace(
        df=pd.DataFrame({"uid": range(16)}),
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
        sample_rows=100,
        search_full_data=True,
        search_training_frames=1,
        fidelity_training_frames=3,
        max_train_timestamps=3,
        schema_depth=1,
    )

    search_labels = problem.label_node.search_source
    fidelity_labels = problem.label_node.fidelity_source
    search_train = search_labels.loc[
        search_labels["__kurversc_relbench_split__"] == "train"
    ]
    fidelity_train = fidelity_labels.loc[
        fidelity_labels["__kurversc_relbench_split__"] == "train"
    ]
    assert search_train["timestamp"].nunique() == 1
    assert fidelity_train["timestamp"].nunique() == 3
    assert len(problem.parent_node.search_source) == 8
    assert len(problem.parent_node.fidelity_source) == 16


def test_problem_adds_full_strictly_temporal_target_history() -> None:
    train = pd.DataFrame(
        {
            "uid": [0, 1, 0, 2],
            "timestamp": pd.to_datetime(
                ["2020-01-01", "2020-01-01", "2020-02-01", "2020-02-01"]
            ),
            "target": [0.1, 0.2, 0.3, 0.4],
        }
    )
    validation = pd.DataFrame(
        {
            "uid": [0, 1],
            "timestamp": pd.to_datetime(["2020-03-01", "2020-03-01"]),
            "target": [0.9, 0.8],
        }
    )
    users = SimpleNamespace(
        df=pd.DataFrame({"uid": [0, 1, 2]}),
        pkey_col="uid",
        time_col=None,
        fkey_col_to_pkey_table={},
    )
    task = SimpleNamespace(
        entity_table="users",
        entity_col="uid",
        time_col="timestamp",
        target_col="target",
        task_type="regression",
    )

    problem = relbench_problem_from_objects(
        task,
        SimpleNamespace(table_dict={"users": users}),
        SimpleNamespace(df=train),
        SimpleNamespace(df=validation),
        sample_rows=2,
        max_train_timestamps=2,
        schema_depth=1,
    )

    assert problem.label_node.target_history is True
    assert len(problem.label_node.history_source) == len(train) + len(validation)
    assert all(table.name != TARGET_HISTORY_TABLE for table in problem.tables)

    root_name, tables = _normalize_tables(problem.parent_node, problem.tables)
    tables, relationships, prediction_history = _prepare_target_history(
        tables,
        problem.relationships,
        root_name=root_name,
        labels_spec=problem.label_node,
        task="regression",
        validation_fraction=0.2,
        random_state=42,
        connection=None,
        duckdb_memory_limit="128GB",
        duckdb_max_temp_directory_size="128GB",
    )
    history = tables[TARGET_HISTORY_TABLE]
    assert len(history.search_source) == len(train)
    assert len(history.source) == len(train)
    assert len(prediction_history) == len(train) + len(validation)
    assert history.sample_in_search is False
    assert history.cutoff_boundary == "strict"
    assert history.columns == ("uid", TARGET_HISTORY_TIME, TARGET_HISTORY_VALUE)
    assert history.source[TARGET_HISTORY_TIME].iloc[0] == (train["timestamp"].iloc[0])
    assert relationships[-1].parent == "users"
    assert relationships[-1].child == TARGET_HISTORY_TABLE

    with _table_workspace(
        None,
        tables,
        sample_rows=2,
        random_state=42,
        search=True,
    ) as workspace:
        assert len(workspace.frame(TARGET_HISTORY_TABLE)) == len(train)

    # A training row sees only earlier targets, never its own same-cutoff label.
    with _table_workspace(
        None,
        tables,
        sample_rows=2,
        random_state=42,
        search=False,
    ) as workspace:
        materialized = _materialize_at_cutoff(
            workspace,
            tables,
            relationships,
            root_name,
            GraphConfig(
                feature_families=("base",),
                depth=1,
                auto_annotate_features=False,
            ),
            train.loc[(train["uid"] == 0) & (train["timestamp"] == "2020-02-01")],
            problem.label_node,
            cut_date=pd.Timestamp("2020-02-01").to_pydatetime(),
            compute_period_days=3650,
            verbose=False,
        )
    assert materialized["targethist___kurversc_prior_target___avg"].iloc[0] == 0.1

    with _table_workspace(
        None,
        tables,
        sample_rows=2,
        random_state=42,
        search=False,
    ) as workspace:
        anchor_materialized = _materialize_at_cutoff(
            workspace,
            tables,
            relationships,
            root_name,
            GraphConfig(
                feature_families=("base",),
                depth=1,
                auto_annotate_features=False,
                include_target_history=False,
            ),
            train.loc[(train["uid"] == 0) & (train["timestamp"] == "2020-02-01")],
            problem.label_node,
            cut_date=pd.Timestamp("2020-02-01").to_pydatetime(),
            compute_period_days=3650,
            verbose=False,
        )
    assert not any(
        column.startswith("targethist_") for column in anchor_materialized.columns
    )


def test_problem_can_disable_target_history_for_an_ablation() -> None:
    labels = pd.DataFrame(
        {
            "uid": [0, 1],
            "timestamp": pd.to_datetime(["2020-01-01", "2020-01-01"]),
            "target": [0.1, 0.2],
        }
    )
    users = SimpleNamespace(
        df=pd.DataFrame({"uid": [0, 1]}),
        pkey_col="uid",
        time_col=None,
        fkey_col_to_pkey_table={},
    )
    task = SimpleNamespace(
        entity_table="users",
        entity_col="uid",
        time_col="timestamp",
        target_col="target",
        task_type="regression",
    )

    problem = relbench_problem_from_objects(
        task,
        SimpleNamespace(table_dict={"users": users}),
        SimpleNamespace(df=labels),
        SimpleNamespace(df=labels),
        sample_rows=2,
        include_target_history=False,
    )

    assert problem.label_node.target_history is False
    root_name, tables = _normalize_tables(problem.parent_node, problem.tables)
    prepared_tables, prepared_relationships, prediction_history = (
        _prepare_target_history(
            tables,
            problem.relationships,
            root_name=root_name,
            labels_spec=problem.label_node,
            task="regression",
            validation_fraction=0.2,
            random_state=42,
            connection=None,
            duckdb_memory_limit="128GB",
            duckdb_max_temp_directory_size="128GB",
        )
    )
    assert TARGET_HISTORY_TABLE not in prepared_tables
    assert prepared_relationships == problem.relationships
    assert prediction_history is None
