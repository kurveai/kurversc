"""Run KurveRSC on the relational CSV files in /usr/local/lake/cust_data."""

from pathlib import Path

import kurversc


DATA = Path("/usr/local/lake/cust_data")

kurversc.configure_logging()


result = kurversc.fit(
    parent_node=kurversc.Table(
        DATA / "cust.csv",
        name="customers",
        key="id",
        # cust.csv contains durable entities and genuinely has no creation date.
        timeless=True,
    ),
    label_node=kurversc.GraphLabels(
        table="orders",
        field="id",
        operation="bool",
        period_days=365,
        # Each cutoff is one point-in-time full-training frame. Search uses
        # only one of them; the production refit below uses both.
        train_cutoffs=("2023-01-01", "2024-01-01"),
        validation_cutoffs=("2025-01-01",),
        test_cutoffs=("2026-01-01",),
        target="will_order",
    ),
    tables=[
        kurversc.Table(
            DATA / "orders.csv", name="orders", key="id", date="ts"
        ),
        kurversc.Table(
            DATA / "order_products.csv", name="order_products", key="id"
        ),
        kurversc.Table(
            DATA / "notifications.csv", name="notifications", key="id", date="ts"
        ),
        kurversc.Table(
            DATA / "notification_interactions.csv",
            name="notification_interactions",
            key="id",
            date="ts",
        ),
        kurversc.Table(
            DATA / "notification_interaction_types.csv",
            name="notification_interaction_types",
            key="id",
        ),
    ],
    relationships=[
        kurversc.Relationship(
            parent="customers", child="orders", parent_key="id", child_key="cid"
        ),
        kurversc.Relationship(
            parent="orders",
            child="order_products",
            parent_key="id",
            child_key="order_id",
        ),
        kurversc.Relationship(
            parent="customers",
            child="notifications",
            parent_key="id",
            child_key="ident_cust",
        ),
        kurversc.Relationship(
            parent="notifications",
            child="notification_interactions",
            parent_key="id",
            child_key="notification_id",
        ),
        kurversc.Relationship(
            parent="notification_interactions",
            child="notification_interaction_types",
            parent_key="interaction_type_id",
            child_key="id",
        ),
    ],
    task="classification",
    sample_rows=100_000,
    search_training_frames=1,
    full_training_frames=2,
)

print("best:", result.best_trial.as_record())
print("recommended:", result.recommended_trial.as_record())
print("full validation score:", result.full_validation_score)
print("frozen plan operations:", len(result.execution_plan["records"]))
print("test predictions:")
print(result.test_predictions.to_string(index=False))
print(result.results.to_string(index=False))
