"""Pure KurveRSC API run for the official rel-stack/user-badge task."""

import kurversc


problem = kurversc.load_relbench_problem(
    "rel-stack",
    "user-badge",
    sample_rows=2_000,
    max_train_timestamps=1,
    schema_depth=2,
)

result = kurversc.fit(
    **problem.fit_kwargs(),
    sample_rows=2_000,
    max_depth=2,
    feature_family_stages=[("base",)],
    auto_annotate_options=[True],
    model_params={"iterations": 30, "depth": 5},
)

print("train_cutoffs:", problem.train_timestamps)
print("validation_cutoffs:", problem.validation_timestamps)
print("best:", result.best_trial.as_record())
print("recommended:", result.recommended_trial.as_record())
print(result.results.to_string(index=False))
