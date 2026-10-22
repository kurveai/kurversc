"""Fresh staged/sequence reproduction with equivalent disk-backed captures.

Predictive configuration is the sequence-recovery recipe, unchanged. This
separate opt-in entry point changes only diagnostic feature-cache storage.
"""

PROFILE = "staged-complete-production-sequence-streamed-v1"


def cache_route(command, *, previous):
    command = previous(command)
    targets = []
    if "-m" in command:
        targets.append(command[command.index("-m")+1])
    if "--stage-module" in command:
        targets.append(command[command.index("--stage-module")+1])
    if "scripts.benchmark_sparse_training" in targets and "--stream-feature-cache" not in command:
        command = [*command, "--stream-feature-cache"]
    return command


def install_execution_contract(staged):
    staged.PROFILE = PROFILE
    staged.CONTRACT = {**staged.CONTRACT, "historical_feature_cache_storage": dict(
        strategy="disk_backed_full_frame_capture", changed_source_sampling=False,
        changed_model_visible_values=False, incumbent_frame_equivalence="exact",
        cohort_order_preserved=True, concatenation_batch_rows=2048)}
    return staged


def install_recipe(*, install_base=None):
    """One recipe installer shared by portable and benchmark execution."""
    from scripts import run_production_finalist_cloud_v2 as base
    from scripts import run_sequence_recovery_cloud as sequence
    from scripts import run_fresh_cloud_campaign as fresh

    original_route = base.route
    base.route = lambda command: cache_route(command, previous=original_route)
    return install_execution_contract(sequence.extend_recipe(
        (install_base or base.install_recipe)(), fresh))


def main():
    from scripts import run_production_finalist_cloud_v2 as base

    original_install = base.install_recipe
    base.install_recipe = lambda: install_recipe(install_base=original_install)
    base.PROFILE, base.MODULE = PROFILE, "scripts.run_streamed_sequence_cloud"
    base.main()


if __name__ == "__main__":
    main()
