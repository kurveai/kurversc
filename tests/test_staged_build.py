import ast
from pathlib import Path

from staged_build import relocated_stage,stage_sources


def test_wheel_ships_only_complete_staged_dependency_closure_not_the_research_archive():
    root = Path(__file__).resolve().parents[1]
    sources = stage_sources(root)
    assert 20<len(sources)<50
    names = {p.stem for p in sources}
    assert {"run_portable_staged","run_admissible_staged","run_generic_graph",
        "benchmark_generic_readouts","generic_staged_helpers","streaming_feature_cache"}<=names
    assert not {"schema_exposure_challenger","avito_exposure_signals",
        "schema_semantic_recurrence","benchmark_recovered_readouts",
        "benchmark_complete_count_challenger","benchmark_production_sequence",
        "confirm_production_sequence","run_reproducible_staged_cloud",
        "run_bounded_view_validation_20261020","confirm_bounded_view_recovery_20261020",
        "run_recovered_staged_cloud","run_staged_complete_cloud"}&names
    assert "benchmark_temporal_gap_stress" not in names
    assert "smoke_portable_staged" not in names
    for source in sources:
        relocated = relocated_stage(source.read_text())
        ast.parse(relocated)
        assert "from scripts import" not in relocated
        assert "scripts." not in relocated
        assert 'Path(__file__).resolve().parents[1]' not in relocated


def test_relocation_changes_module_and_source_paths_without_alias_import_hacks():
    original = 'from scripts import example\nimport scripts.other\npath="scripts/example.py"\nroot=Path(__file__).resolve().parents[1]\n'
    expected = 'from kurversc._staged import example\nimport kurversc._staged.other\npath="kurversc/_staged/example.py"\nroot=Path(__file__).resolve().parents[2]\n'
    assert relocated_stage(original)==expected
