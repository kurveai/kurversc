import ast
from pathlib import Path

from staged_build import relocated_stage,stage_sources


def test_wheel_ships_only_complete_staged_dependency_closure_not_the_research_archive():
    root = Path(__file__).resolve().parents[1]
    sources = stage_sources(root)
    assert 50<len(sources)<150
    names = {p.stem for p in sources}
    assert {"run_portable_staged","run_production_finalist_cloud_v2","benchmark_complete_count_challenger"}<=names
    assert {"run_streamed_sequence_cloud", "streaming_feature_cache", "benchmark_production_sequence",
        "confirm_production_sequence"} <= names
    assert {"run_reproducible_staged_cloud", "run_bounded_view_validation_20261020",
        "confirm_bounded_view_recovery_20261020", "capture_owned_pooled_native_20261020"} <= names
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
