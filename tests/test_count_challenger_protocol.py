import copy

import pytest

from scripts.count_challenger_protocol import count_stage_eligible, validated_count_recipe


def fixture(weight=.125):
    name = f"count_likelihood_{weight:g}"
    meta = dict(metric="mae", fold="current", atomic_date_replay=True)
    manifest = dict(profile="conditional_poisson_median", metadata=meta,
        weights=[.125, .5, 1.], seeds=[0, 1, 2], folds=[0, 1, 2], allowance=1024)
    records = [dict(fold=f, seed=s, native_archive_max_error=0.,
        scores={"fresh":1., "archive":1., "fresh_"+name:.9, "archive_"+name:.9})
        for f in range(3) for s in range(3)]
    gate = dict(candidates={name:dict(reason="eligible", folds=[dict(
        recent_lower_bound=.01, temporal_gain_lower_bound=.01)])})
    result = dict(selected=name, complete_grid=True, official_test_accessed=False,
        default_changed=False, records=records, gates=dict(fresh=copy.deepcopy(gate), archive=copy.deepcopy(gate)))
    return result, manifest, meta


@pytest.mark.parametrize("weight", [.125, .5, 1.])
def test_accepts_only_the_declared_validation_qualified_weight(weight):
    result, manifest, meta = fixture(weight)
    selected = validated_count_recipe(result, manifest, meta)
    assert selected["weight"] == weight
    assert not selected["official_targets_used_for_selection"]


@pytest.mark.parametrize("problem", ["incomplete", "duplicate", "native_mismatch", "older_loss", "recent_uncertain", "temporal_uncertain", "test_access"])
def test_rejects_incomplete_unstable_or_test_selected_evidence(problem):
    result, manifest, meta = fixture()
    if problem == "incomplete": result["records"].pop()
    elif problem == "duplicate": result["records"].append(copy.deepcopy(result["records"][0]))
    elif problem == "native_mismatch": result["records"][0]["native_archive_max_error"] = .001
    elif problem == "older_loss": result["records"][0]["scores"]["archive_count_likelihood_0.125"] = 1.1
    elif problem == "test_access": result["official_test_accessed"] = True
    else:
        key = "recent_lower_bound" if problem == "recent_uncertain" else "temporal_gain_lower_bound"
        result["gates"]["fresh"]["candidates"][result["selected"]]["folds"][-1][key] = -.01
    with pytest.raises(ValueError):
        validated_count_recipe(result, manifest, meta)


def test_failed_challenger_preserves_incumbent_without_official_replay():
    assert validated_count_recipe(dict(selected="incumbent"), {}, {}) is None


def production_fixture():
    result, manifest, meta = fixture()
    manifest.update(profile="complete_conditional_count_production_rule",
        selection_rule="existing_production_temporal_veto", older_all_seed_relative_loss_cap=.01)
    result["selection_rule"] = manifest["selection_rule"]
    return result, manifest, meta


def test_production_rule_is_explicit_and_retains_the_older_practical_loss_cap():
    result, manifest, meta = production_fixture()
    result["records"][0]["scores"]["archive_count_likelihood_0.125"] = 1.005
    assert validated_count_recipe(result, manifest, meta)["weight"] == .125
    result["records"][1]["scores"]["archive_count_likelihood_0.125"] = 1.02
    with pytest.raises(ValueError):
        validated_count_recipe(result, manifest, meta)


def test_production_rule_never_tolerates_a_recent_seed_loss_or_undeclared_policy():
    result, manifest, meta = production_fixture()
    result["records"][-1]["scores"]["fresh_count_likelihood_0.125"] = 1.001
    with pytest.raises(ValueError): validated_count_recipe(result, manifest, meta)
    result, manifest, meta = production_fixture()
    manifest["older_all_seed_relative_loss_cap"] = .02
    with pytest.raises(ValueError): validated_count_recipe(result, manifest, meta)


def test_applicability_never_replaces_a_stronger_readout_or_changes_other_domains():
    raw = dict(metadata=dict(metric="mae", fold="current", atomic_date_replay=True),
        folds=[{}, {}, {}], seeds=[0, 1, 2])
    policy = dict(mode="frame_ensemble", readout="direct", blend_weight=1.)
    recovered = dict(selected="integer_counts", projection="integer_counts", baseline_signal="incumbent")
    assert count_stage_eligible(raw, policy, recovered)
    for selected in ("zero_aware_0.5", "joint_semantic32", "object_full", "incumbent"):
        assert not count_stage_eligible(raw, policy, {**recovered, "selected":selected})
    for field, value in [("mode","pooled"), ("readout","ordered_median"), ("blend_weight",.125)]:
        assert not count_stage_eligible(raw, {**policy, field:value}, recovered)
    assert not count_stage_eligible({**raw, "metadata":dict(metric="roc_auc")}, policy, recovered)
