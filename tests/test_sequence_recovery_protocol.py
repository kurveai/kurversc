import numpy as np
import pytest

from scripts import sequence_recovery_protocol as protocol


def rows():
    return [dict(origin=origin, seed=seed, truth=np.array([0, 1, 0, 1]), groups=np.arange(4),
        native=np.array([.2, .8, .3, .7]), basis=np.array([.1, .9, .25, .75]),
        exact=np.array([.1, .85, .2, .8])) for origin in ("2001-01-01", "2001-02-01") for seed in range(3)]


def test_incomplete_evidence_is_exact_incumbent():
    result = protocol.choose(rows()[:-1])
    assert result["selected"] == "incumbent"
    assert result["coefficients"] == [1., 0., 0.]


def test_duplicate_cases_are_rejected():
    with pytest.raises(ValueError, match="Duplicate"):
        protocol.choose(rows()+rows()[:1])


def test_existing_two_stage_composition_and_order(monkeypatch):
    evidence = []
    def gate(data, **kwargs):
        evidence.append((data, kwargs))
        return {"selected": "sequence_basis_and_exact_time" if len(evidence) == 1 else "sequence_weight_paired_0.5"}
    monkeypatch.setattr(protocol, "matched_gate", gate)
    result = protocol.choose(rows())
    assert result["coefficients"] == [.3828125, .3125, .3046875]
    assert all(options["allowance"] == 1024 for _, options in evidence)
    r = rows()[0]
    np.testing.assert_array_equal(evidence[1][0][0]["control"],
        .765625*r["native"]+.125*r["basis"]+.109375*r["exact"])
    assert result["official_targets_used_for_selection"] is False


def test_invalid_probabilities_and_coefficients():
    with pytest.raises(ValueError):
        protocol.combine([0.], [np.nan], [.5], [1, 0, 0])
    with pytest.raises(ValueError):
        protocol.combine([0.], [.5], [.5], [1, -.5, .5])
