import numpy as np
import pandas as pd
import pytest

from scripts.audit_dormant_reactivation import matched_examples, positive_pair_errors


def test_pair_error_accounting_matches_auc_with_ties():
    from sklearn.metrics import roc_auc_score
    truth = np.array([1, 0, 1, 0, 1, 0])
    prediction = np.array([.3, .3, .1, .9, .9, .2])
    errors = positive_pair_errors(truth, prediction)
    np.testing.assert_array_equal(errors[truth == 0], 0)
    assert errors.sum()/9 == pytest.approx(1-roc_auc_score(truth, prediction))
    assert errors[0] == 1.5


def test_pair_error_rejects_unaligned_nonbinary_and_nonfinite_inputs():
    with pytest.raises(ValueError, match="Aligned binary"):
        positive_pair_errors([1, 2], [.1, .2])
    with pytest.raises(ValueError, match="Finite"):
        positive_pair_errors([1, 0], [np.nan, .2])


def test_matching_uses_recency_and_history_not_high_negative_scores():
    frame = pd.DataFrame(dict(truth=[1, 0, 0, 1], rc_all_recency=[100., 101., 900., 1.],
        rc_all_days=[5, 5, 1, 5], recurrence_pair_errors=[10, 0, 0, 50]))
    match = matched_examples(frame, horizon=90, count=1)
    assert match[0]["positive_row"] == 0
    assert match[0]["negative_row"] == 1
