import numpy as np
import pandas as pd
import pytest

from scripts.avito_exposure_signals import ExposureAdjustedSignals


def source():
    # Query users have identical marginals; peer users establish unequal ad quality.
    users = [1, 1, 2, 2, 3, 3, 4, 4]
    searches = pd.DataFrame({"SearchID": range(8), "UserID": users,
                            "SearchDate": pd.to_datetime(["2020-01-02"]*8)})
    events = pd.DataFrame({"SearchID": range(8), "AdID": [10, 20]*4,
        "Position": [1]*8, "IsClick": [1., 0., 0., 1., 1., 0., 1., 0.],
        "HistCTR": [.75, .25]*4, "SearchDate": pd.to_datetime(["2020-01-02"]*8)})
    ads = pd.DataFrame({"AdID": [10, 20]})
    return searches, events, ads


def transform(data, ids=(1,2), cutoff="2020-01-05"):
    plan = ExposureAdjustedSignals(*data, threads=1)
    try:
        return plan.transform(ids, cutoff)
    finally:
        plan.close()


def test_joint_readout_distinguishes_same_marginals():
    out = transform(source())
    pd.testing.assert_series_equal(out["activity"].iloc[0], out["activity"].iloc[1], check_names=False)
    # Clicking the harder ad produces more unexpected clicks, even with equal raw rate.
    assert out["histctr"].exposure_histctr_all_unexpected_clicks.iloc[1] > out["histctr"].exposure_histctr_all_unexpected_clicks.iloc[0]
    assert out["ad"].exposure_ad_all_unexpected_clicks.iloc[1] > out["ad"].exposure_ad_all_unexpected_clicks.iloc[0]


def test_peer_priors_exclude_all_query_user_events():
    data = source()
    plan = ExposureAdjustedSignals(*data, threads=1)
    plan.transform([1], "2020-01-05")
    before = plan.con.execute("SELECT object_id, expected_ad, expected_ad_position FROM scored ORDER BY object_id").df()
    plan.close()
    events = data[1].copy()
    events.loc[events.SearchID.isin([0,1]), "IsClick"] = [0.,1.]
    plan = ExposureAdjustedSignals(data[0], events, data[2], threads=1)
    plan.transform([1], "2020-01-05")
    after = plan.con.execute("SELECT object_id, expected_ad, expected_ad_position FROM scored ORDER BY object_id").df()
    plan.close()
    pd.testing.assert_frame_equal(before, after)
    assert before.expected_ad.tolist() == pytest.approx([(2+20*.5)/(3+20), (1+20*.5)/(3+20)])


def test_strict_cutoff_both_search_and_impression_no_future_response_leak():
    searches, events, ads = source()
    extra_search = pd.DataFrame({"SearchID": [8,9,10], "UserID": [1,3,3],
        "SearchDate": pd.to_datetime(["2020-01-05", "2020-01-02", "2020-01-05"])})
    extra_events = pd.DataFrame({"SearchID": [8,9,10], "AdID": [10]*3, "Position": [1]*3,
        "IsClick": [1.]*3, "HistCTR": [.8]*3,
        "SearchDate": pd.to_datetime(["2020-01-05", "2020-01-05", "2020-01-02"])})
    baseline = transform((searches,events,ads))
    expanded = transform((pd.concat([searches,extra_search]),pd.concat([events,extra_events]),ads))
    for arm in baseline:
        pd.testing.assert_frame_equal(baseline[arm], expanded[arm])


def test_no_fanout_duplicates_empty_users_missing_objects():
    searches, events, ads = source()
    events = pd.concat([events,events.iloc[[0]]],ignore_index=True)
    events.loc[1,"AdID"] = np.nan
    out = transform((searches,events,ads), [1,5,1])
    assert out["activity"].exposure_activity_all_exposures.tolist() == [3.,0.,3.]
    for frame in out.values():
        pd.testing.assert_series_equal(frame.iloc[0], frame.iloc[2], check_names=False)
        assert not np.isinf(frame.to_numpy()).any()


def test_ad_position_baseline_changes_with_position():
    searches,events,ads=source()
    events.AdID=10
    events.Position=[1,2,1,2,1,2,1,2]
    events.IsClick=[1.,0.,0.,1.,1.,0.,1.,0.]
    plan=ExposureAdjustedSignals(searches,events,ads,threads=1)
    plan.transform([1],"2020-01-05")
    scored=plan.con.execute("SELECT context, expected_ad, expected_ad_position FROM scored ORDER BY context").df()
    plan.close()
    assert scored.expected_ad.nunique()==1
    assert scored.expected_ad_position.iloc[0]>scored.expected_ad_position.iloc[1]


def test_input_order_invariance():
    data=source()
    baseline=transform(data)
    shuffled=transform(tuple(frame.sample(frac=1,random_state=3) for frame in data))
    for arm in baseline:
        pd.testing.assert_frame_equal(baseline[arm],shuffled[arm],atol=1e-6,rtol=1e-6)


def test_declared_keys_and_binary_response_validation():
    searches,events,ads=source()
    with pytest.raises(ValueError,match="unique"):
        ExposureAdjustedSignals(pd.concat([searches,searches.iloc[[0]]]),events,ads)
    events.loc[0,"IsClick"]=2
    with pytest.raises(ValueError,match="binary"):
        transform((searches,events,ads))
