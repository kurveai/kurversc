import numpy as np
import pandas as pd

from scripts.object_breadth_challenger import ObjectBreadth
from test_activity_timeline_challenger import save_table


def test_keys_only_route_and_future_objects_do_not_select_paths_or_features(tmp_path):
    save_table(tmp_path/"people.parquet",{"id":["a","b"]},primary="id")
    save_table(tmp_path/"things.parquet",{"id":np.arange(1000)},primary="id")
    # 70 pre-query objects meet discovery, but are outside the scoring roots.
    owners = ["other"]*70+["a"]*3+["b"]*3+["a"]
    objects = list(range(70))+[101,101,101]+[102,103,104]+[999]
    dates = [pd.Timestamp("2020-01-01")]*70+[pd.Timestamp("2020-01-09")]*6+[pd.Timestamp("2020-01-10")]
    save_table(tmp_path/"events.parquet",{"owner":owners,"object":objects,"at":dates},
        keys={"owner":"people","object":"things"},timestamp="at")
    provider = ObjectBreadth(tmp_path,"people",training_cutoff="2020-01-10",horizon="4D",temporary=tmp_path/"spill")
    requests = pd.DataFrame({"__event_entity":["a","b","a"],"__event_cutoff":["2020-01-10"]*3})
    try:
        frame = provider.transform(requests,phase_boundary="2020-01-10")
        assert frame.ob_path0_w1_events.tolist() == [3,3,3]
        assert frame.ob_path0_w1_objects.tolist() == [1,3,1]
        assert frame.ob_path0_w1_concentration.tolist() == [1,np.float32(1/3),1]
        assert frame.ob_path0_w1_singletons.tolist() == [0,3,0]
        assert len(frame.columns) <= 42
        assert all(not n.endswith(("owner","object","entity","id")) for n in frame)
        np.testing.assert_array_equal(frame.iloc[0],frame.iloc[2])
        assert provider.paths[0].object_key == "object"
    finally:
        provider.close()
