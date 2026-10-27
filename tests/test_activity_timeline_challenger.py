import json

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts.activity_timeline_challenger import (
    ActivityTimeline, discover_channels, mature_label_state,
)


def save_table(path, rows, *, keys=None, timestamp=None, primary=None):
    table = pa.Table.from_pandas(pd.DataFrame(rows),preserve_index=False)
    metadata = dict(table.schema.metadata or {})
    for key,value in dict(fkey_col_to_pkey_table=keys or {},time_col=timestamp,pkey_col=primary).items():
        metadata[key.encode()] = json.dumps(value).encode()
    pq.write_table(table.replace_schema_metadata(metadata),path)


def test_generic_discovery_and_strict_future_exclusion(tmp_path):
    save_table(tmp_path/'people.parquet',{'pk':[1,2]},primary='pk')
    for name in ('channel_a','channel_b'):
        save_table(tmp_path/f'{name}.parquet',{'owner':[1,1,1,2],
            'when':pd.to_datetime(['2020-01-01','2020-01-03','2020-01-05','2020-01-06'])},
            keys={'owner':'people'},timestamp='when')
    channels = discover_channels(tmp_path,'people')
    assert len(channels)==2
    plan = ActivityTimeline(channels,temp_directory=tmp_path/'spill',threads=1)
    requests = pd.DataFrame({'__event_entity':[2,1,1],
        '__event_cutoff':['2020-01-05']*3})
    try:
        frame = plan.transform(requests,phase_boundary='2020-01-05')
        assert frame.iloc[0].tl_events_730==0
        assert frame.iloc[1].tl_events_730==4
        assert frame.iloc[1].tl_active_days_730==2
        assert frame.iloc[1].tl_multichannel_days_7==2
        np.testing.assert_array_equal(frame.iloc[1],frame.iloc[2])
        with pytest.raises(ValueError,match='authorized'):
            plan.transform(requests,phase_boundary='2020-01-04')
    finally:
        plan.close()


def test_label_state_requires_completed_horizon_and_preserves_order():
    history = pd.DataFrame({'person':[1,1,1,2],
        'when':pd.to_datetime(['2020-01-01','2020-01-03','2020-01-05','2020-01-02']),
        'response':[1,0,1,0]})
    requests = pd.DataFrame({'__event_entity':[2,1,1],
        '__event_cutoff':['2020-01-05']*3})
    frame = mature_label_state(history,requests,entity='person',timestamp='when',target='response',
        horizon='2D',phase_boundary='2020-01-05')
    assert frame.iloc[0].ls_last==0
    assert frame.iloc[1].ls_last==1  # Jan 3 outcome is not strictly mature yet.
    assert frame.iloc[1].ls_count==1
    np.testing.assert_array_equal(frame.iloc[1],frame.iloc[2])
    future_changed = history.copy()
    future_changed.loc[future_changed.when.ge(pd.Timestamp('2020-01-03')),'response']=0
    pd.testing.assert_frame_equal(frame,mature_label_state(future_changed,requests,
        entity='person',timestamp='when',target='response',horizon='2D',phase_boundary='2020-01-05'))


def test_label_state_rejects_ambiguous_duplicate_targets():
    history = pd.DataFrame({'person':[1,1],'when':['2020-01-01']*2,'response':[0,1]})
    requests = pd.DataFrame({'__event_entity':[1],'__event_cutoff':['2020-01-05']})
    with pytest.raises(ValueError,match='unique'):
        mature_label_state(history,requests,entity='person',timestamp='when',target='response',
            horizon='1D',phase_boundary='2020-01-05')
