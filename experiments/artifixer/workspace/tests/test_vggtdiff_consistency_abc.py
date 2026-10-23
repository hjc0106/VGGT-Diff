"""Tests target experiment confounds and metric failure modes, not just plumbing."""
import json
import cv2
cv2.setNumThreads(2)
import numpy as np
import pytest
from PIL import Image
import vggtdiff_consistency_abc as abc


def scene(indices=range(16), n=45):
    rng=np.random.default_rng(4)
    xyz=np.c_[rng.uniform(.0,.9,n),rng.uniform(-.6,.6,n),rng.uniform(4,6,n)]
    des=rng.normal(size=(n,128)).astype(np.float32)
    k=np.array([[400.,0,416],[0,400,240],[0,0,1]])
    frames=[]
    for i in indices:
        t=np.eye(4);t[0,3]=-.15*i
        points=xyz+t[:3,3];uv=points@k.T;uv=uv[:,:2]/uv[:,2:]
        frames.append(dict(index=i,original_frame=i,K=k.copy(),T=t,xy=uv,
                           descriptors=des.copy(),mask=np.full((480,832),255,np.uint8),azimuth_deg=float(i*2)))
    return frames


def test_overlap_cameras_are_not_in_relay_conditions():
    assert set(abc.CALLS['OB']['targets'])&set(abc.CALLS['A0']['targets'])=={6,7}
    assert not set(abc.CHAIN_REFS)&set(abc.CALLS['OB']['targets'])
    assert abc.CALLS['B1']['source']==abc.CALLS['OB']['source']=='A0'


def test_exact_boundary_is_compared_inside_c_and_across_split(tmp_path):
    full,_=abc.assess(scene(),tmp_path/'full')
    joint,_=abc.assess(scene(range(4,12)),tmp_path/'joint')
    keys=lambda rows:{(r['frame_a'],r['frame_b']) for r in rows}
    assert set(abc.BOUNDARY_PAIRS)<=keys(full)&keys(joint)
    _,s=abc.assess(scene(range(4,12)),tmp_path/'repeat')
    assert s['boundary']['n_pairs']==3
    assert s['boundary']['supported_pairs']==3
    assert len(s['triples'])==6
    assert s['boundary_raw']['n_tracks']==90
    assert s['boundary_raw']['computed_inside_image_p90_px']<1e-8


def test_no_matches_remains_insufficient_not_zero_error(tmp_path):
    frames=scene(range(3),n=0)
    _,s=abc.assess(frames,tmp_path)
    assert s['pairs']['n_pairs']==3
    assert s['pairs']['supported_pairs']==0
    assert s['pairs']['supported_median_pair_epi_px'] is None
    assert s['raw']['computed_inside_image_median_px'] is None
    assert s['raw']['n_tracks']==0


def test_three_view_detects_displacement_along_epipolar_line(tmp_path):
    frames=scene(range(3))
    frames[1]['xy'][:,0]+=7
    rows,s=abc.assess(frames,tmp_path)
    # Horizontal epipolar displacement is invisible to the pair test.
    assert max(r['epi_median_px'] for r in rows)<1e-8
    assert s['raw']['computed_inside_image_median_px']>5


def test_same_camera_shift_not_hidden_by_alignment():
    a,b=scene(range(2))
    b=dict(a,xy=a['xy']+np.array([5,0]))
    s=abc.same_camera(a,b)
    assert s['displacement_median_px']==pytest.approx(5)
    b=dict(b,T=np.eye(4));b['T'][0,3]=1
    with pytest.raises(AssertionError):abc.same_camera(a,b)


def test_frozen_protocol_does_not_overwrite_existing(tmp_path):
    p=tmp_path/'protocol.json'
    abc.frozen_write(p,{'seed':1})
    with pytest.raises(ValueError):abc.frozen_write(p,{'seed':2})
    assert json.loads(p.read_text())=={'seed':1}


def test_lost_support_retained_in_fixed_pairs():
    def pair(n,e):
        return dict(frame_a=7,frame_b=8,n_matches=n,low_support=n<30,
                    epi_median_px=e,epi_p90_px=e*2,coverage_a=.3,coverage_b=.4)
    rows,s=abc.paired_deltas([pair(50,1.)],[pair(2,.001)],[(7,8)])
    assert rows[0]['lost_support']
    assert s['n_fixed']==1 and s['joint_supported']==0
    assert s['joint_median_paired_delta_px'] is None


def test_json_freeze_is_idempotent_for_tuple_pair_ids(tmp_path):
    p=tmp_path/'protocol.json'
    abc.frozen_write(p,{'pairs':[(7,8)]})
    abc.frozen_write(p,{'pairs':[(7,8)]})


def test_evaluate_and_partial_report_keep_boundary_and_gt_comparisons(tmp_path,monkeypatch):
    monkeypatch.setattr(abc,'OUT',tmp_path)
    monkeypatch.setattr(abc,'validate',lambda: {})
    monkeypatch.setattr(abc,'require_review',lambda *a: None)
    metadata=[dict(index=i,video_frame=1000+i,azimuth_deg=2.*i) for i in range(16)]
    abc.save(tmp_path/'targets.json',{'frames':metadata})
    abc.save(tmp_path/'references.json',{'frames':[dict(index=100+i,video_frame=i) for i in range(6)]})
    def features(rows,root,region):return scene([r['index'] for r in rows])
    monkeypatch.setattr(abc,'load_features',features)
    monkeypatch.setattr(abc,'generated_rows',lambda s,c:[metadata[i] for i in abc.CALLS[c]['targets']])
    abc.evaluate([abc.SEEDS[0]])
    abc.report()
    result=abc.read(tmp_path/'comparison_summary.json')['vehicle']['seeds'][str(abc.SEEDS[0])]
    for group in ['A','B','C']:
        assert result[group]['boundary_gt']['joint_supported']==3
        assert result[group]['boundary_gt']['joint_median_paired_delta_px']==pytest.approx(0)
    assert result['overlap_relay'][0]['displacement_median_px']==0
    assert not abc.read(tmp_path/'status.json')['completed_comparison']
    assert '部分种子' in (tmp_path/'REPORT.md').read_text()
    assert (tmp_path/'evaluation/vehicle'/f'seed_{abc.SEEDS[0]}'/'C/delta_triples_gt.csv').exists()


def test_vectorized_coverage_exactly_matches_legacy_definition():
    rng=np.random.default_rng(9)
    for sparse in [False,True]:
        mask=np.zeros((25,40),np.uint8);mask[3:20,6:35]=255
        if sparse:mask[6:17,9:31]=0
        points=rng.uniform([-5,-4],[45,30],size=(30,2))
        assert abc.coverage_fraction(points,mask)==abc.geom.coverage_fraction(points,mask)
    assert abc.coverage_fraction([],np.zeros((5,5))) is None


def test_fast_pair_metrics_match_legacy_numeric_values():
    a,b=scene(range(2))
    got,ia,ib=abc.pair_metrics(a,b)
    expected=abc.geom.pair_record(0,1,a['xy'][ia],b['xy'][ib],a['mask'],b['mask'],a['K'],a['T'],b['K'],b['T'])
    for k in ['epi_median_px','epi_p90_px','coverage_a','coverage_b','n_matches','low_support','translation_norm']:
        assert got[k]==expected[k]
