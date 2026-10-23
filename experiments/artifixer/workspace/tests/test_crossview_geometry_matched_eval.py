import numpy as np

from crossview_geometry_matched_eval import mask_distances, paired_deltas


def pair(a,b,count,error):
    return dict(frame_a=a,frame_b=b,n_matches=count,low_support=count<30,
                epi_median_px=error,epi_p90_px=error*2,coverage_a=.5,coverage_b=.6)


def test_lost_support_is_retained_and_not_claimed_as_improvement():
    gt=[pair(10,20,60,1.),pair(20,30,70,2.)]
    gen=[pair(10,20,3,.01),pair(20,30,65,3.)]
    rows,summary=paired_deltas(gt,gen,[[10,20],[20,30]])
    assert len(rows)==2 and rows[0]['lost_support']
    assert summary['n_fixed']==2 and summary['joint_supported']==1
    assert summary['lost_gt_support']==1
    assert summary['joint_median_paired_delta_px']==1.


def test_reference_pair_direction_and_frame_identity_are_preserved():
    gt=[pair(1340,0,60,1.)];gen=[pair(1340,0,60,2.)]
    rows,summary=paired_deltas(gt,gen,[[1340,0]])
    assert rows[0]['frame_a']==1340 and rows[0]['frame_b']==0
    assert summary['joint_median_paired_delta_px']==1.


def test_identical_masks_have_zero_boundary_distance():
    a=np.zeros((100,100),np.uint8);a[20:80,20:80]=255
    iou,mean,p90=mask_distances(a,a)
    assert iou==1. and mean==0. and p90==0.


def test_translation_is_measured_without_alignment():
    a=np.zeros((100,100),np.uint8);a[20:70,20:70]=255
    b=np.zeros_like(a);b[20:70,25:75]=255
    iou,mean,p90=mask_distances(a,b)
    assert 0<iou<1 and mean>0 and p90>0
