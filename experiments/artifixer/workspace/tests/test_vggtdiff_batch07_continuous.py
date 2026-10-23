"""Camera errors silently waste long diffusion runs; test rigid orbit invariants."""
from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from vggtdiff_batch07_continuous import continuous_orbit


def anchor():
    # At azimuth0, a camera above +Z observes the origin. Add roll deliberately.
    position=np.array([0.,.2,2.]);z=-position/np.linalg.norm(position)
    x=np.cross(z,[0.,1.,0.]);x/=np.linalg.norm(x);y=np.cross(z,x)
    c=np.eye(4);c[:3,:3]=np.stack([x,y,z],axis=1);c[:3,3]=position
    from scipy.spatial.transform import Rotation
    c[:3,:3]=c[:3,:3]@Rotation.from_rotvec([0,0,.17]).as_matrix()
    return np.linalg.inv(c),np.array([[900.,0,415.5],[0,910.,239.5],[0,0,1.]])


def test_anchor_reproduced_without_orientation_or_intrinsics_jump():
    w,k=anchor();result=continuous_orbit(w,k,0,[0,0,0],[0,2,0],[0])
    np.testing.assert_allclose(result['target_w2c'][0],w,atol=1e-12)
    np.testing.assert_array_equal(result['target_intrinsics'][0],k)


def test_direction_height_distance_and_projection_preserved():
    w,k=anchor();angles=np.array([6,30,90,180,309.375])
    result=continuous_orbit(w,k,0,[0,0,0],[0,1,0],angles)
    for a,target,tk in zip(angles,result['target_w2c'],result['target_intrinsics']):
        tw=np.array(target);position=np.linalg.inv(tw)[:3,3]
        np.testing.assert_allclose(position,[-2*np.sin(np.deg2rad(a)),.2,2*np.cos(np.deg2rad(a))],atol=1e-12)
        np.testing.assert_allclose(tw@np.array([0.,0.,0.,1.]),w@np.array([0.,0.,0.,1.]),atol=1e-12)
        np.testing.assert_allclose(tw[:3,:3]@tw[:3,:3].T,np.eye(3),atol=1e-12)
        assert np.linalg.det(tw[:3,:3])==pytest.approx(1)
        np.testing.assert_array_equal(tk,k)


@pytest.mark.parametrize('up',[[0,0,0],[float('nan'),1,0]])
def test_invalid_rotation_axis_rejected(up):
    w,k=anchor()
    with pytest.raises(ValueError):continuous_orbit(w,k,0,[0,0,0],up,[270])
