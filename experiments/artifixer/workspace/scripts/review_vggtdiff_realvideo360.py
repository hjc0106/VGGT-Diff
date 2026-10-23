"""Verify the generated orbit and summarize image changes at batch boundaries."""
import json
from pathlib import Path
import numpy as np
from PIL import Image
import vggtdiff_realvideo360_8 as run
from vggtdiff_white_car import save, digest


def main():
    out=run.OUT;t=run.read(out/'trajectory.json');v=run.read(out/'verification.json')
    assert v['frame_count']==v['unique_images']==64
    parent=run.read(run.PARENT/'trajectory.json')
    assert t['source_w2c']==parent['source_w2c'] and t['source_intrinsics']==parent['source_intrinsics']
    assert np.allclose(t['target_angles_deg'],np.arange(64)*5.625)
    for w in t['target_w2c']:
        r=np.array(w)[:3,:3]
        assert np.allclose(r@r.T,np.eye(3),atol=1e-6) and abs(np.linalg.det(r)-1)<1e-6
    weights=run.read(run.PARENT/'inference_manifest.json')['weights']
    for entry in weights:
        stat=Path(entry['path']).stat()
        assert stat.st_size==entry['size'] and stat.st_mtime_ns==entry['mtime_ns']
    arrays=[]
    for r in v['frames']:
        p=Path(r['path']);assert digest(p)==r['sha256']
        with Image.open(p) as im:
            assert im.size==(832,480)
            arrays.append(np.asarray(im.convert('RGB').resize((208,120)),dtype=float)/255)
    pairs=[]
    for i in range(64):
        j=(i+1)%64
        pairs.append(dict(from_frame=i,to_frame=j,batch_boundary=j%8==0,cyclic_seam=j==0,
            rgb_mean_absolute_difference=float(np.abs(arrays[i]-arrays[j]).mean())))
    inside=[p['rgb_mean_absolute_difference'] for p in pairs if not p['batch_boundary']]
    between=[p['rgb_mean_absolute_difference'] for p in pairs if p['batch_boundary']]
    audit=dict(frames=64,all_hashes_verified=True,source_rgb_and_cameras_unchanged=True,proper_camera_rotations=True,
        model_file_stats_unchanged=True,angles_deg=t['target_angles_deg'],video=v['video'],
        consecutive_image_change=dict(pairs=pairs,within_batch_median=float(np.median(inside)),batch_boundary_median=float(np.median(between)),
            caution='RGB change includes camera motion, background hallucination, lighting, and detail changes; it is not a geometric consistency score or ground-truth error'))
    save(out/'final_audit.json',audit)
    print(json.dumps({k:audit[k] for k in ['frames','all_hashes_verified','source_rgb_and_cameras_unchanged','model_file_stats_unchanged']}))
    print('median RGB change: within batch',np.median(inside),'boundaries',np.median(between))


if __name__=='__main__':main()
