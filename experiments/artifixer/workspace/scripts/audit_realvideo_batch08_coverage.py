import json,sys,importlib.util
from pathlib import Path
import numpy as np
sys.path.insert(0,'scripts')
from vggtdiff_white_car import BASE,REPO,save
spec=importlib.util.spec_from_file_location('coverage_colmap',REPO/'thirdparty/map-anything/mapanything/utils/colmap.py');c=importlib.util.module_from_spec(spec);sys.modules[spec.name]=c;spec.loader.exec_module(c)
root=REPO/'pipeline_exp/realworld/scene-1-nofull/3dgs_wcar'
images=c.read_images_binary(root/'colmap/sparse/0/images.bin');byname={r.name:r for r in images.values()}
meta=json.loads((root/'frames_manifest.json').read_text())['images']
g=json.loads((BASE/'vggtdiff_realvideo_unseen_right_near_v2/geometry_check.json').read_text())
sources=json.loads((BASE/'vggtdiff_realvideo_unseen_right_near_v2/camera_estimate.json').read_text())
# Match the historical camera-center alignment exactly: Omega -> independentSfM.
source=np.asarray(g['source_centers']);target=[];links=[]
for row in sources['sources']:
 nearest=min(meta,key=lambda r:abs(r['video_frame_index']-row['source_frame_zero_based']))
 im=byname[nearest['image']];target.append(-c.qvec2rotmat(im.qvec).T@im.tvec)
 links.append(dict(real_reference_frame=row['source_frame_zero_based'],sfm_frame=nearest['video_frame_index']))
target=np.array(target);sm=source.mean(0);tm=target.mean(0);sc=source-sm;tc=target-tm
u,sv,vh=np.linalg.svd(tc.T@sc/len(sc));sign=np.ones(3);sign[-1]=np.linalg.det(u@vh)
rotation=u@np.diag(sign)@vh;scale=(sv*sign).sum()/(sc**2).sum()*len(sc);translation=tm-scale*rotation@sm
fit=scale*(source@rotation.T)+translation
baseline=np.median(np.linalg.norm(target[:,None]-target[None],axis=-1)[np.triu_indices(6,1)])
rmse=np.sqrt(((fit-target)**2).sum(1).mean())/baseline
rows=[]
for r in meta:
 im=byname[r['image']];p=-c.qvec2rotmat(im.qvec).T@im.tvec
 omega_center=((p-translation)@rotation)/scale;d=omega_center-np.array(g['center'])
 angle=float(np.degrees(np.arctan2(-d@np.array(g['left']),d@np.array(g['front'])))%360)
 rows.append(dict(**r,approx_azimuth_deg=angle,omega_center=omega_center.tolist(),image_path=str(root/'images'/r['image'])))
save(BASE/'realvideo_batch08_coverage_v1/sfm_coverage.json',dict(method='IndependentSfM centers aligned to Omega using six nearest-time reference camera-center pairs; approximate azimuth only, not exact per-frame calibration.',alignment_rmse_relative_to_baseline=float(rmse),links=links,frames=rows))
print('RMSE',rmse)
for row in rows:print(row['video_frame_index'],round(row['timestamp_sec'],3),round(row['approx_azimuth_deg'],3))

# Project the existing observed car surface only to locate crops, not to make RGB.
import cv2
from PIL import Image,ImageOps,ImageDraw,ImageFont
from plyfile import PlyData
cloud=PlyData.read(BASE/'vggtdiff_latest_coverage_ab_v1/shared/initial_points.ply')['vertex'].data
xyz=np.column_stack([cloud[k] for k in ['x','y','z']])[:17112]
xyz_sfm=scale*(xyz@rotation.T)+translation
cameras=c.read_cameras_binary(root/'colmap/sparse/0/cameras.bin')
out=BASE/'realvideo_batch08_coverage_v1';(out/'real_crops').mkdir(exist_ok=True)
selected_frames=[2872,2812,2751,2691,2630,2570]
selected=[]
for r in rows:
 im=byname[r['image']];camera=cameras[im.camera_id];params=camera.params
 assert camera.model=='OPENCV'
 k=np.array([[params[0],0,params[2]],[0,params[1],params[3]],[0,0,1]])
 uv=cv2.projectPoints(xyz_sfm,cv2.Rodrigues(c.qvec2rotmat(im.qvec))[0],im.tvec,k,params[4:])[0][:,0]
 camera_xyz=xyz_sfm@c.qvec2rotmat(im.qvec).T+im.tvec
 q=np.quantile(uv,[.01,.99],axis=0)
 r['predicted_car_bbox_1pct_99pct']=q.tolist();r['predicted_center_positive_depth']=bool(np.median(camera_xyz[:,2])>0)
 r['projected_surface_inside_fraction']=float(np.mean((camera_xyz[:,2]>0)&(uv[:,0]>=0)&(uv[:,0]<3840)&(uv[:,1]>=0)&(uv[:,1]<2160)))
 if r['video_frame_index'] in selected_frames:
  # Padding covers pose alignment error. Pixels are cropped from original decoded JPEG only.
  pad=np.maximum((q[1]-q[0])*.65,[300,180]);lo=np.maximum(np.floor(q[0]-pad).astype(int),[0,0]);hi=np.minimum(np.ceil(q[1]+pad).astype(int),[3840,2160])
  assert np.all(hi>lo)
  path=out/'real_crops'/f"frame_{r['video_frame_index']}.png"
  Image.open(r['image_path']).convert('RGB').crop(tuple(np.r_[lo,hi])).save(path)
  r['crop_path']=str(path);r['crop_xyxy']=np.r_[lo,hi].tolist();selected.append(r)
selected.sort(key=lambda r:selected_frames.index(r['video_frame_index']))
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',18)
canvas=Image.new('RGB',(2*832,3*340),'#202020');draw=ImageDraw.Draw(canvas)
for j,r in enumerate(selected):
 x=(j%2)*832;y=(j//2)*340
 with Image.open(r['crop_path']) as image:
  tile=ImageOps.pad(image,(832,300),method=Image.Resampling.LANCZOS,color='#202020');canvas.paste(tile,(x,y+40))
 label=f"REAL frame {r['video_frame_index']} | {r['timestamp_sec']:.2f}s | approx {r['approx_azimuth_deg']:.1f} deg"
 if r['video_frame_index']==2570:label+=' | CLIPPED'
 draw.text((x+8,y+9),label,font=font,fill='white')
canvas.save(out/'real_front_candidates6.jpg',quality=96)
save(out/'real_front_candidates.json',dict(selected_frames=selected,
 angle_caution='Approximate angles from aligned independently estimated SfM cameras. These RGBs are original recorded frames, but are not same-camera GT for the virtual batch8 orbit.',
 crop_caution='Common observed3D surface projection only selects crop boxes; display tiles are individually fit for viewing. No generated or retouched pixels.'))
save(out/'sfm_coverage.json',dict(method='IndependentSfM centers aligned to Omega using six nearest-time reference camera-center pairs; approximate azimuth only, not exact per-frame calibration.',
 alignment=dict(scale=float(scale),rotation=rotation.tolist(),translation=translation.tolist(),rmse_relative_to_baseline=float(rmse)),links=links,frames=rows))
print('CROPS_READY',out/'real_front_candidates6.jpg')
