"""Audit and package the mixed15 -> ArtiFixer 2D outputs."""
import subprocess
import numpy as np
from PIL import Image,ImageDraw
from vggtdiff_car15_artifixer import ROOT,PREP,SCENE,REFS
from vggtdiff_car15_prepare import OUT,read,e
from vggtdiff_white_car import save,digest

def main():
    batches=list((ROOT/'results').rglob('frames/batch_0000'));assert len(batches)==1,batches
    batch=batches[0];names=[f'{i:05d}.png' for i in range(64)];review=ROOT/'review';review.mkdir(exist_ok=True)
    for kind in ['pred','pred_composited','rendered']:
        actual=sorted(p.name for p in (batch/kind).glob('*.png'));assert actual==names,(kind,len(actual))
    manifest=read(ROOT/'manifest.json');assert digest(OUT/'mixed15.ply')==manifest['input_sha256']
    rows=[]
    comparison=review/'comparison_frames';comparison.mkdir(exist_ok=True)
    for i,name in enumerate(names):
        a=np.array(Image.open(PREP/'images'/name).convert('RGB'));b=np.array(Image.open(batch/'pred_composited'/name).convert('RGB'));raw=np.array(Image.open(batch/'pred'/name).convert('RGB'))
        assert a.shape==b.shape==raw.shape==(480,832,3),(name,a.shape,b.shape,raw.shape)
        roi=np.load(PREP/'full_tank_roi'/f'{i:05d}.npy')>.01
        low=np.load(PREP/'texture_disorder_v6a_inputs/region_low_precision'/f'{i:05d}.npy');missing=np.load(PREP/'texture_disorder_v6a_inputs/region_missing'/f'{i:05d}.npy')
        delta=abs(a.astype(float)-b.astype(float))/255
        rows.append(dict(index=i,angle=i*5.625,roi_change_mae=float(delta[roi].mean()),background_change_mae=float(delta[~roi].mean()),raw_change_mae=float(abs(a.astype(float)-raw.astype(float)).mean()/255),low_precision_mean=float(low.mean()),missing_mean=float(missing.mean())))
        im=Image.new('RGB',(1664,512));im.paste(Image.fromarray(a),(0,32));im.paste(Image.fromarray(b),(832,32));d=ImageDraw.Draw(im);d.text((10,10),f'Before: mixed15 3DGS | {i*5.625:.3f} deg',fill='white');d.text((842,10),'After: ArtiFixer V6-A (composited)',fill='white');im.save(comparison/name)
    def encode(src,dest):
        subprocess.run(['ffmpeg','-y','-framerate','8','-i',str(src/'%05d.png'),'-c:v','libx264','-crf','18','-pix_fmt','yuv420p',str(dest)],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    encode(comparison,review/'before_after_orbit64.mp4');encode(batch/'pred_composited',review/'repaired_composited_orbit64.mp4');encode(batch/'pred',review/'repaired_raw_orbit64.mp4')
    indices=list(range(0,64,8));e.contact([comparison/names[i] for i in indices],[f'{i}: {i*5.625:.0f} deg' for i in indices],review/'keyframes.jpg',2)
    e.contact([comparison/names[i] for i in range(64)],[f'{i}: {i*5.625:.1f} deg' for i in range(64)],review/'all64.jpg',4)
    e.contact([PREP/'images'/names[i] for i in REFS],[f'ref {i}: {i*5.625:.0f} deg' for i in REFS],review/'references.jpg',3)
    save(review/'audit.json',dict(status='outputs complete; visual review pending',input_ply_unchanged=True,frames=64,width=832,height=480,reference_indices=REFS,batch=str(batch),frame_statistics=rows,mean_roi_change_mae=float(np.mean([r['roi_change_mae'] for r in rows]))))
    manifest['status']='inference completed; visual review pending';manifest['batch_dir']=str(batch);save(ROOT/'manifest.json',manifest)
    print('REVIEW_PREPARED',review,flush=True)
if __name__=='__main__':main()
