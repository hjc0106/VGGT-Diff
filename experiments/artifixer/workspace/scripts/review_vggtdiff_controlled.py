"""Build auditable camera/noise-controlled comparison sheets."""
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageOps
import vggtdiff_controlled_context as run
from vggtdiff_white_car import save,digest


def main():
    out=run.OUT;results={name:run.read(out/name/'result.json') for name in run.CASES if (out/name/'result.json').exists()}
    repro=results['repro5'];canonical=repro['noise']['frame_hashes'];rows=[]
    for name,result in results.items():
        for key,value in result['noise']['frame_hashes'].items():
            if key in canonical:assert canonical[key]==value,f'Public noise mismatch: {name}/{key}'
        for f in result['frames']:assert digest(Path(f['path']))==f['sha256']
        t=run.read(out/name/'trajectory.json');base=run.read(run.PARENT/'trajectory.json')
        assert t['source_w2c']==base['source_w2c'] and t['source_intrinsics']==base['source_intrinsics']
        for f in result['frames']:
            a=f['angle']
            if a not in run.CASES['repro5']:continue
            i=run.CASES['repro5'].index(a)
            assert np.allclose(t['target_w2c'][f['index']],base['target_w2c'][i],rtol=0,atol=1e-12)
            assert t['target_intrinsics'][f['index']]==base['target_intrinsics'][i]
            parent=Path(repro['frames'][i]['path'])
            im=np.asarray(Image.open(f['path']).convert('RGB'),dtype=float)
            ref=np.asarray(Image.open(parent).convert('RGB'),dtype=float)
            rows.append(dict(case=name,angle=a,rgb_mae_vs_repro_01=float(np.abs(im-ref).mean()/255),noise_matched=True,camera_matched=True))
    cols=['Original 5','repro5','count4','context8'];canvas=Image.new('RGB',(416*4,288*5),'#181818');draw=ImageDraw.Draw(canvas)
    for row,a in enumerate(run.CASES['repro5']):
        for col,name in enumerate(cols):
            x=col*416;y=row*288
            if col==0:path=run.PARENT/'pilot/raw'/f'{row:02d}.png'
            else:path=next((Path(f['path']) for f in results.get(name,{}).get('frames',[]) if f['angle']==a),None)
            draw.text((x+8,y+7),f'{name}: {a:g} deg',fill='white')
            if path:
                with Image.open(path) as im:canvas.paste(ImageOps.pad(im.convert('RGB'),(416,240)),(x,y+48))
                if col>0:
                    m=next(r for r in rows if r['case']==name and r['angle']==a)
                    draw.text((x+8,y+27),f'Same-noise image change vs repro5: {m["rgb_mae_vs_repro_01"]:.4f}',fill='#cccccc')
            else:draw.text((x+20,y+150),'NOT GENERATED / NOT IN TARGET SET',fill='#aaaaaa')
    canvas.save(out/'controlled_comparison.jpg',quality=95)
    save(out/'comparison_checks.json',dict(completed_cases=list(results),common_camera_noise_verified=True,rows=rows,
        caution='RGB differences measure sensitivity to target context; they do not measure true image or geometry quality',reproduction_vs_original=repro['old_five_comparison']))
    print({name:[round(x['rgb_mae_vs_repro_01'],4) for x in rows if x['case']==name] for name in results})


if __name__=='__main__':main()
