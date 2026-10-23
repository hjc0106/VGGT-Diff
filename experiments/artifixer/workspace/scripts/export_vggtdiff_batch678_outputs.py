"""Export each final batch separately, preserving full generated frames."""
from pathlib import Path
import json,hashlib,shutil,zipfile
from PIL import Image,ImageDraw,ImageFont
import cv2
import build_vggtdiff_weekly_20261023 as weekly


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    dest=weekly.OUT/'batch_outputs';dest.mkdir(exist_ok=True)
    font='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
    title=ImageFont.truetype(font,46);subtitle=ImageFont.truetype(font,29);label=ImageFont.truetype(font,30)
    runs=[(6,weekly.MATCH/'seed_20261028','六张真实参考；seed20261028；实际目标相机224.60°–263.91°'),
          (7,weekly.B7,'最终双侧参考版：第6批末三张 + 第8批前三张生成图；目标270°–309.375°'),
          (8,weekly.B8,'最终六真实参考版；目标315°–354.375°')]
    manifest={};lines=['# 第6、7、8批分别展示的生成结果','',
        '以下均为最终用于24图重建的VGGT-Diff生成RGB，不是3DGS渲染，也不是参考输入图。总览保留完整832×480画面，没有逐图裁剪、对齐或缩放；原始PNG直接复制保存。','']
    for batch,root,note in runs:
        result=json.loads((root/'result.json').read_text());frames=result['frames'];assert len(frames)==8
        raw=dest/f'batch{batch:02d}_raw';raw.mkdir(exist_ok=True)
        # Four columns and two rows; original pixels pasted at native resolution.
        canvas=Image.new('RGB',(3408,1264),'#F4F6F9');draw=ImageDraw.Draw(canvas)
        draw.text((24,16),f'第{batch}批：8张生成结果',font=title,fill='#203247')
        draw.text((24,80),note,font=subtitle,fill='#526779')
        saved=[]
        for i,row in enumerate(frames):
            source=Path(row['path']);assert digest(source)==row['sha256']
            im=Image.open(source).convert('RGB');assert im.size==(832,480)
            x=16+(i%4)*848;y=144+(i//4)*544
            draw.text((x+6,y),f"局部{i:02d} | {row['angle']:.3f}°",font=label,fill='#203247')
            canvas.paste(im,(x,y+44))
            output=raw/f'{i:02d}.png';shutil.copyfile(source,output);assert digest(output)==row['sha256']
            saved.append(dict(local_index=i,source_index=row['index'],angle=row['angle'],path=str(output),sha256=row['sha256']))
        draw.text((24,1220),'完整生成画面；同一显示尺度；各图角度由实际目标相机记录读取。',font=subtitle,fill='#526779')
        base=dest/f'batch{batch:02d}_generated8'
        canvas.save(base.with_suffix('.png'));canvas.save(base.with_suffix('.jpg'),quality=96,subsampling=0)
        video=dest/f'batch{batch:02d}_generated8.mp4';shutil.copyfile(root/'prediction_8.mp4',video)
        cap=cv2.VideoCapture(str(video));count=0
        while cap.read()[0]:count+=1
        cap.release();assert count==8
        manifest[str(batch)]=dict(source_run=str(root),seed=result['seed'],steps=result['steps'],frames=saved,
            contact_png=str(base.with_suffix('.png')),contact_jpg=str(base.with_suffix('.jpg')),video=str(video),video_frames=count)
        lines += [f'## 第{batch}批','',note,'',f'![第{batch}批八图](batch_outputs/{base.name}.jpg)','',
            f'- [视频](batch_outputs/{video.name})',f'- [原始PNG目录](batch_outputs/{raw.name})','']
    (dest/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    (weekly.OUT/'BATCH678_GENERATED_RESULTS.md').write_text('\n'.join(lines))
    main_manifest=weekly.OUT/'manifest.json';m=json.loads(main_manifest.read_text());m['batch_outputs']=manifest
    main_manifest.write_text(json.dumps(m,ensure_ascii=False,indent=2)+'\n')
    report=weekly.OUT/'WEEKLY_REPORT.md';text=report.read_text()
    addition='\n## 三批生成结果分别展示\n\n见[BATCH678_GENERATED_RESULTS.md](BATCH678_GENERATED_RESULTS.md)：每批独立八图总览、八帧视频及原始PNG；第7/8批均为最终采用的版本。\n'
    if '## 三批生成结果分别展示' not in text:report.write_text(text+addition)
    archive=weekly.OUT.with_suffix('.zip')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(weekly.OUT.rglob('*')):
            if p.is_file():z.write(p,p.relative_to(weekly.OUT.parent))
    print('EXPORTED',dest,'24 raw images, three eight-frame videos; archive updated')


if __name__=='__main__':main()
