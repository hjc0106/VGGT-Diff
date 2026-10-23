"""Compose audited real-video candidate frames and vehicle crop previews."""
import hashlib
import json
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

REPO=Path(__file__).resolve().parents[1]
VIDEO=REPO/'pipeline_exp/realworld/scene-1-nofull/scene-1-wcar.mp4'
OUT=REPO/'pipeline_exp/realworld/scene-1-nofull/06_worldact_white_car_v1/artifixer_episode_chain_64x4/vggtdiff_realvideo_refs_v1'
FRAMES=[0,240,480,720,960,1200,1320,1440]
# Coordinates measured on 960x540 overview images; multiply by four for 4K.
CROPS=[(245,200,455,325),(302,205,520,325),(306,213,548,332),
       (380,202,633,323),(448,204,722,332),(472,202,780,350),
       (435,190,830,390),(510,200,860,382)]


def timestamps(path):
    return [float(x) for x in re.findall(r'n:\s*\d+\s+pts:\s*\d+\s+pts_time:([0-9.]+)',path.read_text())]


def checksum(path):
    h=hashlib.sha256()
    with path.open('rb') as file:
        for block in iter(lambda:file.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    original_times=timestamps(OUT/'candidates_decode.log')
    extra=timestamps(OUT/'extra_candidate_decode.log')
    times=original_times[:6]+[extra[0],original_times[6]]
    assert len(times)==len(FRAMES)==8 and times==sorted(times)
    dest=OUT/'candidates/crops';dest.mkdir(exist_ok=True)
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',20)
    crop_sheet=Image.new('RGB',(1280,4*400),'#181818')
    full_sheet=Image.new('RGB',(1280,4*400),'#181818')
    records=[]
    for j,(frame,time,crop) in enumerate(zip(FRAMES,times,CROPS)):
        path=OUT/'candidates/full'/f'{j+1:02d}.png'
        box=tuple(v*4 for v in crop)
        with Image.open(path) as im:
            assert im.size==(3840,2160)
            full=im.convert('RGB');local=full.crop(box)
        local_path=dest/path.name;local.save(local_path)
        x=j%2*640;y=j//2*400
        label=f'{j+1:02d} | {time:.2f}s | video frame {frame}'
        for canvas,im in [(crop_sheet,local),(full_sheet,full)]:
            tile=ImageOps.pad(im,(640,360),method=Image.Resampling.LANCZOS,color='#101010')
            canvas.paste(tile,(x,y+40))
            ImageDraw.Draw(canvas).text((x+12,y+10),label,font=font,fill='white')
        records.append(dict(candidate=j+1,source_frame_zero_based=frame,pts_seconds=time,
            full_image=str(path.relative_to(OUT)),full_sha256=checksum(path),
            preview_crop_xyxy=list(box),crop_image=str(local_path.relative_to(OUT)),
            crop_sha256=checksum(local_path),crop_resolution=list(local.size)))
    crop_sheet.save(OUT/'candidates_vehicle_preview.jpg',quality=96)
    full_sheet.save(OUT/'candidates_full_preview.jpg',quality=94)
    (OUT/'candidates_manifest.json').write_text(json.dumps(dict(video=str(VIDEO),
        video_size_bytes=VIDEO.stat().st_size,original_resolution=[3840,2160],
        timestamp_basis='Decoded source PTS; zero-based frame indices, not assumed constant 60fps',
        processing='Lossless decoded full-frame PNGs; rectangular crop previews only, no synthesis or retouching',
        candidates=records,excluded=[dict(source_frame=1560,reason='Vehicle tail clipped at image right edge')]),indent=2)+'\n')
    print(json.dumps([dict(candidate=r['candidate'],frame=r['source_frame_zero_based'],time=r['pts_seconds']) for r in records]))


if __name__=='__main__':main()
