"""Preserve native/composited distinctions and finish the inference audit."""
import shutil,subprocess
from pathlib import Path
from PIL import Image,ImageDraw
from vggtdiff_car15_artifixer import ROOT,PREP
from vggtdiff_car15_prepare import read,e
from vggtdiff_white_car import save,digest

def main():
    audit=read(ROOT/'review/audit.json');batch=Path(audit['batch']);review=ROOT/'review';rawcompare=review/'raw_comparison_frames';rawcompare.mkdir(exist_ok=True)
    for i in range(64):
        name=f'{i:05d}.png';im=Image.new('RGB',(1664,512));im.paste(Image.open(PREP/'images'/name),(0,32));im.paste(Image.open(batch/'pred'/name),(832,32));d=ImageDraw.Draw(im);d.text((8,10),f'Before: mixed15 3DGS | {i*5.625:.3f} deg',fill='white');d.text((840,10),'After: ArtiFixer raw prediction',fill='white');im.save(rawcompare/name)
    subprocess.run(['ffmpeg','-y','-framerate','8','-i',str(rawcompare/'%05d.png'),'-c:v','libx264','-crf','18','-pix_fmt','yuv420p',str(review/'before_after_raw_orbit64.mp4')],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    indices=list(range(0,64,8));e.contact([rawcompare/f'{i:05d}.png' for i in indices],[f'{i}: {i*5.625:.0f} deg' for i in indices],review/'raw_keyframes.jpg',2)
    e.contact([rawcompare/f'{i:05d}.png' for i in range(64)],[f'{i}: {i*5.625:.1f} deg' for i in range(64)],review/'raw_all64.jpg',4)
    frozen=read(ROOT/'frozen_inputs.json');checks=[dict(path=x['path'],matches=digest(Path(x['path']))==x['sha256']) for x in frozen];assert all(x['matches'] for x in checks)
    video_checks=[]
    for p in sorted(review.glob('*.mp4')):
        data=__import__('json').loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-select_streams','v:0','-show_entries','stream=width,height,nb_read_frames,r_frame_rate','-show_entries','format=duration','-of','json',str(p)]))
        assert data['streams'][0]['nb_read_frames']=='64';video_checks.append(dict(path=str(p),**data))
    (ROOT/'logs').mkdir(exist_ok=True)
    for stage in ['prepare','analyze','infer','review']:shutil.copyfile(f'/tmp/car15_artifixer_{stage}.log',ROOT/'logs'/f'{stage}.log')
    audit.update(status='completed and visually reviewed',frozen_input_checks=checks,videos=video_checks,visual_review=['64-frame composited contact sheet','raw and composited front/side/rear views','8 key angles'],quality_note='Raw rear cleanup is stronger; front fragmentation remains. Compositing restores some original artifact pixels classified as high precision. No hidden-side GT accuracy claim.')
    save(review/'audit.json',audit)
    m=read(ROOT/'manifest.json');m['status']='completed and visually reviewed';m['sampling_seconds']=159.68;m['processing_seconds_excluding_weight_load']=246.57;m['outputs']={'raw_pred_dir':str(batch/'pred'),'composited_dir':str(batch/'pred_composited'),'raw_comparison_video':str(review/'before_after_raw_orbit64.mp4'),'composited_comparison_video':str(review/'before_after_orbit64.mp4')};m['finalizer_sha256']=digest(Path(__file__));save(ROOT/'manifest.json',m)
    shutil.copyfile(Path(__file__),ROOT/'implementation'/Path(__file__).name)
    p=ROOT/'REPORT.md';text=p.read_text().replace('状态：推理进行中，完成后补充检查结论。','''## 完成状态与结果

推理正常完成，64 张 pred、64 张 pred_composited、64 张输入 rendered 均齐全，全部 832×480。四步采样约 159.68 秒，含预处理、解码和图像导出约 246.57 秒；另有网络盘读取权重的等待。输入 PLY 与 269 项冻结输入/脚本的 SHA256 检查通过，64 帧相机恢复误差小于 1e-15。四个交付视频均解码核对为 64 帧、8 fps、8 秒。

重点查看 `review/before_after_raw_orbit64.mp4`（左输入，右原始修复预测）与 `review/repaired_raw_orbit64.mp4`。合成输出对比另见 `review/before_after_orbit64.mp4`，单独合成输出见 `review/repaired_composited_orbit64.mp4`。`review/raw_keyframes.jpg` 和 `review/keyframes.jpg` 分别为两种输出的关键视角。

已检查全环绕总览及正前、右侧、正后等关键角度。原始 pred 在车尾和部分轮廓处清理较明显，正后图比输入更连续；但正前中央、引擎盖和风挡的严重破碎仍被保留，仍有模糊、扭曲和残留漂浮纹理。车标、车牌等细节也可能被改写，不能认为已保持精确车辆身份或实现完整修复。

pred_composited 为低精度/缺失区域的局部合成，高精度区域回填原渲染。因此当损坏纹理被当前控制图分入高精度区时，原始 pred 的部分清理会在合成结果中被撤回；正后视角可明显看到这一现象。两种输出必须分开评价，合成输出改动少不能等同于模型没有运行。

合成输出在对象 ROI 内平均像素变化 MAE 约 0.02350（0–1 范围），ROI 外最大逐帧平均变化约 2.13e-7，说明背景保持有效。这是变化量，不是修复准确率或质量提升指标。

本次流程已跑通，但修复效果有限，尚未得到干净且完整的全部视角。基于现有掩码和输出，后续可检验正前/正后的坏纹理是否误判为高精度，再做有针对性的编辑区域实验；当前没有执行额外变体或三维蒸馏。
''');p.write_text(text)
    print('FINALIZED',ROOT,flush=True)
if __name__=='__main__':main()
