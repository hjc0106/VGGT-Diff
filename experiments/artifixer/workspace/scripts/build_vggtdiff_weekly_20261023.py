"""Build weekly-report figures from existing experiments; never run inference."""
from pathlib import Path
import hashlib,json,shutil,zipfile
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.backends.backend_pdf import PdfPages

REPO=Path(__file__).resolve().parents[1]
BASE=REPO/'pipeline_exp/realworld/scene-1-nofull/06_worldact_white_car_v1/artifixer_episode_chain_64x4'
OUT=REPO/'reports/vggtdiff_weekly_20261023'
MATCH=BASE/'crossview_geometry_matched_gt_v1'
B7=BASE/'vggtdiff_batch06_08gen_to_batch07_continuous_v1'
B8=BASE/'vggtdiff_batch08_real6_continuous_v1'
STUDY=BASE/'vggtdiff_batch678_regularization_ablation_v1'
GHOST=STUDY/'ghost_projection_diagnosis_v1'
font_manager.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
FAMILY=font_manager.FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc').get_name()
plt.rcParams.update({'font.family':FAMILY,'axes.unicode_minus':False,'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
BLUE,GREEN,ORANGE,GRAY='#3977B7','#299479','#D77D47','#89929C'
SOURCES=set();FIGURES=[]


def read(path):
    SOURCES.add(Path(path));return json.loads(Path(path).read_text())


def image(path,crop=None):
    path=Path(path);SOURCES.add(path);im=Image.open(path).convert('RGB')
    return im.crop(crop) if crop else im


def canvas(title,subtitle):
    fig=plt.figure(figsize=(16,9),facecolor='white')
    fig.text(.035,.95,title,fontsize=23,weight='bold',color='#203247')
    fig.text(.035,.905,subtitle,fontsize=11,color='#596779')
    return fig


def show(fig,rect,im,title):
    ax=fig.add_axes(rect);ax.imshow(im);ax.axis('off');ax.set_title(title,fontsize=12,pad=9)
    return ax


def bars(ax,labels,values,title,ylabel,colors=None,decimals=2):
    b=ax.bar(labels,values,color=colors or [GRAY,BLUE,GREEN,ORANGE][:len(values)],width=.6)
    ax.bar_label(b,fmt=f'%.{decimals}f',padding=4,fontsize=11)
    ax.set_title(title,fontsize=13,pad=15);ax.set_ylabel(ylabel);ax.set_ylim(0,max(values)*1.3)
    ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)


def finish(fig,name,caption,foot,pdf):
    fig.text(.035,.035,foot,fontsize=9,color='#627083')
    fig.savefig(OUT/'figures'/f'{name}.png',dpi=150,facecolor='white')
    fig.savefig(OUT/'figures'/f'{name}.pdf',facecolor='white')
    pdf.savefig(fig);plt.close(fig)
    FIGURES.append(dict(name=name,caption=caption))


def frame(root,local):
    return Path(read(root/'result.json')['frames'][local]['path'])


def main():
    (OUT/'figures').mkdir(parents=True,exist_ok=True)
    matched=read(MATCH/'comparison_summary.json')
    bridge=read(B7/'evaluation_bridge_v1/summary.json')
    three=read(B8/'evaluation_threeway_v1/summary.json')
    ablation=read(STUDY/'comparison/summary.json')
    ghost=read(GHOST/'summary.json')
    crop=(128,112,704,384)
    with PdfPages(OUT/'VGGT-Diff_本周实验图集.pdf') as pdf:
        # 1. Independent held-out real target images and all three generation seeds.
        fig=canvas('01  批内局部自洽，与真实观测仍有投影偏差','六张真实条件图 → 八个留出真实目标相机；三个种子；832×480，50步。示例均为同一目标相机。')
        target=read(MATCH/'targets.json')['frames'][6]
        paths=[Path(target['image'])]+[frame(MATCH/f'seed_{s}',6) for s in [20261026,20261027,20261028]]
        for i,(path,label) in enumerate(zip(paths,['留出真实图 GT','生成：种子26','生成：种子27','生成：种子28'])):
            show(fig,[.035+i*.242,.56,.225,.27],image(path,crop),label)
        seeds=list(matched['seeds'].values())
        ax=fig.add_axes([.075,.15,.38,.29])
        vals=[seeds[0]['paired']['primary14']['joint_median_gt_epi_px']]+[s['paired']['primary14']['joint_median_generated_epi_px'] for s in seeds]
        bars(ax,['GT','种子26','种子27','种子28'],vals,'内部14对近邻：极线误差','中位数 / px',decimals=3)
        ax=fig.add_axes([.57,.15,.38,.29])
        vals=[seeds[0]['paired']['reference_near']['joint_median_gt_epi_px']]+[s['paired']['reference_near']['joint_median_generated_epi_px'] for s in seeds]
        bars(ax,['GT','种子26','种子27','种子28'],vals,'与真实参考12对：极线误差','中位数 / px',decimals=3)
        fig.text(.04,.51,'同相机对应点偏差：4.67–6.00 px；掩膜 IoU：0.920–0.927。内部自洽不能替代真实几何准确性。',fontsize=12,color=BLUE)
        finish(fig,'01_real_gt_three_seeds','固定真实参考和目标相机，三种子生成的批内近邻误差较低，但与真实参考的极线误差及同相机位置偏差更大。',
            '白车区域 SIFT 双向匹配；无极线预筛。固定14/12对均有充分支持。相机为估计值；不同匹配点集不能解释为生成几何优于GT。',pdf)
        # 2. Camera continuity: identical display window, no object alignment.
        fig=canvas('02  统一目标相机轨迹，修正跨批距离跳变','第6批末端约263.91° → 第7批270°；以真实配准坐标系和末端相机为锚，延续半径、高度、朝向及内参。')
        old=BASE/'vggtdiff_batch06gen_to_batch07_v1';new=BASE/'vggtdiff_batch06gen_to_batch07_continuous_v1'
        source=frame(MATCH/'seed_20261028',7)
        for i,(p,label) in enumerate([(source,'第6批末端参考'),(frame(old,0),'第7批原近轨迹'),(frame(new,0),'第7批连续轨迹')]):
            show(fig,[.035+i*.325,.49,.305,.34],image(p,(0,80,832,440)),label)
        geometry=read(BASE/'vggtdiff_realvideo_unseen_right_near_v2/geometry_check.json');center=np.array(geometry['center'])
        trajectories=[read(MATCH/'trajectory.json'),read(old/'trajectory.json'),read(new/'trajectory.json')]
        ws=[trajectories[0]['target_w2c'][7],trajectories[1]['target_w2c'][0],trajectories[2]['target_w2c'][0]]
        ks=[trajectories[0]['target_intrinsics'][7],trajectories[1]['target_intrinsics'][0],trajectories[2]['target_intrinsics'][0]]
        distances=[float(np.linalg.norm(np.linalg.inv(w)[:3,3]-center)) for w in ws]
        bars(fig.add_axes([.08,.14,.36,.25]),['末端参考','原近轨迹','连续轨迹'],distances,'相机到车辆中心的距离','相对世界单位',colors=[GRAY,ORANGE,GREEN],decimals=3)
        bars(fig.add_axes([.57,.14,.36,.25]),['末端参考','原近轨迹','连续轨迹'],[k[0][0] for k in ks],'相机内参 fx','像素',colors=[GRAY,ORANGE,GREEN],decimals=1)
        finish(fig,'02_camera_continuity','将第7批目标相机从独立近轨迹改为第6批末端相机的刚性环绕延续，减轻跨批尺度跳变。',
            '图像使用相同像素裁剪，无逐车缩放或配准。相机连续不保证生成图几何正确；真实拍摄的距离变化本身也不等于重建错误。',pdf)
        # 3. Real-reference conditioning, same requested target cameras.
        fig=canvas('03  第8批真实参考锚定，改善车头外观与局部兼容性','三组目标相机完全一致：315°–354.375°；相同种子、50步；仅展示343.125°和354.375°。')
        roots=[BASE/'vggtdiff_batch06_07gen_to_batch08_continuous_v1',BASE/'vggtdiff_batch08_real_anchors_v1',B8]
        # Baseline paths come from archived manifests, not guessed run names.
        manifest=read(B8/'manifest.json');roots[0]=Path(manifest['baseline']);roots[1]=Path(manifest['mixed_control'])
        names=['六张生成参考','四真实 + 两生成','六张真实参考']
        for col,(root,label) in enumerate(zip(roots,names)):
            for ri,idx in enumerate([5,7]):show(fig,[.035+col*.325,.66-ri*.235,.305,.20],image(frame(root,idx),crop),label if ri==0 else '354.375°')
        vals=[three[k]['track_groups']['within_batch8']['cycle']['computed_inside_image_median_px'] for k in ['all_generated','real4_gen2','real6']]
        bars(fig.add_axes([.09,.13,.4,.21]),names,vals,'批内三视角留出误差','中位数 / px',colors=[GRAY,ORANGE,GREEN])
        fig.text(.58,.29,'六真实参考版本：',fontsize=16,color=GREEN,weight='bold')
        fig.text(.58,.23,'车头外观更稳定；三视角误差 1.01 px',fontsize=14)
        fig.text(.58,.17,'与真实参考的近邻支持图对：20 / 21',fontsize=14)
        fig.text(.58,.11,'仍需检查与上一批的边界兼容性',fontsize=12,color='#596779')
        finish(fig,'03_batch8_real_anchors','六真实参考生成第8批，车头外观稳定性和局部几何兼容性优于本次生成/混合参考版本。',
            '替换参考时源相机也变化，不是像素真实性的单变量因果实验。与条件图的兼容性不等于独立GT准确性；所有图为同像素裁剪。',pdf)
        # 4. Two-sided interpolation of batch7.
        fig=canvas('04  第6、8批双侧参考补第7批，改善跨批衔接','参考：第6批末三张 + 六真实条件生成的第8批前三张；第7批目标相机保持不变。')
        old7=BASE/'vggtdiff_batch06gen_to_batch07_continuous_v1'
        for ri,root in enumerate([old7,B7]):
            for ci,(idx,label) in enumerate([(6,'第7批303.75°'),(7,'第7批309.375°'),(0,'第8批315°（固定）')]):
                p=frame(root,idx) if ci<2 else frame(B8,0)
                show(fig,[.035+ci*.325,.66-ri*.235,.305,.20],image(p,crop),('旧：单侧参考 / ' if ri==0 else '新：双侧参考 / ')+label)
        ax=fig.add_axes([.09,.14,.5,.2]);x=np.arange(3);width=.32
        categories=['第6→7边界','第7批内','第7→8边界'];keys=['left_boundary','within7','right_boundary']
        for delta,key,color,label in [(-width/2,'old7',GRAY,'旧第7批'),(width/2,'new7',GREEN,'双侧参考第7批')]:
            values=[bridge[key]['track_groups'][g]['cycle']['computed_inside_image_median_px'] for g in keys]
            b=ax.bar(x+delta,values,width,color=color,label=label);ax.bar_label(b,fmt='%.2f',padding=3,fontsize=10)
        ax.set_xticks(x,categories);ax.set_ylabel('三视角留出中位 / px');ax.set_ylim(0,14.5);ax.legend(frameon=False);ax.grid(axis='y',alpha=.15)
        fig.text(.67,.27,'第7→8边界',fontsize=16,color=GREEN,weight='bold')
        fig.text(.67,.20,'11.87 → 1.38 px',fontsize=22,color=GREEN)
        fig.text(.67,.13,'充分支持图对：0/3 → 3/3',fontsize=13)
        finish(fig,'04_batch7_two_sided_bridge','双侧参考显著改善第7→8批近邻衔接；第7批内和两侧边界的三视角误差处于约1.4–1.7px量级。',
            '近邻检查涉及条件图；不同版本的匹配点集与轨迹数不同，不能解释为独立GT精度或全范围几何正确。',pdf)
        # 5. Fixed-input reconstruction ablations.
        fig=canvas('05  固定24图重建：轮廓约束有效，针状约束收益有限','相同图像/掩膜/相机/初始化/采样；每组15000步、每图625次更新；展示354.375°。')
        models=[Path(ablation_path) for ablation_path in [str(BASE/'vggtdiff_batch06_07bridge_08_reconstruction_v1'),str(STUDY/'outline'),str(STUDY/'needle')]]
        rows=read(models[0]/'inputs.json')['frames'];r=rows[-1]
        target=np.asarray(image(r['image']));mask=np.asarray(Image.open(r['mask']))>127;SOURCES.add(Path(r['mask']))
        images=[Image.fromarray(target*mask[:,:,None]).crop((144,112,656,344))]
        images += [image(root/'evaluation/training_comparison/023.png',(832+144,32+112,832+656,32+344)) for root in models]
        for ci,(im,label) in enumerate(zip(images,['生成监督图','A 原基线','B 轮廓约束','C 针状约束'])):show(fig,[.035+ci*.242,.53,.225,.3],im,label)
        metrics=ablation['metrics'];labels=['原基线','轮廓约束','针状约束'];keys=['A_baseline','B_outline','C_needle']
        bars(fig.add_axes([.075,.17,.38,.25]),labels,[metrics[k]['endpoint_outside05']*100 for k in keys],'末端轮廓外面积 / 车辆面积','百分比，alpha > 0.5',colors=[GRAY,GREEN,ORANGE])
        bars(fig.add_axes([.57,.17,.38,.25]),labels,[metrics[k]['mean_psnr'] for k in keys],'24图平均前景PSNR','dB',colors=[GRAY,GREEN,ORANGE])
        fig.text(.045,.49,'轮廓约束：末端外面积减少32.7%，但仍有拖影；针状高斯形状改善未转化为明显拖影改善。',fontsize=12,color=BLUE)
        finish(fig,'05_reconstruction_regularization','固定24张生成监督的消融显示：轮廓约束降低外部拖影，但没有消除问题；针状约束独立收益很小。',
            '面积以车辆mask为分母；PSNR针对生成监督，不是独立真实GT。三批覆盖约224.6°–354.4°，360°未覆盖侧仍严重破碎。',pdf)
        # 6. Attribution and counterfactual removal.
        fig=canvas('06  拖影归因：责任高斯也参与正常车体拟合','追踪末端348.75°/354.375°拖影，在24个监督相机下累计真实遮挡后的高斯贡献，并临时移除验证。')
        original=image(GHOST/'training_panels/23.jpg')
        for ci,(source_col,label) in enumerate([(1,'B 原模型'),(2,'责任高斯高亮'),(3,'临时移除责任高斯')]):
            im=original.crop((source_col*416+72,84,source_col*416+328,200))
            show(fig,[.035+ci*.325,.58,.305,.23],im,label)
        fig.text(.04,.49,'3270 个高斯（模型3.7%）解释两末端约90%的轮廓外可见alpha贡献；不能据此直接删除。',fontsize=12,color=BLUE)
        contributions=read(GHOST/'group_view_contributions.json');ax=fig.add_axes([.08,.15,.45,.25])
        for group,label,color in [('upper_outline','上部',BLUE),('middle_outline','中部',ORANGE),('lower_outline','下部',GREEN)]:
            v=[r for r in contributions if r['group']==group];ax.plot([r['angle'] for r in v],[r['outside_fraction']*100 for r in v],label=label,color=color,lw=2)
        ax.set_xlabel('相机方位角 / °');ax.set_ylabel('该组可见贡献位于轮廓外 / %');ax.legend(frameon=False,ncol=3);ax.grid(alpha=.18)
        ax=fig.add_axes([.65,.15,.28,.25]);bars(ax,['原B','临时移除'],[ghost['original']['foreground_psnr'],ghost['all_selected']['foreground_psnr']],'整批移除损伤前景拟合','24图平均PSNR / dB',colors=[BLUE,ORANGE])
        finish(fig,'06_ghost_attribution','责任高斯在第6/7批主要贡献车体内部，到正面末端越出轮廓；整批移除减轻拖影，却明显损伤正常拟合。',
            '临时移除：末端外面积10.91%→5.51%，24图PSNR 32.94→26.25dB。确认模型投影冲突，尚不能区分图像、相机和几何误差来源。',pdf)
        # 7. All final generated views, frozen masks for presentation only.
        fig=canvas('07  第6、7、8批：当前24张生成监督图','第6批：六真实参考；第8批：六真实参考；第7批：第6/8批双侧生成参考。全部图像为生成结果。')
        for ri,batch in enumerate([6,7,8]):
            for ci,row in enumerate(rows[ri*8:(ri+1)*8]):
                rgb=np.asarray(image(row['image']));mask=np.asarray(Image.open(row['mask']))>127;SOURCES.add(Path(row['mask']))
                im=Image.fromarray(rgb*mask[:,:,None]).crop((96,112,768,384))
                show(fig,[.035+ci*.12,.65-ri*.25,.113,.18],im,f'B{batch} | {row["angle"]:.1f}°')
        finish(fig,'07_generated_24_views','当前第6、7、8批共24张生成监督图，展示由真实锚定与双侧补间形成的局部连续视角范围。',
            '背景仅按已审阅mask置黑；所有图使用同像素裁剪，无逐车配准或尺度拟合。外观连续不等于真实几何正确。',pdf)
        import sys
        from build_vggtdiff_batch678_setup_assets import add_figures
        add_figures(sys.modules[__name__],pdf)
    # Include short native videos, keeping a self-contained report package.
    videos={
        '01_gt_vs_three_seeds.mp4':MATCH/'gt_vs_three_seeds.mp4',
        '03_batch8_three_reference_sets.mp4':B8/'evaluation_threeway_v1/comparison_threeway_8.mp4',
        '04_batch7_bridge_old_new.mp4':B7/'evaluation_bridge_v1/bridge_old_new_sequence.mp4',
        '05_reconstruction_ablation.mp4':STUDY/'comparison/supervised_arc_comparison.mp4',
        '06_ghost_removal.mp4':GHOST/'supervised_arc_removal_comparison.mp4'}
    (OUT/'videos').mkdir(exist_ok=True)
    for name,source in videos.items():SOURCES.add(source);shutil.copyfile(source,OUT/'videos'/name)
    # Thumbnail index, source-resolution figures remain separate.
    overview=Image.new('RGB',(1600,1350),'#E8EDF2')
    for i,r in enumerate([FIGURES[j] for j in [0,2,3,5,7,8]]):
        im=Image.open(OUT/'figures'/f'{r["name"]}.png').resize((800,450));overview.paste(im,(800*(i%2),450*(i//2)))
    overview.save(OUT/'overview.jpg',quality=94)
    captions=['# 周报配图说明','']
    for i,r in enumerate(FIGURES,1):captions += [f'## 图{i}：{r["name"]}',r['caption'],f'![图{i}](figures/{r["name"]}.png)','']
    (OUT/'FIGURE_CAPTIONS.md').write_text('\n'.join(captions))
    write_report()
    # Verify every exported image and video before packaging.
    import cv2
    video_checks={}
    for name in videos:
        cap=cv2.VideoCapture(str(OUT/'videos'/name));n=0
        while cap.read()[0]:n+=1
        cap.release();assert n>0;video_checks[name]=n
    assert list(video_checks.values())==[8,8,12,128,128],video_checks
    for r in FIGURES:
        with Image.open(OUT/'figures'/f'{r["name"]}.png') as im:assert im.size==(2400,1350);im.verify()
    manifest=dict(period='2026-10-19—2026-10-23',scope='This week experiment records; no new inference/training',figures=FIGURES,video_frames=video_checks,
        sources=[dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(SOURCES)],
        caution='Single white-car scene; estimated cameras; conditional references are not independentGT; diagnostics are exploratory')
    (OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    archive=OUT.with_suffix('.zip')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(OUT.rglob('*')):
            if p.is_file():z.write(p,p.relative_to(OUT.parent))
    print('READY',OUT,archive,'figures',len(FIGURES),'videos',video_checks,flush=True)


def write_report():
    text='''# VGGT-Diff 本周工作总结

时间：2026年10月19日—10月23日。按本周会话涉及的实验与归档结果整理。

## 可直接放入周报的摘要

本周围绕VGGT-Diff多视角扩展及其对3DGS重建的影响，完成真实图像几何评估基线、三随机种子同相机生成对照、第6/7/8批连续轨迹与参考策略实验，以及24张生成监督的重建和拖影归因。结果表明：生成图可以具有较好的批内局部自洽性，但仍与真实观测存在投影偏差；统一相机轨迹、真实参考锚定和双侧补间可改善跨批衔接。双侧参考将第7→8批三视角误差中位数从11.87px降至1.38px，充分支持近邻图对从0/3增至3/3。重建侧轮廓约束将末端轮廓外面积由16.21%降至10.91%，但仍有残留拖影。进一步定位到3270个责任高斯，它们同时参与正常车体拟合，整批移除会使24图平均PSNR从32.94降至26.25dB。因此，下一阶段转向真实覆盖区验证、统一相机规划和带重叠检查的小步视角扩展。

## 本周完成

1. **建立几何一致性评估。** 在白车掩膜内提取SIFT双向匹配，采用极线误差、描述子闭环三视角留出重投影误差、同相机对应点偏移与轮廓差异。先用真实图和已知扰动验证敏感性；匹配不足保留为证据不足，不以误差预筛制造低误差。
2. **完成三种子真实GT对照。** 固定六张真实条件图及八个留出真实相机。内部14对近邻极线误差中位为0.416–0.643px，GT为0.351px；与真实参考12对则为1.819–2.872px，GT为0.376px。同相机对应点偏移4.67–6.00px、mask IoU约0.920–0.927。支持“局部内部自洽与真实投影偏差同时存在”。见图1。
3. **统一第6→7批目标相机。** 从第6批末端实际相机刚性延续位置与姿态，保持半径、高度及K，修正原近轨迹带来的尺度跳变。视距为共同坐标系下的相对尺度，非米。见图2。
4. **比较第8批参考策略。** 六生成、四真实+两生成、六真实三组目标相机和采样参数一致。六真实组保持较稳定本田车头，三视角中位1.01px，前两组为2.10/4.93px；与真实条件图近邻支持20/21。源相机随参考替换而变化，因此不作纯真实性的单变量归因。见图3。
5. **双侧补间重生成第7批。** 取第6批末三张和真实条件生成的第8批前三张为参考，目标相机不变。第7→8边界三视角误差11.87→1.38px；第7批内1.83→1.41px；第6→7边界1.68→1.70px，不能表述为所有指标都改善。见图4、图7。
6. **完成24图重建与正则消融。** 全部监督RGB均为生成图，相机固定，初始化含真实几何先验。轮廓约束使24图平均轮廓外面积6.33%→3.53%，末端16.21%→10.91%，平均PSNR降低0.43dB。针状项改变高斯形状但末端外面积仅降到16.16%，按预设条件未跑组合组。见图5。
7. **定位残留拖影责任高斯。** 3270/87770个高斯解释两末端约90%的轮廓外可见alpha贡献。同一高斯组在第6/7批主要位于车内，到末端大量越出轮廓。临时移除后末端外面积10.91%→5.51%，平均PSNR 32.94→26.25dB、缺失0.549%→1.111%，不宜直接整批删除。模型未改写，五轨迹诊断视频完整。见图6。

## 当前结论

- **批内：** 可靠参考和连续相机条件下，近邻局部兼容性较好；低极线误差不等于深度、遮挡或全局形状准确。
- **批间：** 单向接力对参考质量敏感，错误外观和相机配置可能沿链传播；真实锚定和双侧参考能够改善本场景近邻衔接。
- **重建：** 几何指标改善后仍可能拉丝；当前确认的是模型的投影冲突，尚未分离生成图、掩膜、相机或错误空间分布的因果作用。

## 下阶段计划（尚未执行）

重建和正则实验先停在当前结果。扩图采用“真实参考→真实覆盖区留出验证→覆盖边缘小步扩展→带重叠检查的接力→有两端时双侧补间”。相机先在统一坐标系规划，再检查生成图是否遵守目标；不根据生成图看起来变大就直接重定下一批视距。分别评估批内、跨批边界、重叠相机复现及与真实GT兼容性，报告三个种子与匹配支持量。

## 使用这些材料

- `figures/`：9张2400×1350 PNG，每张附PDF；均为16:9，适合周报/幻灯片。
- `VGGT-Diff_本周实验图集.pdf`：9页独立图集。
- `BATCH678_SETUP.md`：三批的完整参考帧、精确目标角度、相机视距与最终生成依赖。
- `FIGURE_CAPTIONS.md`：每张图的一句话说明及预览。
- `videos/`：5段原实验对比视频，未新生成实验图像。
- `manifest.json`：源数据绝对路径、哈希和视频帧数。

### 展示顺序建议

短版用图1、图3、图4、图6；重建进展补图5；附录放图2、图7。若周报只允许两张图，优先图4（跨批改进）和图6（问题定位）。

## 证据边界

仅一个白车场景；全部相机为估计值。真实完整车辆样本已确认至少覆盖约207°–337°；当前24张生成监督覆盖约224.6°–354.4°。近邻条件图兼容性不是独立GT；三视角不同版本对应点集合不同，不能报告匹配点级显著性。重建面积以mask为分母，PSNR针对生成训练图。原图采用固定像素裁剪、统一显示缩放，没有逐车配准或形变；图7按mask置黑仅用于观看。未覆盖侧的360°破碎不能被表述为已解决。
'''
    # Add clickable absolute links to primary reports for review outside this bundle.
    text+='\n## 三批实验设置与视角关系补充\n\n第6批使用真实留出图对应的非等间隔相机，实际覆盖224.598°–263.911°；第7批270°–309.375°与第8批315°–354.375°以5.625°间隔共同延续第6批末端相机，视距固定0.624325相对单位。最终生成依赖为“真实参考A→B6；真实参考B→B8；B6末三张+B8前三张→B7”，并非最终仍是单向6→7→8。详见[BATCH678_SETUP.md](BATCH678_SETUP.md)及图8、图9。\n'
    text+='\n## 原始报告\n\n'
    for label,p in [('真实GT三种子对照',MATCH/'REPORT.md'),('第8批参考对照',B8/'evaluation_threeway_v1/REPORT.md'),('第7批双侧补间',B7/'evaluation_bridge_v1/REPORT.md'),('重建消融',STUDY/'comparison/REPORT.md'),('拖影归因',GHOST/'REPORT.md')]:
        SOURCES.add(p);text+=f'- [{label}]({p})\n'
    (OUT/'WEEKLY_REPORT.md').write_text(text)


if __name__=='__main__':main()
