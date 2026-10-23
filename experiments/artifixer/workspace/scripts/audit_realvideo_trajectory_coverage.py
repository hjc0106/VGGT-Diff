"""Summarize visibility, azimuth and range of the already aligned real SfM path."""
from pathlib import Path
import json
import shutil
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from vggtdiff_white_car import BASE,REPO,save,digest

SOURCE=BASE/'realvideo_batch08_coverage_v1'
OUT=SOURCE/'trajectory_audit'


def main():
    source=json.loads((SOURCE/'sfm_coverage.json').read_text());g=json.loads((BASE/'vggtdiff_realvideo_unseen_right_near_v2/geometry_check.json').read_text())
    center=np.array(g['center']);up=np.array(g['up']);up/=np.linalg.norm(up)
    rows=[]
    # Visibility is reviewed at these registered samples, not inferred just from position angle.
    for r in source['frames']:
        f=r['video_frame_index'];status='full_car' if f<=1481 or f>=2630 else ('partial_car' if f<=1844 or f==2570 else 'car_out_of_view')
        delta=np.array(r['omega_center'])-center;height=float(delta@up)
        rows.append(dict(frame=f,time_sec=r['timestamp_sec'],azimuth_deg=r['approx_azimuth_deg'],
            distance=float(np.linalg.norm(delta)),horizontal_radius=float(np.linalg.norm(delta-height*up)),height=height,
            visibility=status,center_world=r['omega_center'],image_path=r['image_path']))
    full=[r for r in rows if r['visibility']=='full_car'];first=[r for r in full if r['frame']<=1481];last=[r for r in full if r['frame']>=2630]
    def stats(values):
        return dict(samples=len(values),frame_range=[values[0]['frame'],values[-1]['frame']],
            angle_range_deg=[min(r['azimuth_deg'] for r in values),max(r['azimuth_deg'] for r in values)],
            distance_range=[min(r['distance'] for r in values),max(r['distance'] for r in values)],
            height_range=[min(r['height'] for r in values),max(r['height'] for r in values)])
    positions=np.array([r['center_world'] for r in rows]);steps=np.linalg.norm(np.diff(positions,axis=0),axis=1)
    summary=dict(method='53 registered real SfM camera centers aligned to the Omega coordinate frame; visual full/partial/absent labels at registered samples.',
        first_full_segment=stats(first),late_full_segment=stats(last),all_full_samples=stats(full),
        source_frame0_reference_angle_deg=321.888390327335,
        all_camera_distance_range=[min(r['distance'] for r in rows),max(r['distance'] for r in rows)],
        per_approximately_1s_center_steps=dict(median=float(np.median(steps)),maximum=float(steps.max()),min=float(steps.min())),
        alignment_rmse_relative_to_baseline=source['alignment']['rmse_relative_to_baseline'],
        limitations=['Positions/angles/ranges are estimated, not calibrated ground truth; units are relative, not meters.',
            'Visibility labels describe manually reviewed registered samples and approximate time ranges, not an exhaustive pixelwise label for3207 frames.',
            'Vehicle-center approximation has larger angular sensitivity when camera passes close by; position azimuth does not imply vehicle visible.',
            'The SfM model uses one shared OPENCV intrinsic calibration; that assumption alone cannot prove the physical focal length never changes.'],frames=rows)
    save(OUT/'summary.json',summary)
    times=np.array([r['time_sec'] for r in rows]);az=np.rad2deg(np.unwrap(np.deg2rad([r['azimuth_deg'] for r in rows])));ranges=np.array([r['distance'] for r in rows])
    colors={'full_car':'#16723e','partial_car':'#df8b19','car_out_of_view':'#a5a5a5'}
    labels={'full_car':'Full car visible (reviewed samples)','partial_car':'Car clipped / partial','car_out_of_view':'Car outside the image'}
    fig,(ax1,ax2,ax3)=plt.subplots(3,1,figsize=(12,12))
    ax1.plot(times,az,color='#555555',linewidth=1)
    ax2.plot(times,ranges,color='#555555',linewidth=1)
    for status,color in colors.items():
        mask=np.array([r['visibility']==status for r in rows]);ax1.scatter(times[mask],az[mask],s=25,color=color,label=labels[status]);ax2.scatter(times[mask],ranges[mask],s=25,color=color)
    ax1.axhspan(315,354.375,color='#5577dd',alpha=.08,label='Batch8 requested azimuth315..354.375')
    ax1.set_ylabel('Azimuth (deg, unwrapped)');ax1.set_xlabel('Video time (s)');ax1.set_title('Real camera position angle and actual vehicle visibility');ax1.legend(fontsize=9,loc='upper left');ax1.grid(alpha=.2)
    ax2.axhline(.6243251548365562,color='#4776cf',linestyle='--',label='Current generated orbit: constant center range0.6243')
    ax2.set_ylabel('Range to estimated vehicle center\n(relative world units)');ax2.set_xlabel('Video time (s)');ax2.set_title('Real range varies continuously; gray intervals have no visible car');ax2.legend(fontsize=9);ax2.grid(alpha=.2)
    delta=positions-center;front=delta@np.array(g['front']);left=delta@np.array(g['left'])
    ax3.plot(front,left,color='#555555',linewidth=1)
    for status,color in colors.items():
        mask=np.array([r['visibility']==status for r in rows]);ax3.scatter(front[mask],left[mask],s=25,color=color)
    ax3.scatter([0],[0],c='black',marker='x',s=70,label='Estimated car center')
    th=np.linspace(0,2*np.pi,256);ax3.plot(.6201062213721221*np.cos(th),.6201062213721221*np.sin(th),'--',color='#4776cf',alpha=.6,label='Generated orbit radius0.6201')
    for i in [0,24,30,34,42,43,52]:ax3.annotate(f'{times[i]:.1f}s',(front[i],left[i]),xytext=(4,5),textcoords='offset points',fontsize=8)
    ax3.set_xlabel('Front direction (relative units)');ax3.set_ylabel('Left direction (relative units)');ax3.set_title('Camera center path, vehicle coordinate top view');ax3.set_aspect('equal',adjustable='datalim');ax3.legend(fontsize=9);ax3.grid(alpha=.2)
    fig.tight_layout();fig.savefig(OUT/'trajectory_coverage.png',dpi=150);plt.close(fig)
    (OUT/'REPORT.md').write_text('''# 真实视频视角、可见性与距离审计

实际拍摄不是固定半径圆轨迹，也不是一整圈持续对准车辆。当前已确认完整车辆样本的方位角至少覆盖约207°–337°，其余方向不能视作已有完整真实监督。

| 视频区间（约） | 白车可见性 | 相机位置方位角（约） | 到车辆中心距离（相对单位） |
|---|---|---|---|
| 0–24.5s | 完整为主 | 322°逐步降到207° | 0.70逐步降到0.38 |
| 25.5–30.5s | 车辆被裁切，逐渐离开画面 | 接近后方，约200°→187°，随后转为212° | 0.33降至0.08 |
| 31.5–41.5s | 未拍到目标白车 | 快速经过前方，约300°→0°附近→346° | 最近约0.073，随后增至0.57 |
| 42.5s附近 | 车辆重新进入，车头被裁切 | 约342° | 约0.61 |
| 43.5–53s | 完整车辆重新可用 | 约337°降到301° | 约0.63–0.68 |

这些时间和角度是已检查约1秒间隔注册样本的近似范围，不是逐帧精确边界。最初0帧的Omega估计角度321.89°；53帧SfM样本从0.50秒开始，因此首样本320.88°略有差别。

完整监督的主要缺口：0°–180°、接近正后方180°–约207°以及约342°–360°缺少完整白车图；315°–337°确实存在真实车头画面。对应既有八批定义，第6、第7批角度范围有真实覆盖，第5批后段有覆盖，第8批前段有覆盖；第1–4批没有找到完整真实车身覆盖。不能用相机走过某个角度代替该角度的车辆成像证据。

存在明显距离变化，但现有约1秒采样轨迹未显示离群的空间瞬移：相邻中心位移中位0.0570、最大0.0631相对单位。约30–32秒角度快速变化与相机接近车辆中心投影、从车旁通过有关；方位角359°→0°还包含角度记法的环绕。不能把这些角度变化直接解释为视频剪切或相机瞬移。

原视频的SfM采用同一套OPENCV内参；此前832×480参考图不同的fx/fy还包含裁剪、旋转重采样的影响，不意味着拍摄中一定发生了变焦。真实相机距离变化与内参差异本身都可由准确逐图相机模型表示，不能据此直接认定3DGS必然飘逸。

估计局限：全部数值来自53个独立SfM相机中心与六个近时刻Omega相机中心的相似变换对齐，对齐中心RMSE约为基线的1.75%。距离为任意共同尺度，不是米；估计车辆中心在近距离时会使角度更敏感。没有逐帧独立标定3207帧。图中绿色/橙色/灰色是逐抽帧的人工可见性判断，公共点云投影只能辅助定位，不能替代图像检查。

轨迹图：trajectory_coverage.png。数值：summary.json。原始全景、近车离场与后段返回检查图位于上级full_video_overview.jpg、rear_exit_review.jpg、late_front_coverage_full.jpg。已有生成轨迹、真实输入和模型未修改。
''')
    files=[Path(__file__).resolve(),SOURCE/'sfm_coverage.json',BASE/'vggtdiff_realvideo_unseen_right_near_v2/geometry_check.json']
    save(OUT/'manifest.json',dict(files=[dict(path=str(p),sha256=digest(p)) for p in files],plot_sha256=digest(OUT/'trajectory_coverage.png'),summary_sha256=digest(OUT/'summary.json')))
    (OUT/'implementation').mkdir(exist_ok=True);shutil.copyfile(__file__,OUT/'implementation'/Path(__file__).name)
    print('FIRST',summary['first_full_segment']);print('LATE',summary['late_full_segment']);print('OUTPUT',OUT)

if __name__=='__main__':main()
