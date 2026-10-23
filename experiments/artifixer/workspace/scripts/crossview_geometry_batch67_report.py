"""Write the frozen batch6/batch7 geometry audit interpretation and tables."""
from pathlib import Path
import csv
import shutil
import crossview_geometry_batch67 as exp


def main():
    exp.validate();s=exp.read(exp.OUT/'summary.json');p=exp.read(exp.OUT/'protocol.json')
    audit=exp.read(exp.OUT/'audit.json');assert audit['state']=='passed'
    table=[]
    names={'batch6':'第6批内部','batch7':'第7批内部','cross_batch':'跨批次近邻'}
    for group in names:
        v=s['pair_groups'][group]['primary15']
        table.append(f"| {names[group]} | {v['supported_pairs']}/{v['n_pairs']} | {v['supported_median_pair_epi_px']:.3f} | {v['supported_median_pair_p90_px']:.3f} | {v['median_matches']:g} |")
    pairs=[]
    with (exp.OUT/'pair_metrics.csv').open() as f:
        for row in csv.DictReader(f):
            if row['group']=='cross_batch' and float(row['azimuth_gap_deg'])<=15:
                pairs.append(row)
    boundary=s['boundary_pair'];track_table=[]
    for group,name in [('batch6','第6批内部'),('boundary','跨批次交界'),('batch7','第7批内部')]:
        v=s['track_groups'][group]['cycle']
        track_table.append(f"| {name} | {v['n_tracks']} | {v['computed_inside_image_median_px']:.3f} | {v['computed_inside_image_p90_px']:.3f} | {v['computational_failure_rate']:.2%} | {v['valid_track_fraction_4px']:.2%} |")
    individual=[]
    for row in s['triples']:
        if row['group']=='boundary':
            v=row['cycle'];individual.append(f"| {' + '.join(row['labels'])} | {v['n_tracks']} | {v['computed_inside_image_median_px']:.3f} | {v['computed_inside_image_p90_px']:.3f} | {v['valid_track_fraction_2px']:.2%} | {v['valid_track_fraction_4px']:.2%} |")
    cross='\n'.join(f"| {r['label_a']} → {r['label_b']} | {float(r['azimuth_gap_deg']):.3f}° | {r['n_matches']} | {float(r['epi_median_px']):.3f} | {float(r['epi_p90_px']):.3f} | {float(r['coverage_a']):.1%}/{float(r['coverage_b']):.1%} |" for r in pairs)
    report='''# 第6批与连续相机第7批：车辆区域几何一致性

## 结论

局部近邻视角上，两批的极线兼容性处在相近量级，批次交界没有显示显著的极线误差突增。跨批次三视角检查仍有约1–3px的典型残余误差，两个交界三视角组表现明显不同。因此可认为局部衔接较好，不能宣称完全几何一致，也不能据此保证联合3DGS不会飘逸。

极线误差只检查对应点与指定相机极线的兼容性，沿极线的深度/视差偏差可能仍很小；三视角留出重投影更直接检查对应点是否可以来自同一个三维位置。不能只依据低极线误差判断重建质量。

## 数据与冻结方法

第6批为crossview_geometry_matched_gt_v1/seed_20261028的全部八张图，实际224.598°–263.911°；第7批为vggtdiff_batch06gen_to_batch07_continuous_v1的八张图，270°–309.375°。B6_00–B6_07、B7_00–B7_07是各批局部序号；文件内0–15是本次评估ID，不是原视频帧号。

仅在白车mask内提取SIFT（12000上限），双向mutual匹配、Lowe ratio0.75；不做极线或RANSAC预筛，也不重新拟合相机。第6批复用既有人工检查的SAM3 mask和特征；第7批使用相同SAM3白车选择方法。全部16张mask已检查，保留车窗、轮胎，排除路面、阴影与背景车辆。

误差统一以832×480像素报告，极线误差为sqrt(Sampson)。少于30个对应点标为证据不足，同时报告车辆空间覆盖。此支持数规则用于证据描述，不是质量合格线。所有角差≤15°的图像对是事前固定主集合。

本次计算全部120对，包括批内28+28对和跨批次64对。主集合为第6批14对、第7批13对、跨批次3对。三视角为顺序相邻的14个局部三元组：第6批6个、交界2个、第7批6个；原始图匹配轨迹和直接三边描述子闭环轨迹分别报告，不按留出重投影误差挑选轨迹。主三角化视差要求1°，另保留0.5°和2°结果。

## 两视角主指标

下表对图像对等权聚合：“误差中位数”是各支持对中位误差的中位数，“P90中位数”是各支持对P90的中位数，并非把所有点混合后的P90。

| 分组，角差≤15° | 支持对数/总数 | 误差中位数px | 各对P90中位数px | 匹配数中位数 |
|---|---:|---:|---:|---:|
'''+ '\n'.join(table)+'''

| 跨批次主集合 | 角差 | 匹配数 | 中位误差px | P90px | 车辆覆盖A/B |
|---|---:|---:|---:|---:|---:|
'''+cross+f'''

直接交界B6_07→B7_00有{boundary['n_matches']}个匹配，中位误差{boundary['epi_median_px']:.3f}px，P90为{boundary['epi_p90_px']:.3f}px。mask向内腐蚀4px后仍有{boundary['n_matches_erode4']}个匹配，中位误差{boundary['epi_median_erode4_px']:.3f}px，说明该结论不是仅由轮廓边缘匹配支撑。

## 三视角留出重投影

以下使用直接三边描述子闭环轨迹；中位数/P90针对计算成功且投影落在图内的留出尝试，同时报告计算失败。每条轨迹有三个轮流留出的视角；不同局部三元组可能重复观察同一特征，因此轨迹数不是独立物理点数。

| 分组 | 局部闭环轨迹数 | 中位误差px | P90px | 计算失败率 | 三个留出误差均≤4px的轨迹比例 |
|---|---:|---:|---:|---:|---:|
'''+ '\n'.join(track_table)+'''

交界处必须分开看，汇总值会掩盖两组之间的差别：

| 三视角组 | 轨迹数 | 中位误差px | P90px | 三个留出误差均≤2px | 均≤4px |
|---|---:|---:|---:|---:|---:|
'''+ '\n'.join(individual)+f'''

第一组典型误差2.90px，严格2px以内的轨迹比例较低；第二组约1.02px、明显更好。这说明残余三维兼容性偏差仍在，不能用0.50px极线误差替代这项检查。2px/4px仅用于描述分布，不是预设质量合格线。全部原始轨迹和失败原因均保留在CSV中。

## 证据范围

- 跨批次全部64对中，{s['pair_groups']['cross_batch']['all']['supported_pairs']}对至少30个匹配，其余51对证据不足。宽角差不能据此判为一致或不一致，较小主集合也不能代表全部区域。
- 三个主跨批次对均涉及用来生成第7批的条件参考图，因此这不是独立真实GT验证。第7批没有同相机真实GT。
- 指标使用冻结的请求相机参数，检验图像对应关系与这些参数的兼容性，没有从生成图反估并验证绝对R/t、尺度或真实车身三维尺寸。
- 真实第6批GT此前主集合中位误差约0.351px，可作量级参照；第7批视角、基线和特征不同，不是严格配对的GT误差比较。
- 尚未评估旧近距版的同类指标，因此本报告没有声称连续相机修改已经降低几何误差，也未进行16帧联合3DGS重建。

建议下一步用准确对应相机联合重建这16帧，比较第6批8帧与第6+7批16帧在共同局部新视角的拉丝/轮廓表现。当前生成数据有继续验证的依据，尚不能预先保证重建无飘逸。

## 产物与检查

[跨批次主集合匹配图](cross_primary_matches.jpg)、[误差与支持数热力图](pair_heatmaps.png)、[车辆mask检查图](mask_review.jpg)、[逐对指标](pair_metrics.csv)、[闭环轨迹](tracks_local_cycle.csv)、[完整汇总](summary.json)。

图像、mask、相机、协议和评估代码均冻结。第6批主集合0.4162051191px与之前结果精确复现，证明复用了相同的匹配/误差约定。审计通过，详见audit.json；审计通过不表示设定了统一几何质量合格线。

复现入口：scripts/crossview_geometry_batch67.py（prepare、检查mask、accept-masks、evaluate），环境.venv-car15，只有prepare的SAM3分割需要GPU；报告脚本为scripts/crossview_geometry_batch67_report.py。原图像及历史实验均未修改。
'''
    (exp.OUT/'REPORT.md').write_text(report)
    exp.job.chain.frozen_copy(Path(__file__).resolve(),exp.OUT/'implementation'/Path(__file__).name)
    for path,name in [('/tmp/batch67_geometry_prepare.log','prepare.log'),('/tmp/batch67_geometry_evaluate.log','evaluate.log')]:
        dest=exp.OUT/'logs'/name;dest.parent.mkdir(exist_ok=True);shutil.copyfile(path,dest)
    feature_hashes=[dict(path=str(p),sha256=exp.job.chain.digest(p)) for p in sorted((exp.OUT/'features').glob('*.npz'))]
    exp.save(exp.OUT/'final_audit.json',dict(state='completed',report_sha256=exp.job.chain.digest(exp.OUT/'REPORT.md'),
        audit=audit,feature_hashes=feature_hashes,report_script_sha256=exp.job.chain.digest(Path(__file__)),
        interpretation='Local epipolar compatibility is comparable across the boundary; residual heldout3view errors remain. No whole-scene or reconstruction pass claim.'))
    exp.save(exp.OUT/'status.json',dict(state='completed',images=16,pairs=120,primary_cross_pairs=3,local_triples=14))
    print('REPORT_READY',exp.OUT/'REPORT.md')

if __name__=='__main__':main()
