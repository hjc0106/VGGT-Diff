"""Add verified batch6/7/8 reference, target-camera and dependency figures."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch,Rectangle


def add_figures(w,pdf):
    refs6=w.read(w.MATCH/'references.json')['frames']
    refs8=w.read(w.B8/'manifest.json')['sources']
    refs7=w.read(w.B7/'manifest.json')['sources']
    t6=w.read(w.MATCH/'trajectory.json');t7=w.read(w.B7/'trajectory.json');t8=w.read(w.B8/'trajectory.json')
    geometry=w.read(w.BASE/'vggtdiff_realvideo_unseen_right_near_v2/geometry_check.json')
    center=np.array(geometry['center']);front=np.array(geometry['front']);left=np.array(geometry['left'])
    up=np.array(geometry['up']);up/=np.linalg.norm(up)
    colors=[w.BLUE,w.ORANGE,w.GREEN]
    trajectories=[t6,t7,t8]
    stats={}
    for batch,t in zip([6,7,8],trajectories):
        delta=np.linalg.inv(np.array(t['target_w2c']))[:,:3,3]-center
        height=delta@up;radius=np.linalg.norm(delta-height[:,None]*up,axis=1)
        stats[batch]=dict(distance=np.linalg.norm(delta,axis=1).tolist(),radius=radius.tolist(),height=height.tolist())
    # Spatial positions from actual per-image W2C; no radius normalization.
    fig=w.canvas('08  第6、7、8批：参考来源、目标相机与生成依赖','最终采用：真实参考A → 第6批；真实参考B → 第8批；再由第6/8批双侧补第7批。')
    ax=fig.add_axes([.045,.29,.44,.55]);ax.set_aspect('equal')
    theta=np.linspace(0,2*np.pi,300);r=stats[7]['radius'][0]
    ax.plot(r*np.sin(theta),r*np.cos(theta),'--',color='#BBC5D0',lw=1)
    ax.add_patch(Rectangle((-.06,-.12),.12,.24,facecolor='#DDE5ED',edgecolor='#586A7B'))
    ax.annotate('',xy=(0,.22),xytext=(0,.03),arrowprops=dict(arrowstyle='->',color='#314B62',lw=2))
    ax.text(.035,.17,'车头',fontsize=11)
    for j,(t,color,batch) in enumerate(zip(trajectories,colors,[6,7,8])):
        d=np.linalg.inv(np.array(t['target_w2c']))[:,:3,3]-center
        x=-d@left;y=d@front
        ax.plot(x,y,'o-',color=color,lw=2,ms=5,label=f'第{batch}批目标相机')
        if batch in [6,8]:
            idx=[5,6,7] if batch==6 else [0,1,2]
            ax.scatter(x[idx],y[idx],marker='*',s=135,color=color,edgecolors='white',linewidths=.5,zorder=6)
        if batch==6:
            ax.annotate('224.6°',xy=(x[0],y[0]),xytext=(12,-20),textcoords='offset points',fontsize=10,color=color)
            ax.annotate('263.9°',xy=(x[-1],y[-1]),xytext=(-5,-25),textcoords='offset points',fontsize=10,color=color)
        if batch==8:
            ax.annotate('354.4°',xy=(x[-1],y[-1]),xytext=(12,3),textcoords='offset points',fontsize=10,color=color)
    for rows,Tkey,marker,color,label in [(refs6,'T','^',w.BLUE,'第6批真实参考'),(refs8,'w2c','D',w.GREEN,'第8批真实参考')]:
        d=np.linalg.inv(np.array([r[Tkey] for r in rows]))[:,:3,3]-center
        ax.scatter(-d@left,d@front,marker=marker,s=48,facecolors='none',edgecolors=color,label=label,alpha=.8)
    ax.set_xlim(-.85,.85);ax.set_ylim(-.85,.85);ax.set_xticks([-.6,0,.6]);ax.set_yticks([-.6,0,.6]);ax.grid(alpha=.13)
    ax.set_title('实际相机中心的水平投影（相对世界尺度）',fontsize=12)
    ax.text(.45,.7,'0° 前方',fontsize=10,color='#637487');ax.text(-.13,-.77,'180° 后方',fontsize=10,color='#637487')
    ax.legend(loc='lower right',fontsize=8.5,frameon=True,facecolor='white')
    # Angle-only lanes clearly separate the variable-distance real cameras from targets.
    lane=fig.add_axes([.62,.40,.34,.39])
    specs=[([r['azimuth_deg'] for r in refs6],5,'真实参考A',w.BLUE,'^',False),
        (t6['target_angles_deg'],4,'第6批目标',w.BLUE,'o',True),
        (t7['target_angles_deg'],3,'第7批目标',w.ORANGE,'o',True),
        (t8['target_angles_deg'],2,'第8批目标',w.GREEN,'o',True),
        ([r['angle'] for r in refs8],1,'真实参考B',w.GREEN,'D',False)]
    for angles,y,label,color,marker,solid in specs:
        lane.plot([min(angles),max(angles)],[y,y],color=color,alpha=.3,lw=4)
        lane.scatter(angles,[y]*len(angles),marker=marker,s=38,color=color if solid else 'white',edgecolor=color,zorder=3)
        for a in angles:lane.text(a,y+.13,f'{a:.1f}',rotation=55,fontsize=7.2,ha='left',color=color)
    lane.scatter(np.array(t6['target_angles_deg'])[[5,6,7]],[4]*3,marker='*',s=120,color=w.BLUE,zorder=4)
    lane.scatter(np.array(t8['target_angles_deg'])[[0,1,2]],[2]*3,marker='*',s=120,color=w.GREEN,zorder=4)
    lane.set_yticks([1,2,3,4,5],['真实参考B','第8批目标','第7批目标','第6批目标','真实参考A'])
    lane.set_xlim(205,362);lane.set_ylim(.7,5.8);lane.set_xticks([210,240,270,300,330,360]);lane.set_xlabel('车辆坐标系方位角 / °')
    lane.set_title('角度关系：星号为补第7批时采用的生成参考',fontsize=11,pad=12);lane.grid(axis='x',alpha=.15)
    fig.text(.54,.29,'第6批：真实目标相机，非等间隔，视距0.444–0.624。',fontsize=10,color=w.BLUE)
    fig.text(.54,.255,'第7/8批：间隔5.625°；视距0.624；fx/fy=964.6/968.6。',fontsize=10,color=w.GREEN)
    # Final dependency graph, independent real anchors at both sides.
    boxes=[('六张真实参考A',w.BLUE),('生成第6批',w.BLUE),('双侧补第7批',w.ORANGE),('生成第8批',w.GREEN),('六张真实参考B',w.GREEN)]
    for j,(label,color) in enumerate(boxes):
        x=.035+j*.195
        patch=FancyBboxPatch((x,.13),.15,.065,boxstyle='round,pad=0.008',transform=fig.transFigure,facecolor=color,edgecolor='none')
        fig.patches.append(patch);fig.text(x+.075,.158,label,ha='center',va='center',fontsize=12,color='white')
    for start,end in [(.194,.221),(.389,.416),(.611,.584),(.806,.779)]:
        fig.add_artist(FancyArrowPatch((start,.162),(end,.162),transform=fig.transFigure,
            arrowstyle='-|>',mutation_scale=18,color='#566A7A',lw=1.8,zorder=20))
    fig.text(.245,.08,'B6取局部05/06/07',fontsize=10,color=w.BLUE)
    fig.text(.585,.08,'B8取局部00/01/02',fontsize=10,color=w.GREEN)
    w.finish(fig,'08_batch678_view_relationship','三批最终关系：第6、8批分别由真实图锚定，第7批由两端生成图补间；第6批是真实目标相机，第7/8批是连续固定半径目标轨迹。',
        '左图来自实际W2C，不按相机半径归一化；右图仅展示方位角。距离为相对单位，非米；星号参考图本身为生成图。',pdf)
    # Contact sheet of all actual references, with explicit provenance.
    fig=w.canvas('09  三批实验实际使用的参考图','第6批：六真实；第7批最终版：三张第6批生成 + 三张第8批生成；第8批最终版：六真实。')
    for ri,(rows,label) in enumerate([(refs6,'第6批输入：六张真实图'),(refs7,'第7批输入：双侧生成参考'),(refs8,'第8批输入：六张真实图')]):
        y=.65-ri*.265
        fig.text(.035,y+.215,label,fontsize=13,weight='bold',color=colors[ri])
        for ci,row in enumerate(rows):
            if ri==0:path=row['image'];title=f"真实帧{row['original_frame']} | {row['azimuth_deg']:.1f}°"
            elif ri==1:path=row['original'];title=f"B{row['source_batch']}局部{row['source_local_index']:02d} | {row['angle']:.1f}°"
            else:path=row['original'];title=f"真实帧{row['original_frame']} | {row['angle']:.1f}°"
            ax=w.show(fig,[.035+ci*.16,y,.148,.17],w.image(path),title);ax.title.set_fontsize(9)
    w.finish(fig,'09_batch678_reference_images','三批最终输入参考图及来源：第6批真实参考跨度较大；第8批真实参考集中于车头邻近；第7批由两端各三张生成图补间。',
        '参考图按实际输入顺序展示，保留各自裁剪与内参；仅统一显示尺寸。真实参考有不同视距及有效内参，不把它们当作固定半径相机。',pdf)
    write_details(w,refs6,refs7,refs8,t6,t7,t8,stats)


def write_details(w,refs6,refs7,refs8,t6,t7,t8,stats):
    lines=['# 第6、7、8批：实验设置、参考图、目标相机和结果','',
        '本说明描述最终用于24图重建的版本，并将早期单向链式生成作为对照。批次编号来自原64视角分段；第6批本轮实际相机来自真实GT匹配实验，不能直接套用225°起的等间隔表。','',
        '## 一页设置与结果','',
        '| 批次 | 实际参考图 | 八个目标相机 | 结果与证据范围 |',
        '|---|---|---|---|',
        '| 第6批 | 六真实帧0/480/720/960/1200/1440 | 224.598°–263.911°，真实留出图对应相机，非等间隔 | 三种子近邻内部极线误差0.416–0.643px；同相机生成–GT位置偏差4.67–6.00px。后续采用seed20261028。 |',
        '| 第7批最终版 | B6局部05/06/07 + B8局部00/01/02，六张均为生成图 | 270°–309.375°，间隔5.625°，延续B6末端相机 | 批内三视角中位1.413px；左边界1.699px，右边界1.376px。均为近邻条件兼容性证据。 |',
        '| 第8批最终版 | 六真实帧3054/2993/2872/2751/2691/2630 | 315°–354.375°，间隔5.625°，与B7相同轨道和K | 六真实组车头更稳定；批内三视角中位1.009px；不能据此认定与旧B7自动衔接。 |','',
        '## 参考图及精确角度','']
    for batch,rows in [(6,refs6),(7,refs7),(8,refs8)]:
        lines += [f'### 第{batch}批参考','', '| 输入顺序 | 来源 | 方位角 |','|---:|---|---:|']
        for j,row in enumerate(rows):
            if batch==6:source=f"真实视频帧{row['original_frame']}";angle=row['azimuth_deg']
            elif batch==8:source=f"真实视频帧{row['original_frame']}";angle=row['angle']
            else:source=f"第{row['source_batch']}批生成局部{row['source_local_index']:02d}，源运行输出编号{row['source_output_index']}";angle=row['angle']
            lines.append(f'| {j} | {source} | {angle:.6f}° |')
        lines.append('')
    lines += ['## 目标视角与相机','']
    for batch,t in zip([6,7,8],[t6,t7,t8]):
        s=stats[batch]
        lines += [f'- 第{batch}批目标角度：'+ '、'.join(f'{a:.6f}°' for a in t['target_angles_deg'])+'。',
            f"- 第{batch}批视距范围：{min(s['distance']):.6f}–{max(s['distance']):.6f}；水平半径{min(s['radius']):.6f}–{max(s['radius']):.6f}；高度{min(s['height']):.6f}–{max(s['height']):.6f}。"]
    lines += ['',
        '所有目标图为832×480，目标K相同：fx=964.644898，fy=968.608741，cx=415.5，cy=239.5；W2C为OpenCV约定。第6批各真实目标相机的距离/高度不同；第7/8批以第6批末端相机为锚，绕共同车辆中心/up轴同时旋转相机位置与姿态，因此保持轨道半径、高度、中心投影及内参。',
        '第6批八个目标对应真实视频帧1340/1300/1220/1180/1100/1060/980/910，作为留出GT，不属于六张生成条件图。',
        '第6批末端263.911216°→第7批首帧270°实际跨批角差6.088784°；第7批末帧309.375°→第8批首帧315°角差5.625°。所有距离为任意共同尺度，不是米。','',
        '## 实际生成依赖与迭代','',
        '最终采用的依赖为：真实参考A→第6批；真实参考B→第8批；第6批末三张+第8批前三张→重生成第7批。第7批是双侧补间，不应画成最终仍由第6批单向生成第7批、再由第7批生成第8批。','',
        '早期对照包括：B6生成参考→原第7批近轨迹（尺度跳变）；B6生成参考→连续轨迹第7批（相机连续，但车头偏离）；B6/B7生成参考→第8批（车头外观错误）；混合真实/生成参考第8批（仍有外观切换）；六真实参考第8批（改善）；最终双侧补第7批（改善与新版B8的边界）。','',
        '## 结果应如何表述','',
        '- B6三种子共同显示局部内部自洽与真实投影偏差同时存在。采用seed20261028是后续候选选择，不代表该种子在每一项内部指标上最优。',
        '- B8六真实版本的三视角中位1.009px，生成/混合参考对照为2.102/4.934px；源相机随参考替换变化，不能单独归因于真实性。',
        '- B7双侧补间后，B7→B8三视角中位11.866→1.376px、P90 31.127→2.591px、近邻支持0/3→3/3。B6→B7中位1.677→1.699px，不能说两侧所有指标均改善。',
        '- 三批局部兼容性改善后，24图3DGS仍有拖影。现有证据不支持“全局几何正确”或“拉丝已解决”。','',
        '## 推理配置与数据来源','',
        '三批均使用相同VGGT-Diff NVS权重、832×480、50步、每次八目标。B6种子20261026/27/28；最终使用20261028。B7最终种子20261029，B8最终种子20261030。相机为已有Omega/PnP/SfM注册估计，非独立标定位姿GT。','']
    for label,path in [('B6相机',w.MATCH/'trajectory.json'),('B6真实参考',w.MATCH/'references.json'),('B7输入',w.B7/'manifest.json'),('B7相机',w.B7/'trajectory.json'),('B8输入',w.B8/'manifest.json'),('B8相机',w.B8/'trajectory.json')]:lines.append(f'- [{label}]({path})')
    lines += ['', '![视角与生成关系](figures/08_batch678_view_relationship.png)','', '![实际参考图](figures/09_batch678_reference_images.png)','']
    (w.OUT/'BATCH678_SETUP.md').write_text('\n'.join(lines))
    w.write_setup_stats=dict(stats=stats,reference6_angles=[r['azimuth_deg'] for r in refs6])
