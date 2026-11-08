"""Render the four-panel displacement-error figure from frozen field arrays."""
from pathlib import Path
import argparse
import hashlib
import json
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import pandas as pd

from support.audit_panel_alignment import require_matplotlib_panel_alignment

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
DISP = ROOT / '分析/第5步_链路核查与模型修正_20261025'
PRIOR = ROOT / '分析/第一篇_第五章_PI原文重读与重整_20261027'
NAME = 'Fig5_1_Displacement_error_distributions_realdata'
QA = OUT / 'qa'
EXPECTED_COUNTS = {1: 9, 2: 9, 3: 9, 4: 8, 5: 6}
FIG_WIDTH_MM = 180
FIG_HEIGHT_MM = 135


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def load(path):
    with np.load(path, allow_pickle=False) as a:
        return {key: a[key] for key in a.files}


def build_data():
    sources = [DISP/'independent_design/predictions_before_native.npz',
               DISP/'validation/新独立_原材料坐标参考场.npz',
               DISP/'validation/新独立50快照_质量状态.csv',
               DISP/'independent_design/predictions_frozen.json',
               DISP/'models_v2/独立验证前最终模型冻结.json',
               PRIOR/'逐快照指标_质量合格205行.csv']
    protected = {str(p): digest(p) for p in sources}
    predicted = load(sources[0])
    reference = load(sources[1])
    quality = pd.read_csv(sources[2]).sort_values('snapshot_index')
    frozen = json.loads(sources[3].read_text(encoding='utf-8'))
    assert protected[str(sources[0])] == frozen['predictions_sha256']
    assert quality.snapshot_index.tolist() == list(range(50))
    for key in ['case_id','step','valid_mask']:
        np.testing.assert_array_equal(predicted[key],reference[key])
    np.testing.assert_array_equal(quality.case_id.to_numpy(),predicted['case_id'])
    np.testing.assert_array_equal(quality.step.to_numpy(),predicted['step'])
    assert not quality.model_error_used_for_quality.any()
    accepted = quality.quality_status.eq('ACCEPTED_LARGE_REFERENCE').to_numpy()
    assert int(accepted.sum()) == 41
    records = []
    for i in np.flatnonzero(accepted):
        valid = predicted['valid_mask'][i]
        for target in ['uz','ux']:
            a = reference[target+'_reference_m'][i][valid].astype(np.float64)
            b = predicted[target+'_repaired_m'][i][valid].astype(np.float64)
            assert np.isfinite(a).all() and np.isfinite(b).all()
            errors = b-a
            mae = float(np.mean(np.abs(errors)))
            rmse = float(np.sqrt(np.mean(errors**2)))
            records.append({'snapshot_index':int(i),'case_id':str(predicted['case_id'][i]),
                            'physical_group':str(predicted['physical_group'][i]),
                            'support_pressure_MPa':float(predicted['features'][i,8]),
                            'step':int(predicted['step'][i]),'target':target+'_m',
                            'valid_points':int(valid.sum()),'mae_m':mae,'rmse_m':rmse,
                            'mae_cm':mae*100.,'rmse_cm':rmse*100.,
                            'quality_status':'ACCEPTED_LARGE_REFERENCE',
                            'model_variant':'Guard64_margin00'})
    frame = pd.DataFrame(records).sort_values(['target','step','snapshot_index']).reset_index(drop=True)
    assert len(frame) == 82
    assert frame.groupby('target').size().to_dict() == {'ux_m':41,'uz_m':41}
    assert not frame.duplicated(['target','snapshot_index']).any()
    for target in ['uz_m','ux_m']:
        assert frame[frame.target.eq(target)].groupby('step').size().to_dict() == EXPECTED_COUNTS
    old = pd.read_csv(sources[-1])
    joined = frame.merge(old[['target','snapshot_index','mae','rmse']],on=['target','snapshot_index'],validate='one_to_one')
    assert len(joined) == 82
    np.testing.assert_allclose(joined['mae_m'],joined['mae'],rtol=1e-12,atol=1e-14)
    np.testing.assert_allclose(joined['rmse_m'],joined['rmse'],rtol=1e-12,atol=1e-14)
    frame.to_csv(OUT/'SourceData_Fig5_1.csv',index=False,encoding='utf-8-sig')
    stats = []
    for (target,step), group in frame.groupby(['target','step']):
        for metric in ['mae_cm','rmse_cm']:
            values = group[metric].to_numpy()
            stats.append({'target':target,'step':int(step),'metric':metric,'n':len(values),
                          'physical_groups':int(group.physical_group.nunique()),
                          'mean_cm':float(values.mean()),'sd_cm':float(values.std(ddof=1)),
                          'median_cm':float(np.median(values)),'min_cm':float(values.min()),
                          'max_cm':float(values.max())})
    pd.DataFrame(stats).to_csv(OUT/'Fig5_1_group_statistics.csv',index=False,encoding='utf-8-sig')
    return frame, protected


def horizontal_offset(case_id):
    # Stable display-only horizontal jitter; the measured ordinate stays unchanged.
    integer = int(hashlib.sha256(case_id.encode('utf-8')).hexdigest()[:8],16)
    return (integer/(2**32-1)-.5)*.10


def draw(frame, preview_only=False):
    plt.rcParams.update({
        'font.family':'serif','font.serif':['Times New Roman'],
        'font.size':8.5,'mathtext.fontset':'custom','axes.titlesize':9.,
        'mathtext.rm':'Times New Roman','mathtext.it':'Times New Roman:italic',
        'mathtext.bf':'Times New Roman:bold','mathtext.bfit':'Times New Roman:italic:bold',
        'mathtext.fallback':None,
        'axes.labelsize':8.5,'axes.linewidth':.75,
        'xtick.labelsize':8.2,'ytick.labelsize':8.2,
        'xtick.major.width':.7,'ytick.major.width':.7,
        'xtick.major.size':3.,'ytick.major.size':3.,
        'xtick.direction':'in','ytick.direction':'in',
        'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none',
        'figure.facecolor':'white','axes.facecolor':'white',
        'savefig.facecolor':'white','text.usetex':False,
    })
    fig, axes = plt.subplots(2,2,figsize=(FIG_WIDTH_MM/25.4,FIG_HEIGHT_MM/25.4))
    fig.subplots_adjust(left=.085,right=.99,bottom=.18,top=.93,wspace=.235,hspace=.47)
    specifications = [
        ('a','uz_m','mae_cm',r'Vertical displacement $u_z$: MAE','MAE (cm)',3.5,np.arange(0,3.51,.5)),
        ('b','uz_m','rmse_cm',r'Vertical displacement $u_z$: RMSE','RMSE (cm)',4.5,np.arange(0,4.51,1.)),
        ('c','ux_m','mae_cm',r'Horizontal displacement $u_x$: MAE','MAE (cm)',.70,np.arange(0,.701,.1)),
        ('d','ux_m','rmse_cm',r'Horizontal displacement $u_x$: RMSE','RMSE (cm)',1.20,np.arange(0,1.201,.2)),
    ]
    center_color = '#0876B8'
    point_color = '#197FBE'
    body_color = '#9CCBE5'
    panel_checks = []
    plotted_points = []
    for ax, spec in zip(axes.flat,specifications):
        label,target,metric,title,y_label,y_top,y_ticks = spec
        groups = [frame[frame.target.eq(target)&frame.step.eq(step)].sort_values('snapshot_index') for step in range(1,6)]
        values = [group[metric].to_numpy() for group in groups]
        violins = ax.violinplot(values,positions=np.arange(1,6),widths=.66,points=200,
                               showmeans=False,showextrema=False,showmedians=False,bw_method='scott')
        for body in violins['bodies']:
            body.set_facecolor(body_color)
            body.set_edgecolor('none')
            body.set_alpha(.68)
            body.set_zorder(1)
        for step,(group,ys) in enumerate(zip(groups,values),1):
            assert len(ys) == EXPECTED_COUNTS[step]
            xs = step + np.array([horizontal_offset(case) for case in group.case_id])
            assert np.isfinite(ys).all() and float(ys.min()) >= 0
            assert float(ys.max()) < y_top
            ax.vlines(step,float(ys.min()),float(ys.max()),color=center_color,lw=.65,alpha=.72,zorder=2)
            ax.scatter(xs,ys,s=11.,color=point_color,edgecolors='none',alpha=.88,zorder=3)
            median = float(np.median(ys))
            ax.hlines(median,step-.175,step+.175,color=center_color,lw=1.7,zorder=4)
            for (_, row),px,py in zip(group.iterrows(),xs,ys):
                assert float(py) == float(row[metric])
                plotted_points.append({'panel':label,'snapshot_index':int(row.snapshot_index),
                                       'case_id':row.case_id,'target':target,'step':step,
                                       'x_display':float(px),'y_cm':float(py),'metric':metric})
        ax.set_xlim(.55,5.45)
        ax.set_ylim(0,y_top)
        ax.set_yticks(y_ticks)
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax.set_xticks(np.arange(1,6),[f'Step {step}\n$n$={EXPECTED_COUNTS[step]}' for step in range(1,6)])
        ax.tick_params(axis='x',pad=4.)
        ax.tick_params(axis='y',pad=2.)
        ax.set_ylabel(y_label,labelpad=3.)
        ax.yaxis.label.set_rotation_mode('anchor')
        ax.set_title(title,pad=7.)
        ax.set_axisbelow(True)
        ax.grid(axis='y',color='#D6DADF',linestyle='--',linewidth=.55,alpha=.9)
        ax.annotate(f'({label})',xy=(0,1),xycoords='axes fraction',
                    xytext=(-22,8),textcoords='offset points',ha='left',va='bottom',
                    fontsize=10.,fontweight='bold',annotation_clip=False)
        panel_checks.append({'panel':label,'target':target,'metric':metric,
                             'observations':sum(len(v) for v in values),'counts':EXPECTED_COUNTS,
                             'median_horizontal_bars':True,'vertical_lines':'observed min to max',
                             'y_limits_cm':[0,y_top],'all_real_observations_retained':True,
                             'kde':'Scott bandwidth, evaluated only from observed min to max',
                             'uncertainty':'descriptive distribution; no inferential interval or p value'})
    assert len(plotted_points) == 164
    pd.DataFrame(plotted_points).to_csv(QA/'plotted_points.csv',index=False,encoding='utf-8-sig')
    fig.text(.5,.079,'Fig. 5.1  Distribution of displacement prediction errors.',
             ha='center',va='center',fontsize=9.)
    fig.text(.5,.039,'Errors are calculated per quality-qualified snapshot over all valid grid points; bars indicate medians.',
             ha='center',va='center',fontsize=7.2,color='#454545')
    fig.canvas.draw()
    report = require_matplotlib_panel_alignment(
        fig,json_out=QA/'Fig5_1.alignment.json',overlay_svg=QA/'Fig5_1.alignment.svg',
        tolerance_pt=1.5,gutter_tolerance_pt=1.5,require_panel_labels=True,strict=True)
    assert all(tick._tickdir=='in' for ax in axes.flat
               for axis in [ax.xaxis,ax.yaxis] for tick in axis.get_major_ticks())
    fig.savefig(OUT/f'{NAME}_preview.png',dpi=300,bbox_inches=None)
    if preview_only:
        plt.close(fig)
        return report
    fig.savefig(OUT/f'{NAME}.png',dpi=600,bbox_inches=None)
    fig.savefig(OUT/f'{NAME}.pdf',bbox_inches=None,
                metadata={'Title':'Fig. 5.1 Displacement prediction error distributions','Author':''})
    fig.savefig(OUT/f'{NAME}.svg',bbox_inches=None)
    fig.savefig(OUT/f'{NAME}.tiff',dpi=600,bbox_inches=None,pil_kwargs={'compression':'tiff_lzw'})
    renderer = fig.canvas.get_renderer()
    for ax,label in zip(axes.flat,'abcd'):
        bounds = ax.get_tightbbox(renderer).expanded(1.045,1.04).transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(QA/f'panel_{label}.png',dpi=300,bbox_inches=bounds,pad_inches=.015)
    (QA/'panel_contracts.json').write_text(json.dumps(panel_checks,ensure_ascii=False,indent=2),encoding='utf-8')
    plt.close(fig)
    return report


def write_notes(frame, protected, alignment):
    caption_zh = ('图5.1 位移预测的逐快照空间误差分布。'
                  '(a)、(b)分别为竖向位移uz的MAE和RMSE；(c)、(d)分别为水平位移ux的MAE和RMSE。'
                  '每张快照在全部有效场点上计算误差，单位由m换算为cm。'
                  '小提琴表示描述性核密度，散点为全部质量合格快照，水平短线表示中位数，细竖线表示实际最小值至最大值。'
                  '各步快照数依次为9、9、9、8、6，总计41张，覆盖5个物理组合。'
                  '同组压力和开采步存在关联，快照数不等于独立地质试验次数。')
    caption_en = ('Fig. 5.1. Per-snapshot displacement prediction error distributions. '
                  '(a,b) MAE and RMSE for vertical displacement uz; (c,d) MAE and RMSE for horizontal displacement ux. '
                  'Errors are calculated for each quality-qualified snapshot over all valid grid points and expressed in cm. '
                  'Violins show descriptive kernel-density estimates; dots show all observations, horizontal bars indicate medians, '
                  'and thin vertical lines span the observed minimum and maximum. The five steps contain 9, 9, 9, 8, and 6 snapshots, '
                  'respectively. The 41 snapshots cover five physical groups and include related pressure cases and excavation steps; '
                  'they are not independent geological replicates.')
    (OUT/'图5.1_中英文图注.md').write_text('# 图5.1图注\n\n'+caption_zh+'\n\n'+caption_en+'\n',encoding='utf-8')
    contract = '''# 图5.1绘图约定

本图回答两个位移分量的误差分布怎样随开采步变化。沿用所给样图的四面板、小提琴、散点、蓝色中位数标记和衬线字体布局，数据与纵轴范围由冻结数组决定。

后端：沿用本任务已建立的Python绘图流程，使用Matplotlib。类型：等尺寸定量网格。四个面板分别区分分量与MAE/RMSE，均使用cm单位。物理尺寸180 mm × 135 mm。

输入为50张预先冻结位移预测及同身份参考。只采用原既定参考质量名单的41张快照，每个分量全部保留，不按误差大小剔除病例。分组为Step 1至5，各组n为9、9、9、8、6。n为快照数，覆盖5个物理组合，不能当作独立重复数。

先在逐快照有效掩膜内计算空间MAE和RMSE，再将m乘100换算为cm。小提琴采用Scott带宽，密度只在该组实测最小值和最大值之间求值；宽度为显示归一化，不表示样本数量。叠加全部原始观测。只施加可重复的横向散点偏移，纵轴值不变。横线为中位数，竖线为实际范围，无置信区间或显著性检验。

版式来自所给样图。保留其蓝色体系与Times New Roman字体；示意数据、下降趋势、原纵轴范围和示意字样均由真实结果替换。参考图只用于风格，不作为数值来源。

输出：600 dpi PNG、可编辑PDF和SVG、预览图、82行源数据、20行分组统计、可复现脚本和QA记录。四个面板合计164个绘图点，每个位移快照分别进入MAE与RMSE面板。
'''
    (OUT/'figure_contract.md').write_text(contract,encoding='utf-8')
    assert protected == {path:digest(path) for path in protected}
    result = {'date':'2026-10-27','timezone':'Asia/Tokyo','backend':'Python/Matplotlib',
              'model_variant':'Guard64_margin00','source_files_sha256':protected,
              'source_input_snapshots':50,'quality_qualified_snapshots_per_component':41,
              'source_data_rows':len(frame),'plotted_marks':164,'physical_groups':5,
              'counts_per_step':EXPECTED_COUNTS,'metric_unit_raw':'m','plot_unit':'cm',
              'conversion_factor':100.,'pixel_data_reconstructed_from_mockup':False,
              'synthetic_data_used':False,'observations_dropped_for_aesthetics':0,
              'sample_image_use':'style-only inheritance','center_statistic':'median',
              'thin_vertical_lines':'observed range, not uncertainty interval',
              'jitter':'deterministic horizontal offsets only','kde_bandwidth':'scott',
              'kde_support':'group observed minimum to maximum',
              'panel_alignment':alignment.get('verdict'),
              'final_dimensions_mm':[180,135],'PNG_dpi':600,
              'existing_inputs_unchanged':True,'new_training':False,'new_native_runs':0,
              'caption_zh':caption_zh,'caption_en':caption_en}
    (OUT/'source_provenance.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser()
    parser.add_argument('--preview-only',action='store_true')
    args = parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    QA.mkdir(exist_ok=True)
    frame, protected = build_data()
    alignment = draw(frame,preview_only=args.preview_only)
    if not args.preview_only:
        write_notes(frame,protected,alignment)
    print(json.dumps({'figure':str(OUT/f'{NAME}.png'),'snapshots_per_component':41,
                      'source_data_rows':len(frame),'counts':EXPECTED_COUNTS,
                      'alignment_verdict':alignment.get('verdict'),'source_arrays_unchanged':True},ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
