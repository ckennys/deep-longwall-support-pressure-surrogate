"""Plot support-state and finite-demand evaluation from archived predictions."""
from io import BytesIO
from pathlib import Path
import hashlib
import json
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from PIL import Image

from support.audit_panel_alignment import require_matplotlib_panel_alignment
from support.visual_qa import audit_layout


ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parents[1]
DATA = ROOT / 'data'
QA = ROOT / 'qa_uniform_title_size_9pt'
NAME = 'Fig5_5_Support_state_and_demand_compact_uniform_9pt'
WIDTH_MM, HEIGHT_MM = 190., 168.
TITLE_SIZE_PT = 9.
BLUE, ORANGE = '#2166AC', '#D95F02'
SETS = [
    ('Stage7_matched_OOF', 'Stage7_OOF', 'Development OOF', 960, 639),
    ('Stage8_locked_ensemble', 'Stage8_locked', 'Archived evaluation', 64, 48),
]
TICKS = [.5, .8, 1.1, 1.4, 1.7, 2., 2.3, 2.6]


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str) + '\n', encoding='utf-8')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def configure():
    for family in ['Times New Roman', 'SimSun']:
        font_manager.findfont(font_manager.FontProperties(family=family), fallback_to_default=False)
    plt.rcParams.update({
        'font.family': 'serif', 'font.serif': ['Times New Roman'], 'font.weight': 'normal',
        'font.size': 8.5, 'axes.labelsize': TITLE_SIZE_PT, 'axes.linewidth': .65,
        'axes.titleweight': 'normal', 'axes.labelweight': 'normal',
        'xtick.labelsize': 8., 'ytick.labelsize': 8.,
        'xtick.direction': 'in', 'ytick.direction': 'in',
        'xtick.major.size': 2.6, 'ytick.major.size': 2.6,
        'xtick.major.width': .6, 'ytick.major.width': .6,
        'xtick.major.pad': 3., 'ytick.major.pad': 3.,
        'axes.unicode_minus': False, 'axes.grid': False,
        'xtick.minor.visible': False, 'ytick.minor.visible': False,
        'pdf.fonttype': 42, 'svg.fonttype': 'none', 'text.usetex': False,
        'figure.facecolor': 'white', 'axes.facecolor': 'white', 'savefig.facecolor': 'white',
        'legend.frameon': False,
    })


def prepare_data():
    source = WORKSPACE / '分析/algorithm_innovation_20260811/tables/cmcrnet_all_predictions.csv'
    summaries = WORKSPACE / '分析/algorithm_innovation_20260811/tables/cmcrnet_with_existing_baselines.csv'
    prior_csv = ROOT.parent / '第一篇_图5_5_样图提示与参考_20261028/Fig5_5_网页版绘图真实数据.csv'
    protected = {str(p): sha(p) for p in [source, summaries, prior_csv]}
    original = pd.read_csv(source)
    records = original.loc[original['variant'].eq('full_CMCRNet')].copy()
    assert len(records) == 1024
    assert records.groupby(['dataset', 'sample_id']).size().max() == 1
    assert set(records['actual_boundary']) == {0, 1}
    assert set(records['predicted_boundary']) == {0, 1}
    numeric = pd.to_numeric(records['required_support_pressure_MPa'], errors='coerce')
    assert numeric.isna().equals(records['actual_boundary'].eq(1))
    assert records.loc[records['actual_boundary'].eq(1), 'required_support_pressure_MPa'].eq('>2.6').all()
    assert np.isfinite(records['predicted_finite_pressure_MPa']).all()
    assert records['predicted_finite_pressure_MPa'].between(.5 - 1e-6, 2.6 + 1e-6).all()
    records['reference_pressure_MPa'] = numeric
    eval_map = {a: c for a, b, c, n, nf in SETS}
    records['evaluation_set'] = records['dataset'].map(eval_map)
    assert records['evaluation_set'].notna().all()
    baseline = pd.read_csv(summaries)
    prior = pd.read_csv(prior_csv)
    prior = prior.set_index(['evaluation_set', 'sample_id'])
    current = records.set_index(['evaluation_set', 'sample_id'])
    previous = prior.loc[current.index]
    for col in ['actual_boundary', 'predicted_boundary']:
        np.testing.assert_array_equal(current[col], previous[col])
    for col in ['reference_pressure_MPa', 'predicted_finite_pressure_MPa']:
        np.testing.assert_allclose(current[col], previous[col], rtol=0, atol=5e-15, equal_nan=True)
    outputs, stats, matrix_rows = [], [], []
    for dataset, summary_key, display, n, nf in SETS:
        g = records.loc[records['dataset'].eq(dataset)].copy()
        assert len(g) == n
        cm = np.array([[int(((g['actual_boundary'] == i) & (g['predicted_boundary'] == j)).sum())
                        for j in [0, 1]] for i in [0, 1]], dtype=int)
        expected = np.array([[604, 35], [12, 309]]) if n == 960 else np.array([[48, 0], [0, 16]])
        np.testing.assert_array_equal(cm, expected)
        assert np.all(cm.sum(axis=1) > 0)
        pct = cm / cm.sum(axis=1, keepdims=True) * 100.
        np.testing.assert_allclose(pct.sum(axis=1), [100., 100.])
        finite = g.loc[g['actual_boundary'].eq(0)].copy()
        assert len(finite) == nf
        y = finite['reference_pressure_MPa'].to_numpy()
        yp = finite['predicted_finite_pressure_MPa'].to_numpy()
        assert np.isfinite(y).all() and np.isfinite(yp).all()
        assert set(np.round(y, 8)).issubset(set(TICKS))
        residual = yp - y
        denom = np.sum((y - y.mean()) ** 2)
        assert denom > 0
        metric = {
            'evaluation_set': display, 'model': 'CMCRNet', 'all_records': n, 'true_finite_records': nf,
            'true_unmet_records': int(cm[1].sum()), 'state_error_finite_records': int(cm[0, 1]),
            'unmet_recall': float(cm[1, 1] / cm[1].sum()),
            'unmet_precision': float(cm[1, 1] / cm[:, 1].sum()),
            'false_finite_rate': float(cm[1, 0] / cm[1].sum()),
            'R2': float(1 - np.sum(residual ** 2) / denom),
            'RMSE_MPa': float(np.sqrt(np.mean(residual ** 2))),
            'MAE_MPa': float(np.abs(residual).mean()),
        }
        old = baseline.loc[baseline['dataset'].eq(summary_key) & baseline['variant'].eq('full_CMCRNet')]
        assert len(old) == 1
        old = old.iloc[0]
        np.testing.assert_allclose(
            [metric['R2'], metric['RMSE_MPa'], metric['MAE_MPa'], metric['unmet_recall'],
             metric['unmet_precision'], metric['false_finite_rate']],
            old[['finite_R2', 'finite_RMSE_MPa', 'finite_MAE_MPa', 'boundary_recall',
                 'boundary_precision', 'false_finite_rate']].to_numpy(dtype=float), rtol=0, atol=2e-14)
        for i in [0, 1]:
            for j in [0, 1]:
                matrix_rows.append({'evaluation_set': display, 'reference_state': i, 'predicted_state': j,
                                    'count': int(cm[i, j]), 'row_percentage': float(pct[i, j])})
        outputs.append({'records': g, 'finite': finite, 'matrix': cm, 'percentage': pct, 'metric': metric})
        stats.append(metric)
    records.to_csv(DATA / 'Fig5_5_source_records.csv', index=False, encoding='utf-8-sig', float_format='%.17g')
    pd.concat([o['finite'] for o in outputs]).to_csv(DATA / 'Fig5_5_all_true_finite_points.csv', index=False,
                                                   encoding='utf-8-sig', float_format='%.17g')
    pd.DataFrame(stats).to_csv(DATA / 'Fig5_5_recomputed_metrics.csv', index=False, encoding='utf-8-sig', float_format='%.17g')
    pd.DataFrame(matrix_rows).to_csv(DATA / 'Fig5_5_confusion_matrices.csv', index=False, encoding='utf-8-sig')
    write_json(DATA / 'source_provenance.json', {
        'model_variant': 'full_CMCRNet', 'source_sha256': protected,
        'plotting_backend': 'python-matplotlib', 'pressure_output': 'predicted_finite_pressure_MPa',
        'classification_rows': 'reference', 'classification_columns': 'predicted',
        'classification_order': ['Finite', 'Grid-unmet'], 'color_normalization': 'row percentage 0 to 100',
        'finite_subset_filter': 'actual_boundary == 0', 'predicted_state_filter': None,
        'censored_pressure_records_excluded_from_regression_only': 337,
        'points_drawn_c': 639, 'points_drawn_d': 48,
        'jitter': False, 'downsampling': False, 'smoothing': False,
        'new_model_training': False, 'archive_identity_preserved': True,
    })
    write_json(QA / 'numeric_verification.json', {'status': 'PASS', 'metrics': stats,
        'confusion_matrices': [o['matrix'].tolist() for o in outputs],
        'metric_matches_archived_summary': True, 'matches_prepared_web_csv': True})
    return outputs, protected


def add_axis(fig, left_mm, bottom_mm, letter):
    side_mm = 62.
    ax = fig.add_axes([left_mm / WIDTH_MM, bottom_mm / HEIGHT_MM,
                       side_mm / WIDTH_MM, side_mm / HEIGHT_MM], label=letter)
    ax.set_box_aspect(1)
    ax.annotate(f'({letter})', xy=(0, 1), xycoords='axes fraction', xytext=(-7, 4),
                textcoords='offset points', ha='left', va='bottom', fontsize=TITLE_SIZE_PT,
                fontweight='bold', annotation_clip=False)
    return ax


def point_annotation_audit(fig, axes, texts, legends, collections):
    """Check in-panel labels and legends against every actual scatter center."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    result = []
    for ax, group, legend, plotted in zip(axes, texts, legends, collections):
        boxes = [(t.get_text(), t.get_window_extent(renderer=renderer)) for t in group]
        if legend is not None:
            boxes.append(('state legend', legend.get_window_extent(renderer=renderer)))
        points = np.concatenate([np.asarray(p.get_offsets(), dtype=float) for p in plotted])
        xy = ax.transData.transform(points)
        for label, bbox in boxes:
            expanded = bbox.expanded(1.015, 1.04)
            inside = ((xy[:, 0] >= expanded.x0) & (xy[:, 0] <= expanded.x1)
                      & (xy[:, 1] >= expanded.y0) & (xy[:, 1] <= expanded.y1))
            result.append({'panel': ax.get_label(), 'annotation': label, 'covered_point_centers': int(inside.sum())})
    write_json(QA / 'annotation_point_clearance.json', result)
    assert all(r['covered_point_centers'] == 0 for r in result), result


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    for directory in [DATA, QA]:
        directory.mkdir(parents=True, exist_ok=True)
    outputs, protected = prepare_data()
    configure()
    fig = plt.figure(figsize=(WIDTH_MM / 25.4, HEIGHT_MM / 25.4), dpi=300)
    cm_axes, scatter_axes, panel_axes = [], [], []
    metric_texts, legends, scatter_collections, point_reports = [], [], [], []
    cmap = LinearSegmentedColormap.from_list('Support_blue', ['#F3F7FC', '#B6D1E8', '#2166AC', '#083B75'], N=256)
    norm = Normalize(vmin=0., vmax=100.)
    for col, (left_mm, obj) in enumerate(zip([24., 108.], outputs)):
        cm_ax = add_axis(fig, left_mm, 92., 'ab'[col])
        cm_ax.imshow(obj['percentage'], cmap=cmap, norm=norm, interpolation='nearest', aspect='equal')
        cm_ax.set_xticks([0, 1], ['Finite', 'Grid-unmet'])
        cm_ax.set_yticks([0, 1], ['Finite', 'Grid-unmet'])
        cm_ax.tick_params(axis='both', length=0, pad=4.)
        cm_ax.set_xlabel('Predicted state', labelpad=4.)
        cm_ax.set_ylabel('Reference state', labelpad=5.)
        cm_ax.axhline(.5, color='white', linewidth=.8)
        cm_ax.axvline(.5, color='white', linewidth=.8)
        for i in [0, 1]:
            for j in [0, 1]:
                color = 'white' if obj['percentage'][i, j] >= 55 else '#152535'
                cm_ax.text(j, i - .075, str(obj['matrix'][i, j]), ha='center', va='center',
                           fontsize=13., color=color, fontweight='normal')
                cm_ax.text(j, i + .15, f"({obj['percentage'][i, j]:.1f}%)", ha='center', va='center',
                           fontsize=9.5, color=color)
        cm_axes.append(cm_ax)
        ax = add_axis(fig, left_mm, 18., 'cd'[col])
        ax.plot([.35, 2.75], [.35, 2.75], color='#222222', linewidth=.8, zorder=1)
        finite = obj['finite']
        plotted = []
        for state in [0, 1]:
            subset = finite.loc[finite['predicted_boundary'].eq(state)]
            if len(subset) == 0:
                continue
            x = subset['reference_pressure_MPa'].to_numpy()
            y = subset['predicted_finite_pressure_MPa'].to_numpy()
            if state == 0:
                artist = ax.scatter(x, y, s=10.5, c=BLUE, alpha=.52, linewidths=0., zorder=2)
            else:
                artist = ax.scatter(x, y, s=20., facecolors='none', edgecolors=ORANGE,
                                    marker='^', linewidths=.65, alpha=.9, zorder=3)
            np.testing.assert_array_equal(np.asarray(artist.get_offsets()), np.column_stack([x, y]))
            plotted.append(artist)
        assert sum(len(p.get_offsets()) for p in plotted) == len(finite)
        scatter_collections.append(plotted)
        ax.set(xlim=(.35, 2.75), ylim=(.35, 2.75), xticks=TICKS, yticks=TICKS, aspect='equal')
        ax.set_xticklabels([f'{v:.1f}' for v in TICKS])
        ax.set_yticklabels([f'{v:.1f}' for v in TICKS])
        ax.set_xlabel('Reference support demand (MPa)', labelpad=4.)
        ax.set_ylabel('Predicted support demand (MPa)', labelpad=5.)
        mt = ax.text(.045, .955, f"R² = {obj['metric']['R2']:.4f}\nRMSE = {obj['metric']['RMSE_MPa']:.4f} MPa",
                     transform=ax.transAxes, ha='left', va='top', fontsize=8.6, linespacing=1.5)
        metric_texts.append([mt])
        legend = None
        if col == 0:
            handles = [Line2D([], [], linestyle='none', marker='o', markersize=3.5,
                              markerfacecolor=BLUE, markeredgewidth=0., alpha=.75, label='Predicted finite'),
                       Line2D([], [], linestyle='none', marker='^', markersize=4.2,
                              markerfacecolor='none', markeredgecolor=ORANGE, markeredgewidth=.65,
                              label='Predicted grid-unmet')]
            legend = ax.legend(handles=handles, loc='lower right', bbox_to_anchor=(.99, .025),
                               borderaxespad=.15, handletextpad=.45, handlelength=1.,
                               labelspacing=.55, fontsize=7.6)
        legends.append(legend)
        scatter_axes.append(ax)
        fig.text((left_mm + 31.) / WIDTH_MM, 159.5 / HEIGHT_MM, obj['metric']['evaluation_set'],
                 ha='center', va='center', fontsize=TITLE_SIZE_PT, fontweight='normal')
        point_reports.append({'panel': 'cd'[col], 'all_true_finite_points': len(finite),
                              'predicted_finite_points': int(finite['predicted_boundary'].eq(0).sum()),
                              'predicted_unmet_points': int(finite['predicted_boundary'].eq(1).sum()),
                              'all_scatter_offsets_equal_source': True, 'no_jitter': True,
                              'reference_sha256': hashlib.sha256(finite['reference_pressure_MPa'].to_numpy().tobytes()).hexdigest(),
                              'prediction_sha256': hashlib.sha256(finite['predicted_finite_pressure_MPa'].to_numpy().tobytes()).hexdigest()})
    cax = fig.add_axes([174.5 / WIDTH_MM, 92. / HEIGHT_MM, 2.8 / WIDTH_MM, 62. / HEIGHT_MM])
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax, ticks=[0, 25, 50, 75, 100])
    cb.set_label('Row percentage (%)', fontsize=TITLE_SIZE_PT, labelpad=4.)
    cb.ax.tick_params(direction='in', length=2., width=.55, labelsize=7.6, pad=2.)
    cb.outline.set_linewidth(.55)
    fig.text(.5, 165. / HEIGHT_MM, 'CMCRNet', ha='center', va='center', fontsize=TITLE_SIZE_PT, fontweight='normal')
    fig.text(.5, 4.8 / HEIGHT_MM, '图5.5 支护状态识别与有限支护需求预测', ha='center', va='center',
             fontproperties=font_manager.FontProperties(family=['Times New Roman', 'SimSun'], size=10.))
    panel_axes = [cm_axes[0], cm_axes[1], scatter_axes[0], scatter_axes[1]]
    fig.canvas.draw()
    issues = audit_layout(fig)
    write_json(QA / 'visual_layout_check.json', {'issues': issues})
    assert not any(level in ['FAIL', 'WARN'] for level, message in issues), issues
    point_annotation_audit(fig, scatter_axes, metric_texts, legends, scatter_collections)
    require_matplotlib_panel_alignment(fig, axes=panel_axes, panel_ids=list('abcd'),
        row_groups=[list('ab'), list('cd')], column_groups=[list('ac'), list('bd')],
        tolerance_pt=1.5, gutter_tolerance_pt=1.5, require_panel_labels=True, strict=True,
        json_out=QA / 'alignment.json', overlay_svg=QA / 'alignment.svg')
    fig.savefig(ROOT / f'{NAME}_preview.png', dpi=300, bbox_inches=None)
    fig.savefig(QA / f'{NAME}.pdf', bbox_inches=None)
    fig.savefig(QA / f'{NAME}.svg', bbox_inches=None)
    buffer = BytesIO()
    fig.savefig(buffer, format='png', dpi=600, bbox_inches=None)
    buffer.seek(0)
    with Image.open(buffer) as raster:
        raster.convert('RGB').save(ROOT / f'{NAME}.tiff', compression='tiff_lzw', dpi=(600, 600))
    with Image.open(ROOT / f'{NAME}.tiff') as exported:
        export = {'size_px': list(exported.size), 'mode': exported.mode,
                  'dpi': [float(v) for v in exported.info['dpi']], 'compression': exported.info.get('compression'),
                  'physical_dimensions_mm': [WIDTH_MM, HEIGHT_MM]}
        assert exported.mode == 'RGB'
        assert exported.info.get('compression') == 'tiff_lzw'
        np.testing.assert_allclose(np.asarray(exported.info['dpi'], dtype=float), [600, 600], rtol=0, atol=1e-8)
    assert protected == {path: sha(path) for path in protected}
    write_json(QA / 'rendered_point_checks.json', point_reports)
    write_json(QA / 'export_checks.json', export)
    plt.close(fig)
    print(json.dumps({'TIFF': str(ROOT / f'{NAME}.tiff'), 'points_c': 639, 'points_d': 48,
                      'source_files_unchanged': True, 'export': export}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
