"""Reconstruct the two-panel constraint diagnostic from archived real records."""
from io import BytesIO
from pathlib import Path
import hashlib
import json
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
import numpy as np
import pandas as pd
from PIL import Image

from support.audit_panel_alignment import require_matplotlib_panel_alignment
from support.visual_qa import audit_layout


ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parents[1]
DATA = ROOT / 'data_extra_compact'
QA = ROOT / 'qa_extra_compact'
NAME = 'Fig5_6_CMCRNet_constraints_extra_compact'
WIDTH_MM, HEIGHT_MM = 180., 84.
FONT_PT = 9.
VARIANTS = ['independent_multitask', 'plus_censored_loss', 'plus_coupled_ranking', 'full_CMCRNet']
LABELS = ['Anchored\nMTL', '+\nCensoring', '+ Coupling\nand ranking', 'Full\nCMCRNet']
DATASETS = ['Stage7_matched_OOF', 'Stage8_locked_ensemble']
DISPLAY_DATASETS = ['Development OOF', 'Archived evaluation']
FEATURES = ['M_m', 'q_top_MPa', 'alpha_E_roof', 'alpha_c_roof', 'phi_roof_deg', 'alpha_t_roof']
FEATURE_LABELS = [r'$M$', r'$q_{\mathrm{top}}$', r'$\alpha_E$', r'$\alpha_c$', r'$\phi$', r'$\alpha_t$']
COLORS = ['#2E5EA4', '#8DB5E3']
THRESHOLD_MPA = 2.7


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def js(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + '\n', encoding='utf-8')


def configure():
    for family in ['Times New Roman', 'SimSun']:
        font_manager.findfont(font_manager.FontProperties(family=family), fallback_to_default=False)
    plt.rcParams.update({
        'font.family': 'serif', 'font.serif': ['Times New Roman'], 'font.weight': 'normal',
        'font.size': FONT_PT, 'axes.labelsize': FONT_PT, 'axes.titlesize': FONT_PT,
        'axes.titleweight': 'normal', 'axes.labelweight': 'normal', 'axes.linewidth': .65,
        'xtick.labelsize': FONT_PT, 'ytick.labelsize': FONT_PT,
        'xtick.direction': 'in', 'ytick.direction': 'in',
        'xtick.major.size': 2.8, 'ytick.major.size': 2.8,
        'xtick.major.width': .6, 'ytick.major.width': .6,
        'xtick.major.pad': 4., 'ytick.major.pad': 3.,
        'xtick.minor.visible': False, 'ytick.minor.visible': False,
        'axes.grid': False, 'axes.unicode_minus': False,
        'mathtext.fontset': 'custom', 'mathtext.rm': 'Times New Roman',
        'mathtext.it': 'Times New Roman:italic', 'mathtext.bf': 'Times New Roman:bold',
        'mathtext.bfit': 'Times New Roman:italic:bold', 'mathtext.fallback': None,
        'pdf.fonttype': 42, 'svg.fonttype': 'none', 'text.usetex': False,
        'legend.fontsize': FONT_PT, 'legend.frameon': False,
        'figure.facecolor': 'white', 'axes.facecolor': 'white', 'savefig.facecolor': 'white',
    })


def prepare():
    folder = WORKSPACE / '分析/algorithm_innovation_20260811/tables'
    audit_folder = WORKSPACE / '分析/第一篇_5.5图表配置核对_20261029'
    paths = [folder / 'cmcrnet_all_predictions.csv', folder / 'cmcrnet_ablation_metrics.csv',
             folder / 'cmcrnet_monotonic_violation_rates.csv',
             WORKSPACE / '脚本/run_censored_monotonic_joint_network.py']
    prior = json.loads((audit_folder / '图表配置核对.json').read_text(encoding='utf-8'))
    protected = {str(p): sha(p) for p in paths}
    for p in paths:
        assert protected[str(p)] == prior['source_sha256'][str(p)]
    records, summary, mono = [pd.read_csv(p) for p in paths[:3]]
    assert set(records['variant']) == set(VARIANTS) and len(records) == 4096
    assert records.groupby(['variant', 'dataset', 'sample_id']).size().max() == 1
    bars = np.empty((4, 2), dtype=float)
    rows, used = [], []
    saved = pd.read_csv(audit_folder / '图5.6_两种截尾阈值的真实统计.csv')
    expected_counts = [[59, 0], [247, 13], [226, 15], [238, 16]]
    for i, variant in enumerate(VARIANTS):
        for j, dataset in enumerate(DATASETS):
            all_rows = records.loc[records['variant'].eq(variant) & records['dataset'].eq(dataset)]
            assert len(all_rows) == [960, 64][j]
            subset = all_rows.loc[all_rows['actual_boundary'].eq(1)].copy()
            assert len(subset) == [321, 16][j]
            assert subset['required_support_pressure_MPa'].eq('>2.6').all()
            latent = subset['predicted_latent_demand_MPa'].to_numpy()
            assert np.isfinite(latent).all()
            satisfied = latent >= THRESHOLD_MPA
            assert int(satisfied.sum()) == expected_counts[i][j]
            old = summary.loc[summary['variant'].eq(variant) & summary['dataset'].eq(dataset)]
            assert len(old) == 1
            assert abs(np.mean(latent >= 2.6) - float(old.iloc[0]['boundary_censor_margin_satisfaction'])) < 1e-14
            bars[i, j] = satisfied.mean() * 100.
            previous = saved.loc[saved['variant'].eq(variant) & saved['dataset'].eq(dataset)].iloc[0]
            np.testing.assert_allclose(bars[i, j], previous['training_margin_ge_2p7_rate'] * 100., rtol=0, atol=2e-12)
            subset['training_margin_threshold_MPa'] = THRESHOLD_MPA
            subset['training_margin_satisfied'] = satisfied
            used.append(subset)
            rows.append({'variant': variant, 'evaluation_set': DISPLAY_DATASETS[j],
                         'true_unmet_combinations': len(subset), 'satisfied_combinations': int(satisfied.sum()),
                         'threshold_MPa': THRESHOLD_MPA, 'satisfaction_percent': bars[i, j]})
    assert len(mono) == 120 and mono['dataset'].eq('Stage8_locked').all()
    assert mono.groupby(['variant', 'fold', 'feature']).size().eq(1).all()
    assert set(mono['variant']) == set(VARIANTS) and set(mono['feature']) == set(FEATURES)
    assert set(mono['fold']) == {1, 2, 3, 4, 5}
    rate = mono['demand_violation_rate'].to_numpy()
    assert np.isfinite(rate).all() and np.all((0 <= rate) & (rate <= 1))
    np.testing.assert_allclose(rate * 64, np.round(rate * 64), rtol=0, atol=2e-12)
    directions = {'M_m': 1, 'q_top_MPa': 1, 'alpha_E_roof': -1, 'alpha_c_roof': -1,
                  'phi_roof_deg': -1, 'alpha_t_roof': -1}
    assert all(mono.loc[mono['feature'].eq(f), 'expected_direction'].eq(direction).all() for f, direction in directions.items())
    heat_frame = mono.pivot_table(index='feature', columns='variant', values='demand_violation_rate', aggfunc='mean').loc[FEATURES, VARIANTS]
    heat = heat_frame.to_numpy() * 100.
    old_heat = pd.read_csv(audit_folder / '图5.6_固定教师锚点局部需求违例率_五折均值.csv').set_index('feature').loc[FEATURES, VARIANTS]
    np.testing.assert_allclose(heat, old_heat.to_numpy() * 100., rtol=0, atol=2e-12)
    assert heat.shape == (6, 4) and np.isfinite(heat).all() and np.all((heat >= 0) & (heat <= 100))
    np.testing.assert_allclose(heat.mean(0), [47.239583333333336, 40.05208333333333, 43.333333333333336, 32.34375], atol=1e-12)
    pd.DataFrame(rows).to_csv(DATA / 'Fig5_6a_training_margin_counts_and_rates.csv', index=False, encoding='utf-8-sig', float_format='%.17g')
    pd.concat(used).to_csv(DATA / 'Fig5_6a_all_true_unmet_records.csv', index=False, encoding='utf-8-sig', float_format='%.17g')
    mono.to_csv(DATA / 'Fig5_6b_all_120_fold_feature_records.csv', index=False, encoding='utf-8-sig', float_format='%.17g')
    pd.DataFrame(heat, index=FEATURES, columns=VARIANTS).rename_axis('feature').to_csv(DATA / 'Fig5_6b_mean_violation_percent.csv', encoding='utf-8-sig', float_format='%.17g')
    np.savez_compressed(DATA / 'Fig5_6_plotted_arrays.npz', satisfaction_percent=bars, violation_percent=heat,
                        variants=np.asarray(VARIANTS), features=np.asarray(FEATURES))
    js(DATA / 'source_provenance.json', {'source_sha256': protected,
        'panel_a_filter': 'actual_boundary == 1', 'panel_a_threshold_MPa': THRESHOLD_MPA,
        'panel_a_all_true_unmet_records': sum(len(s) for s in used), 'panel_a_denominators_per_variant': [321, 16],
        'panel_b_records_used': len(mono), 'panel_b_aggregation': 'equal mean across 5 fold models for each feature and variant',
        'panel_b_combinations_per_fold': 64, 'panel_b_teacher_anchor_held_fixed': True,
        'panel_b_standardized_input_perturbation': .08, 'panel_b_demand_tolerance_MPa': .0001,
        'sampling': False, 'inferred_error_bars': False, 'new_model_training': False,
        'figure_data_reconstructed_from_original_archives': True, 'mockup_used_for_style_only': True})
    js(QA / 'numeric_verification.json', {'status': 'PASS', 'bar_count': 8, 'heatmap_cells': 24,
        'satisfied_counts': expected_counts, 'satisfaction_percent': bars.tolist(),
        'violation_percent': heat.tolist(), 'violation_column_means_percent': heat.mean(0).tolist(),
        'all_24_cells_and_all_8_bars_match_prior_verified_statistics': True,
        'percentage_label_rounding': 'Python fixed-point formatting to 1 decimal; unrounded values used for geometry'})
    return bars, heat, protected


def rect(left, bottom, width, height):
    return [left / WIDTH_MM, bottom / HEIGHT_MM, width / WIDTH_MM, height / HEIGHT_MM]


def add_panel_label(ax, letter):
    return ax.annotate(f'({letter})', xy=(0, 1), xycoords='axes fraction', xytext=(-28, 4),
                       textcoords='offset points', ha='left', va='bottom', fontsize=FONT_PT,
                       fontweight='bold', annotation_clip=False)


def contrast_text_color(rgb):
    channel = np.asarray(rgb[:3], dtype=float)
    linear = np.where(channel <= .04045, channel / 12.92, ((channel + .055) / 1.055) ** 2.4)
    luminance = float(np.dot(linear, [.2126, .7152, .0722]))
    black_contrast = (luminance + .05) / .05
    white_contrast = 1.05 / (luminance + .05)
    return ('black', black_contrast) if black_contrast >= white_contrast else ('white', white_contrast)


def audit_categorical_labels(fig, axes):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    records = []
    for ax in axes:
        labels = ax.get_xticklabels()
        boxes = [label.get_window_extent(renderer) for label in labels]
        for i, (a, b) in enumerate(zip(boxes[:-1], boxes[1:])):
            gap_pt = (b.x0 - a.x1) * 72. / fig.dpi
            records.append({'panel': ax.get_label(), 'adjacent_category_pair': [i, i + 1], 'gap_pt': gap_pt})
            assert gap_pt >= .5, records
    js(QA / 'category_label_clearance.json', records)


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    for directory in [DATA, QA]:
        directory.mkdir(parents=True, exist_ok=True)
    bars, heat, protected = prepare()
    configure()
    fig = plt.figure(figsize=(WIDTH_MM / 25.4, HEIGHT_MM / 25.4), dpi=300)
    a = fig.add_axes(rect(13.5, 21., 70., 52.5), label='a')
    b = fig.add_axes(rect(97., 21., 70., 52.5), label='b')
    x = np.arange(4)
    width = .34
    containers, bar_texts = [], []
    for j in [0, 1]:
        centers = x + (j - .5) * width
        container = a.bar(centers, bars[:, j], width=width, color=COLORS[j], edgecolor='#17212B',
                          linewidth=.55, label=DISPLAY_DATASETS[j], zorder=2)
        np.testing.assert_allclose([p.get_height() for p in container.patches], bars[:, j], rtol=0, atol=1e-12)
        labels = a.bar_label(container, labels=[f'{v:.1f}' for v in bars[:, j]], padding=3., fontsize=FONT_PT)
        containers.append(container)
        bar_texts.extend(labels)
    assert sum(len(c.patches) for c in containers) == 8
    a.set(xlim=(-.52, 3.52), ylim=(0, 110), xticks=x, xticklabels=LABELS, yticks=[0, 25, 50, 75, 100])
    a.set_xlabel('Model configuration', labelpad=5.)
    a.set_ylabel('Satisfaction rate (%)', labelpad=5.)
    a.set_title('Training margin satisfaction', fontsize=FONT_PT, fontweight='normal', pad=5.5)
    legend = a.legend(loc='upper left', bbox_to_anchor=(.01, .995), borderaxespad=.2,
                      handlelength=1.5, handletextpad=.65, labelspacing=.4, fontsize=FONT_PT)
    for text in legend.get_texts():
        text.set_fontweight('normal')
    add_panel_label(a, 'a')
    cmap = LinearSegmentedColormap.from_list('Constraint_blue', ['#EFF5FB', '#8EB5DD', '#174E85'], N=256)
    norm = Normalize(vmin=0., vmax=100.)
    image = b.imshow(heat, cmap=cmap, norm=norm, interpolation='nearest', aspect='auto')
    np.testing.assert_array_equal(np.asarray(image.get_array()), heat)
    b.set(xticks=x, xticklabels=LABELS, yticks=np.arange(6), yticklabels=FEATURE_LABELS)
    b.set_xlabel('Model configuration', labelpad=5.)
    b.set_ylabel('Perturbed parameter', labelpad=5.)
    b.set_title('Local demand monotonicity violations', fontsize=FONT_PT, fontweight='normal', pad=5.5)
    b.tick_params(axis='y', pad=4.)
    for boundary in np.arange(.5, 3.5, 1.):
        b.axvline(boundary, color='white', linewidth=.5)
    for boundary in np.arange(.5, 5.5, 1.):
        b.axhline(boundary, color='white', linewidth=.5)
    contrast_records = []
    heat_texts = []
    for row in range(6):
        for col in range(4):
            color, contrast = contrast_text_color(cmap(norm(heat[row, col])))
            assert contrast >= 4.5
            heat_texts.append(b.text(col, row, f'{heat[row, col]:.1f}', ha='center', va='center',
                                     fontsize=FONT_PT, fontweight='normal', color=color))
            contrast_records.append({'row': row, 'column': col, 'text_color': color, 'contrast_ratio': contrast})
    add_panel_label(b, 'b')
    cax = fig.add_axes(rect(169.5, 21., 2.6, 52.5))
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax, ticks=[0, 20, 40, 60, 80, 100])
    cb.ax.set_title('Violation\nrate (%)', fontsize=FONT_PT, fontweight='normal', pad=5.5)
    cb.ax.tick_params(direction='out', labelsize=FONT_PT, width=.55, length=2., pad=2.)
    cb.outline.set_linewidth(.6)
    fig.text(.5, 4. / HEIGHT_MM, '图5.6 CMCRNet截尾约束与局部单调性诊断', ha='center', va='center',
             fontproperties=font_manager.FontProperties(family=['Times New Roman', 'SimSun'], size=10.))
    fig.canvas.draw()
    issues = audit_layout(fig)
    js(QA / 'visual_layout_check.json', {'issues': issues})
    assert not any(level in ['FAIL', 'WARN'] for level, message in issues), issues
    audit_categorical_labels(fig, [a, b])
    require_matplotlib_panel_alignment(fig, axes=[a, b], panel_ids=['a', 'b'], row_groups=[['a', 'b']],
        tolerance_pt=1.5, gutter_tolerance_pt=1.5, require_panel_labels=True, strict=True,
        json_out=QA / 'alignment.json', overlay_svg=QA / 'alignment.svg')
    assert all(abs(t.get_fontsize() - FONT_PT) < 1e-12 for t in [a.title, b.title, cax.title,
        a.xaxis.label, a.yaxis.label, b.xaxis.label, b.yaxis.label, *bar_texts, *heat_texts, *legend.get_texts()])
    js(QA / 'heatmap_text_contrast.json', contrast_records)
    js(QA / 'rendered_data_checks.json', {'bar_count': 8, 'bar_heights_equal_unrounded_source_percentages': True,
        'heatmap_shape': [6, 4], 'all_24_cells_equal_unrounded_source_means': True,
        'all_titles_and_axis_titles_pt': FONT_PT, 'same_plot_area_mm': [70., 52.5],
        'all_configurations_and_six_parameter_directions_retained': True})
    fig.savefig(ROOT / f'{NAME}_preview.png', dpi=300, bbox_inches=None)
    fig.savefig(QA / f'{NAME}.pdf', bbox_inches=None)
    fig.savefig(QA / f'{NAME}.svg', bbox_inches=None)
    buffer = BytesIO()
    fig.savefig(buffer, format='png', dpi=600, bbox_inches=None)
    buffer.seek(0)
    with Image.open(buffer) as raster:
        raster.convert('RGB').save(ROOT / f'{NAME}.tiff', compression='tiff_lzw', dpi=(600, 600))
    with Image.open(ROOT / f'{NAME}.tiff') as raster:
        export = {'size_px': list(raster.size), 'mode': raster.mode, 'dpi': [float(v) for v in raster.info['dpi']],
                  'compression': raster.info.get('compression'), 'width_mm': WIDTH_MM, 'height_mm': HEIGHT_MM}
        assert raster.mode == 'RGB' and raster.info.get('compression') == 'tiff_lzw'
        np.testing.assert_allclose(export['dpi'], [600., 600.], rtol=0, atol=1e-9)
    js(QA / 'export_checks.json', export)
    assert protected == {p: sha(p) for p in protected}
    plt.close(fig)
    print(json.dumps({'TIFF': str(ROOT / f'{NAME}.tiff'), 'bar_count': 8, 'heatmap_cells': 24,
                      'source_files_unchanged': True, 'export': export}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
