"""Reconstruct the prespecified development diagnostic from archived field arrays."""
from io import BytesIO
from pathlib import Path
import hashlib
import json
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager, ticker
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch
import numpy as np
import pandas as pd
from PIL import Image

from support.audit_panel_alignment import require_matplotlib_panel_alignment
from support.visual_qa import audit_layout


ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parents[1]
DATA = ROOT / 'data_failure_case'
QA = ROOT / 'qa_failure_case'
NAME = 'Fig5_7_Guard64_displacement_errors_compact_A_failure_case'
WIDTH_MM, HEIGHT_MM = 190., 124.
FONT_PT = 9.
CASE = 's52_g012_p07'
TARGETS = ['uz', 'ux']
SYMBOLS = [r'$u_z$', r'$u_x$']
ERROR_SYMBOLS = [r'$|e_z|$', r'$|e_x|$']
PALETTE = ['#0000ff', '#00ffff', '#00ff00', '#ffff00', '#ff0000']
PLOT_WIDTH_MM = 70.
PLOT_HEIGHT_MM = PLOT_WIDTH_MM * 100. / 172.


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def js(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + '\n', encoding='utf-8')


def configure():
    for family in ['Times New Roman', 'SimSun']:
        font_manager.findfont(font_manager.FontProperties(family=family), fallback_to_default=False)
    plt.rcParams.update({
        'font.family': 'serif', 'font.serif': ['Times New Roman'], 'font.weight': 'normal',
        'font.size': FONT_PT, 'axes.titlesize': FONT_PT, 'axes.labelsize': FONT_PT,
        'axes.titleweight': 'normal', 'axes.labelweight': 'normal', 'axes.linewidth': .6,
        'xtick.labelsize': FONT_PT, 'ytick.labelsize': FONT_PT,
        'xtick.direction': 'in', 'ytick.direction': 'in',
        'xtick.major.size': 2.6, 'ytick.major.size': 2.6,
        'xtick.major.width': .55, 'ytick.major.width': .55,
        'xtick.major.pad': 2., 'ytick.major.pad': 2.,
        'xtick.minor.visible': False, 'ytick.minor.visible': False,
        'axes.grid': False, 'axes.unicode_minus': False,
        'mathtext.fontset': 'custom', 'mathtext.rm': 'Times New Roman',
        'mathtext.it': 'Times New Roman:italic', 'mathtext.bf': 'Times New Roman:bold',
        'mathtext.bfit': 'Times New Roman:italic:bold', 'mathtext.fallback': None,
        'pdf.fonttype': 42, 'svg.fonttype': 'none', 'text.usetex': False,
        'figure.facecolor': 'white', 'axes.facecolor': 'white', 'savefig.facecolor': 'white',
    })


def prepare():
    development = WORKSPACE / '分析/第5步_链路核查与模型修正_20261025/development'
    before = WORKSPACE / '图表/第一篇_第五章_D1_D2_D3初版_20261023/D3/data/Fig07_all_selected_fields.npz'
    audit_csv = WORKSPACE / '分析/第一篇_第五章_PI原文重读与重整_20261027/图5.7_开发空间诊断来源核对.csv'
    palette_source = WORKSPACE / '图表/第一篇_图5_2_空间场_AB_真实数据_20261027/plot_fig5_2_A_compact.py'
    source_paths = [development / '可信候选参考_265快照.npz', before,
        *[development / f'cv/{t}_Guard64_margin00_held_predictions.npz' for t in TARGETS],
        development / '物理组合交叉验证_分组.json', development / '逐快照_系数边界OOF指标.csv',
        audit_csv, palette_source]
    protected = {str(path): sha(path) for path in source_paths}
    prior = json.loads((WORKSPACE / '分析/第一篇_5.5图表配置核对_20261029/图表配置核对.json').read_text(encoding='utf-8'))
    assert protected[str(audit_csv)] == prior['source_sha256'][str(audit_csv)]
    prior_fields = prior['figure_5p7']
    assert prior_fields['case_id'] == CASE and prior_fields['step'] == 5
    assert "['#0000ff', '#00ffff', '#00ff00', '#ffff00', '#ff0000'], N=12" in palette_source.read_text(encoding='utf-8')
    fold_meta = json.loads(source_paths[4].read_text(encoding='utf-8'))
    assert fold_meta['same_group_pressures_and_steps_never_split']
    assert fold_meta['fold_by_group']['g012'] == 0 and len(fold_meta['fold_by_group']) == 21
    scores = pd.read_csv(source_paths[5])
    expected = pd.read_csv(audit_csv).set_index('target')
    refs, predictions_before, predictions_after, metrics = [], [], [], []
    with np.load(source_paths[0], allow_pickle=False) as dev, np.load(before, allow_pickle=False) as old:
        select = np.flatnonzero((dev['case_id'] == CASE) & (dev['step'] == 5))
        assert select.tolist() == [119]
        index = int(select[0])
        original_index = int(dev['snapshot_index'][index])
        assert original_index == 174 and old['case_id'][original_index] == CASE
        assert old['step_index'][original_index] == 5 and dev['physical_group'][index] == 'g012'
        features = dev['features'][index].copy()
        assert abs(features[8] - 2.3) < 1e-6
        valid = dev['valid_mask'][index].reshape(64, 86)
        assert valid.sum() == 5336 and valid.size == 5504
        np.testing.assert_array_equal(old['valid_mask'][original_index], valid)
        x, z = old['x_grid_m'].astype(float), old['z_grid_m'].astype(float)
        np.testing.assert_array_equal(x, np.arange(1., 172., 2.))
        np.testing.assert_allclose(z, np.linspace(.5, 99.5, 64), rtol=0, atol=4e-6)
        for row, target in enumerate(TARGETS):
            ref = dev[target][index].reshape(64, 86).astype(float)
            pred_before = old[target + '_prediction_m'][original_index].astype(float)
            with np.load(development / f'cv/{target}_Guard64_margin00_held_predictions.npz', allow_pickle=False) as held:
                assert len(held['source_rows']) == 53 and len(np.unique(held['source_rows'])) == 53
                assert np.all(dev['step'][held['source_rows']] == 5)
                held_index = np.flatnonzero(held['source_rows'] == index)
                assert held_index.tolist() == [23]
                pred_after = held['prediction'][int(held_index[0])].reshape(64, 86).astype(float)
            selected_score = scores.loc[scores['variant'].eq('Guard64_margin00') & scores['target'].eq(target) & scores['case_id'].eq(CASE)]
            assert len(selected_score) == 1 and int(selected_score.iloc[0]['fold']) == 0
            for variant, prediction in [('before', pred_before), ('after', pred_after)]:
                assert np.isfinite(ref[valid]).all() and np.isfinite(prediction[valid]).all()
                error = np.abs(prediction[valid] - ref[valid])
                assert np.all(error > 0), 'Log scale requires strictly positive measured errors.'
                rmse = float(np.sqrt(np.mean((prediction[valid] - ref[valid]) ** 2)))
                previous = expected.loc[target, 'before_rmse_m' if variant == 'before' else 'guard_oof_rmse_m']
                np.testing.assert_allclose(rmse, previous, rtol=0, atol=1e-12)
                np.testing.assert_allclose(rmse, prior_fields[variant + '_RMSE_m'][row], rtol=0, atol=1e-12)
                if variant == 'after':
                    np.testing.assert_allclose(rmse, selected_score.iloc[0]['rmse_m'], rtol=0, atol=1e-12)
                metrics.append({'target': target, 'variant': variant, 'valid_positions': int(valid.sum()),
                    'RMSE_m': rmse, 'MAE_m': float(error.mean()), 'min_absolute_error_m': float(error.min()),
                    'max_absolute_error_m': float(error.max()), 'zero_absolute_errors': int(np.sum(error == 0))})
            refs.append(ref)
            predictions_before.append(pred_before)
            predictions_after.append(pred_after)
        feature_names = dev['feature_names'].copy()
    arrays = {'reference': np.stack(refs), 'before': np.stack(predictions_before), 'after': np.stack(predictions_after)}
    errors = np.stack([np.abs(arrays[key] - arrays['reference']) for key in ['before', 'after']], axis=1)
    assert errors.shape == (2, 2, 64, 86)
    for array in arrays.values():
        array[:, ~valid] = np.nan
    errors[:, :, ~valid] = np.nan
    scales = []
    for row, target in enumerate(TARGETS):
        values = errors[row, :, valid]
        vmin, vmax = 10. ** np.floor(np.log10(values.min())), 10. ** np.ceil(np.log10(values.max()))
        assert values.min() >= vmin and values.max() <= vmax
        scales.append({'target': target, 'unit': 'm', 'normalization': 'logarithmic',
            'vmin': float(vmin), 'vmax': float(vmax),
            'ticks': np.logspace(np.log10(vmin), np.log10(vmax), int(round(np.log10(vmax / vmin))) + 1).tolist(),
            'bounds_12_intervals': np.geomspace(vmin, vmax, 13).tolist(),
            'valid_min': float(values.min()), 'valid_max': float(values.max()),
            'all_valid_errors_positive': True, 'clipped_values': 0})
    xx, zz = np.meshgrid(x, z)
    source = pd.DataFrame({'grid_index': np.arange(valid.size), 'x_m': xx.ravel(), 'aligned_z_m': zz.ravel(), 'valid': valid.ravel()})
    for row, target in enumerate(TARGETS):
        for key, array in arrays.items():
            source[f'{target}_{key}_m'] = array[row].ravel()
        source[f'{target}_absolute_error_before_m'] = errors[row, 0].ravel()
        source[f'{target}_absolute_error_after_m'] = errors[row, 1].ravel()
    source.to_csv(DATA / 'SourceData_Fig5_7_full_grid.csv', index=False, encoding='utf-8-sig', float_format='%.17g')
    pd.DataFrame(metrics).to_csv(DATA / 'Fig5_7_metrics.csv', index=False, encoding='utf-8-sig', float_format='%.17g')
    np.savez_compressed(DATA / 'Fig5_7_selected_fields.npz', **arrays, absolute_error=errors, valid_mask=valid,
        x_grid_m=x, z_grid_m=z, feature_names=feature_names, features=features, case_id=np.asarray(CASE), step=np.asarray(5), pressure_MPa=np.asarray(2.3))
    js(DATA / 'color_scales.json', scales)
    js(DATA / 'source_provenance.json', {'case_id': CASE, 'pressure_MPa': 2.3, 'step': 5,
        'development_row': 119, 'original_prediction_cache_row': 174, 'held_prediction_row': 23,
        'physical_group': 'g012', 'coefficient_bounds_group_fold': 0, 'number_of_group_folds': 5,
        'pretrained_PCA_and_MLP_frozen': True, 'reference': 'Same large-strain development reference for both variants',
        'independent_test': False, 'valid_positions_per_panel': 5336, 'grid_positions': 5504,
        'old_reference_arrays_used_as_truth': False, 'all_four_RMSEs_match_previous_audits': True,
        'color_source': 'User-requested Figure 5.2 palette with 12 bands',
        'palette': PALETTE, 'smoothing': False, 'interpolation': False, 'subsampling': False,
        'new_model_training': False, 'new_FLAC_runs': 0, 'source_sha256': protected})
    js(QA / 'numeric_verification.json', {'status': 'PASS', 'metrics': metrics, 'scales': scales,
        'all_four_panels_use_same_mask_and_same_reference': True, 'all_5336_valid_positions_retained': True})
    return x, z, valid, errors, metrics, scales, protected


def rect(left, bottom, width, height):
    return [left / WIDTH_MM, bottom / HEIGHT_MM, width / WIDTH_MM, height / HEIGHT_MM]


def draw_cells(ax, values, valid, xe, ze, norm, cmap):
    normalized = np.asarray(norm(values[valid]))
    assert np.all((normalized >= 0) & (normalized < 1))
    bins = np.full(valid.shape, -1, dtype=np.int8)
    bins[valid] = np.floor(normalized * 12).astype(np.int8)
    count = 0
    for band in range(12):
        iz, ix = np.nonzero(valid & (bins == band))
        if not len(iz):
            continue
        vertices = np.stack([np.column_stack([xe[ix], ze[iz]]), np.column_stack([xe[ix + 1], ze[iz]]),
            np.column_stack([xe[ix + 1], ze[iz + 1]]), np.column_stack([xe[ix], ze[iz + 1]])], axis=1)
        ax.add_patch(PathPatch(MplPath.make_compound_path_from_polys(vertices), facecolor=cmap(band),
            edgecolor='none', linewidth=0, antialiased=False))
        count += len(iz)
    assert count == int(valid.sum())
    mappable = ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array(np.ma.array(values, mask=~valid))
    np.testing.assert_array_equal(mappable.get_array().compressed(), values[valid])
    return count


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    for directory in [DATA, QA]:
        directory.mkdir(parents=True, exist_ok=True)
    x, z, valid, errors, metrics, scales, protected = prepare()
    configure()
    fig = plt.figure(figsize=(WIDTH_MM / 25.4, HEIGHT_MM / 25.4), dpi=300)
    xe = np.r_[0., (x[:-1] + x[1:]) / 2., 172.]
    ze = np.r_[0., (z[:-1] + z[1:]) / 2., 100.]
    assert np.all(np.diff(xe) > 0) and np.all(np.diff(ze) > 0)
    cmap = LinearSegmentedColormap.from_list('FLAC_style', PALETTE, N=12)
    cmap.set_bad('white')
    axes, records, text_artists = [], [], []
    for row, bottom in enumerate([72., 16.]):
        norm = LogNorm(vmin=scales[row]['vmin'], vmax=scales[row]['vmax'], clip=False)
        for col, left in enumerate([12., 100.]):
            letter = 'abcd'[row * 2 + col]
            ax = fig.add_axes(rect(left, bottom, PLOT_WIDTH_MM, PLOT_HEIGHT_MM), label=letter)
            values = errors[row, col]
            count = draw_cells(ax, values, valid, xe, ze, norm, cmap)
            ax.contour(x, z, valid.astype(float), levels=[.5], colors='black', linewidths=.55)
            ax.set(xlim=(0, 172), ylim=(0, 100), aspect='equal', xticks=[0, 40, 80, 120, 172], yticks=[0, 25, 50, 75, 100])
            ax.set_xlabel(r'$x$ (m)', labelpad=2.4)
            ax.set_ylabel(r'Aligned $z$ (m)', labelpad=2.4)
            name = ['Before Guard64', 'After Guard64'][col]
            rmse = metrics[row * 2 + col]['RMSE_m']
            ax.set_title(f'{SYMBOLS[row]} {name}, RMSE = {rmse:.4f} m', fontsize=FONT_PT, fontweight='normal', pad=5.)
            ax.annotate(f'({letter})', xy=(0, 1), xycoords='axes fraction', xytext=(-8, 5), textcoords='offset points',
                ha='left', va='bottom', fontsize=FONT_PT, fontweight='bold', annotation_clip=False)
            axes.append(ax)
            text_artists.extend([ax.title, ax.xaxis.label, ax.yaxis.label])
            records.append({'panel': letter, 'target': TARGETS[row], 'variant': ['before', 'after'][col],
                'rendered_cells': count, 'values_sha256': hashlib.sha256(values[valid].tobytes()).hexdigest(),
                'RMSE_m': rmse, 'shared_row_limits': [scales[row]['vmin'], scales[row]['vmax']],
                'no_internal_grid_lines': True, 'all_valid_values_drawn': True, 'clipped_values': 0})
        cax = fig.add_axes(rect(173., bottom, 2.6, PLOT_HEIGHT_MM))
        cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax, ticks=scales[row]['ticks'],
            format=ticker.LogFormatterMathtext(base=10), drawedges=False)
        cb.set_label(f'{ERROR_SYMBOLS[row]} (m, log scale)', fontsize=FONT_PT, labelpad=3.)
        cb.ax.tick_params(labelsize=FONT_PT, length=2., width=.5, pad=2., direction='out')
        cb.minorticks_off()
        cb.outline.set_linewidth(.55)
        if cb.solids is not None:
            cb.solids.set_edgecolor('face')
        text_artists.append(cax.yaxis.label)
    fig.text(.5, 120.5 / HEIGHT_MM, 'Coefficient overshoot case: g012, 2.3 MPa, Step 5',
        ha='center', va='center', fontsize=FONT_PT, fontweight='normal')
    fig.text(.5, 4. / HEIGHT_MM, '图5.7 Guard64系数范围约束前后的位移绝对误差对比', ha='center', va='center',
        fontproperties=font_manager.FontProperties(family=['Times New Roman', 'SimSun'], size=10.))
    fig.canvas.draw()
    issues = audit_layout(fig)
    js(QA / 'visual_layout_check.json', {'issues': issues})
    assert not any(level in ['FAIL', 'WARN'] for level, message in issues), issues
    require_matplotlib_panel_alignment(fig, axes=axes, panel_ids=list('abcd'), row_groups=[list('ab'), list('cd')],
        column_groups=[list('ac'), list('bd')], tolerance_pt=1.5, gutter_tolerance_pt=1.5,
        require_panel_labels=True, strict=True, json_out=QA / 'alignment.json', overlay_svg=QA / 'alignment.svg')
    assert all(t.get_fontsize() == FONT_PT and t.get_fontweight() == 'normal' for t in text_artists)
    js(QA / 'rendered_data_checks.json', records)
    fig.savefig(ROOT / f'{NAME}_preview.png', dpi=300, bbox_inches=None)
    fig.savefig(QA / f'{NAME}.pdf', bbox_inches=None)
    fig.savefig(QA / f'{NAME}.svg', bbox_inches=None)
    buffer = BytesIO()
    fig.savefig(buffer, format='png', dpi=600, bbox_inches=None)
    buffer.seek(0)
    with Image.open(buffer) as raster:
        raster.convert('RGB').save(ROOT / f'{NAME}.tiff', compression='tiff_lzw', dpi=(600, 600))
    with Image.open(ROOT / f'{NAME}.tiff') as raster:
        export = {'pixels': list(raster.size), 'mode': raster.mode, 'dpi': [float(v) for v in raster.info['dpi']],
            'compression': raster.info['compression'], 'dimensions_mm': [WIDTH_MM, HEIGHT_MM]}
        assert raster.mode == 'RGB' and raster.info['compression'] == 'tiff_lzw'
        assert export['dpi'] == [600., 600.]
    js(QA / 'export_checks.json', export)
    assert protected == {path: sha(path) for path in protected}
    plt.close(fig)
    print(json.dumps({'TIFF': str(ROOT / f'{NAME}.tiff'), 'export': export, 'source_files_unchanged': True,
        'panels': 4, 'valid_positions_per_panel': int(valid.sum())}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
