"""Render the verified 2.0 MPa fields in the same compact A-version layout."""
from io import BytesIO
from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.cm import ScalarMappable
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch
import numpy as np
import pandas as pd
from PIL import Image

import sys
FIG53_ROOT = Path(__file__).resolve().parent.parent / '第一篇_图5_3_真实数据_散点布局_20261028'
sys.path.insert(0, str(FIG53_ROOT))
import plot_fig5_3 as base
from support.audit_panel_alignment import require_matplotlib_panel_alignment


ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'A_FLAC_contours_compact'
QA = OUT / 'qa'
NAME = 'Fig5_2_A_FLAC_contours_compact'
FIGURE_TITLE = '图5.2 支护压力2.0 MPa下位移与竖向正应力场对比'
WIDTH_MM, HEIGHT_MM = 190., 136.


def main():
    QA.mkdir(parents=True, exist_ok=True)
    paths = [ROOT / 'data/Fig5_2_selected_fields.npz', ROOT / 'data/color_scales.json',
             ROOT / 'data/Fig5_2_case_metrics.csv', ROOT / 'data/SourceData_Fig5_2_grid.csv',
             ROOT / 'A_FLAC_contours/Fig5_2_A_FLAC_contours.tiff', ROOT / 'plot_fig5_2.py']
    protected = {str(path): base.sha(path) for path in paths}
    fields = base.load(paths[0])
    assert fields['case_id'].item() == 'r55_g001_p06' and fields['step'].item() == 5
    assert np.isclose(fields['pressure_MPa'].item(), 2.0)
    scales = json.loads(paths[1].read_text(encoding='utf-8'))
    archived_grid = pd.read_csv(paths[3])
    valid_source = fields['valid_mask']
    assert len(archived_grid) == 5504
    np.testing.assert_array_equal(archived_grid['valid'].to_numpy(), valid_source.ravel())
    old_records = []
    for ri, component in enumerate(base.COMPONENTS):
        for ci, quantity in enumerate(base.QUANTITIES):
            values = fields[quantity][ri]
            unit = scales[ri]['unit']
            archived_values = archived_grid[f'{component}_{quantity}_{unit}'].to_numpy().reshape(64, 86)
            np.testing.assert_allclose(values[valid_source], archived_values[valid_source], rtol=1e-12, atol=1e-12)
            boundaries = scales[ri]['error_boundaries' if ci == 2 else 'raw_boundaries']
            old_records.append({'panel': 'abcdefghi'[ri * 3 + ci], 'component': component, 'quantity': quantity,
                'points': int(valid_source.sum()), 'values_sha256': hashlib.sha256(values[valid_source].tobytes()).hexdigest(),
                'limits': [boundaries[0], boundaries[-1]], 'unit': unit, 'no_clipping': True})
    old_layout = json.loads((FIG53_ROOT / 'compact_vertical/qa/layout_and_data_comparison.json').read_text(encoding='utf-8'))
    base.configure()
    plt.rcParams.update({'pdf.fonttype': 42, 'svg.fonttype': 'none', 'axes.grid': False})
    fig = plt.figure(figsize=(WIDTH_MM / 25.4, HEIGHT_MM / 25.4), dpi=300)
    x, z, valid = fields['x_grid_m'], fields['z_grid_m'], fields['valid_mask']
    assert valid.sum() == 5360
    xe = np.r_[0., (x[:-1] + x[1:]) / 2., 172.]
    ze = np.r_[0., (z[:-1] + z[1:]) / 2., 100.]
    assert np.all(np.diff(xe) > 0) and np.all(np.diff(ze) > 0)
    cmap = LinearSegmentedColormap.from_list('FLAC_style', ['#0000ff', '#00ffff', '#00ff00', '#ffff00', '#ff0000'], N=12)
    cmap.set_bad('white')
    plot_width = 42.
    plot_height = plot_width * 100. / 172.
    axes, records, cell_checks = [], [], []
    for row, bottom in enumerate([93., 57., 21.]):
        for col, left in enumerate([10., 73., 136.]):
            letter = 'abcdefghi'[row * 3 + col]
            ax = fig.add_axes([left / WIDTH_MM, bottom / HEIGHT_MM, plot_width / WIDTH_MM, plot_height / HEIGHT_MM], label=letter)
            cax = fig.add_axes([(left + plot_width + 1.5) / WIDTH_MM, bottom / HEIGHT_MM, 2.1 / WIDTH_MM, plot_height / HEIGHT_MM])
            quantity = base.QUANTITIES[col]
            values = fields[quantity][row]
            scale = scales[row]
            boundaries = scale['error_boundaries' if col == 2 else 'raw_boundaries']
            ticks = scale['error_ticks' if col == 2 else 'raw_ticks']
            assert values[valid].min() >= boundaries[0] and values[valid].max() <= boundaries[-1]
            norm = BoundaryNorm(boundaries, 12, clip=False)
            masked = np.ma.array(values, mask=~valid)
            artist = ScalarMappable(norm=norm, cmap=cmap)
            artist.set_array(masked)
            np.testing.assert_array_equal(artist.get_array().compressed(), values[valid])
            bins = np.ma.asarray(norm(masked)).filled(-1)
            assert np.all((bins[valid] >= 0) & (bins[valid] < 12))
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
            ax.contour(x, z, valid.astype(float), levels=[.5], colors='black', linewidths=.5)
            ax.set(xlim=(0, 172), ylim=(0, 100), aspect='equal', xticks=[0, 40, 80, 120, 172], yticks=[0, 25, 50, 75, 100])
            ax.set_xlabel(r'$x$ (m)', labelpad=2.4)
            ax.set_ylabel(r'Aligned $z$ (m)', labelpad=2.4)
            ax.set_title(f'{base.SYMBOLS[row]} {base.TITLES[col]} ({scale["unit"]})', fontsize=9., pad=5.)
            ax.annotate(f'({letter})', xy=(0, 1), xycoords='axes fraction', xytext=(-8, 5), textcoords='offset points',
                        ha='left', va='bottom', fontsize=10.5, fontweight='bold', annotation_clip=False)
            bar = fig.colorbar(artist, cax=cax, ticks=ticks, spacing='proportional', drawedges=False)
            bar.set_ticks(ticks, labels=[f'{value:g}' for value in ticks])
            bar.minorticks_off()
            bar.ax.tick_params(labelsize=7.3, width=.5, length=2., pad=1.8, direction='in')
            bar.outline.set_linewidth(.5)
            axes.append(ax)
            records.append({'panel': letter, 'component': base.COMPONENTS[row], 'quantity': quantity, 'points': count,
                            'values_sha256': hashlib.sha256(values[valid].tobytes()).hexdigest(),
                            'limits': [boundaries[0], boundaries[-1]], 'unit': scale['unit'], 'no_clipping': True})
            cell_checks.append({'panel': letter, 'cells_drawn': count, 'internal_edges': False, 'interpolation': False})
    for left, title in zip([10., 73., 136.], ['FLAC3D reference', 'Surrogate prediction', 'Absolute error']):
        fig.text((left + plot_width / 2) / WIDTH_MM, 130. / HEIGHT_MM, title,
                 ha='center', va='center', fontsize=10.5, fontweight='bold')
    fig.text(.5, 6. / HEIGHT_MM, FIGURE_TITLE, ha='center', va='center',
             fontproperties=font_manager.FontProperties(family=['Times New Roman', 'SimSun'], size=9.2))
    fig.canvas.draw()
    issues = base.audit_layout(fig)
    base.js(QA / 'visual_qa.json', {'issues': issues})
    assert not any(level == 'FAIL' for level, message in issues), issues
    require_matplotlib_panel_alignment(fig, axes=axes, panel_ids=list('abcdefghi'),
        row_groups=[list('abc'), list('def'), list('ghi')], column_groups=[list('adg'), list('beh'), list('cfi')],
        json_out=QA / 'alignment.json', overlay_svg=QA / 'alignment.svg', tolerance_pt=1.5,
        gutter_tolerance_pt=1.5, require_panel_labels=True, strict=True)
    rectangles = [[ax.get_position().x0 * WIDTH_MM, ax.get_position().y0 * HEIGHT_MM,
                   ax.get_position().width * WIDTH_MM, ax.get_position().height * HEIGHT_MM] for ax in axes]
    np.testing.assert_allclose(rectangles, old_layout['axes_rectangles_mm'], rtol=0, atol=1e-10)
    assert records == old_records
    base.js(QA / 'plotted_field_checks.json', records)
    base.js(QA / 'rendered_cell_checks.json', cell_checks)
    fig.savefig(OUT / f'{NAME}_preview.png', dpi=300, bbox_inches=None)
    fig.savefig(QA / f'{NAME}.pdf', bbox_inches=None)
    buffer = BytesIO()
    fig.savefig(buffer, format='png', dpi=600, bbox_inches=None)
    buffer.seek(0)
    with Image.open(buffer) as raster:
        raster.convert('RGB').save(OUT / f'{NAME}.tiff', compression='tiff_lzw', dpi=(600, 600))
    plt.close(fig)
    assert protected == {str(path): base.sha(path) for path in paths}
    base.js(QA / 'variant_provenance.json', {'variant': 'A_FLAC_contours_compact', 'width_mm': WIDTH_MM, 'height_mm': HEIGHT_MM,
        'case_id': 'r55_g001_p06', 'step': 5, 'pressure_MPa': 2.0, 'points_per_panel': 5360,
        'all_9_panel_values_equal_to_archived_CSV': True, 'geometry_equal_to_Fig5_3_compact_A': True,
        'same_palette_and_color_limits': True, 'internal_grid_lines': False, 'interpolation': False,
        'smoothing': False, 'downsampling': False, 'original_A_version_preserved': True, 'protected_inputs': protected})
    print(json.dumps({'variant': 'A', 'panels': 9, 'cells_per_panel': 5360, 'data_and_geometry_unchanged': True,
                      'TIFF': str(OUT / f'{NAME}.tiff')}, ensure_ascii=True))


if __name__ == '__main__':
    main()
