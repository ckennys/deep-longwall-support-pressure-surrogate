"""Render the verified fields as contiguous cells without internal grid lines."""
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
from PIL import Image

import plot_fig5_4 as base
from support.audit_panel_alignment import require_matplotlib_panel_alignment


ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'no_grid'
QA = OUT / 'qa'
NAME = 'Fig5_4_Multicase_displacement_compact_no_grid'
WIDTH_MM, HEIGHT_MM = base.WIDTH_MM, base.HEIGHT_MM


def main():
    QA.mkdir(parents=True, exist_ok=True)
    paths = [ROOT / 'data/Fig5_4_selected_fields.npz', ROOT / 'data/color_scales.json',
             ROOT / 'data/Fig5_4_case_metrics.csv', ROOT / 'data/SourceData_Fig5_4_grid.csv',
             ROOT / 'Fig5_4_Multicase_displacement_compact.tiff', ROOT / 'plot_fig5_4.py']
    protected = {str(path): base.sha(path) for path in paths}
    original_records = json.loads((ROOT / 'qa/plotted_field_checks.json').read_text(encoding='utf-8'))
    original_layout = json.loads((ROOT / 'qa/layout_dimensions.json').read_text(encoding='utf-8'))
    fields = base.load(paths[0])
    scales = json.loads(paths[1].read_text(encoding='utf-8'))
    base.configure()
    plt.rcParams.update({'pdf.fonttype': 42, 'svg.fonttype': 'none', 'axes.grid': False})
    fig = plt.figure(figsize=(WIDTH_MM / 25.4, HEIGHT_MM / 25.4), dpi=300)
    x, z = fields['x_grid_m'], fields['z_grid_m']
    x_edges = np.r_[0., (x[:-1] + x[1:]) / 2., 172.]
    z_edges = np.r_[0., (z[:-1] + z[1:]) / 2., 100.]
    assert np.all(np.diff(x_edges) > 0) and np.all(np.diff(z_edges) > 0)
    cmap = LinearSegmentedColormap.from_list('FLAC_style', base.PALETTE, N=12)
    cmap.set_bad('white')
    axes, records, representatives, rendered_cells = [], [], {}, []
    for row, bottom in enumerate(base.BOTTOMS_MM):
        ci = row // 2
        quantity = 'reference' if row % 2 == 0 else 'prediction'
        norm = BoundaryNorm(scales[ci]['raw_boundaries'], 12, clip=False)
        for col, left in enumerate(base.LEFTS_MM):
            letter = 'abcdefghijklmnop'[row * 4 + col]
            ax = fig.add_axes([left / WIDTH_MM, bottom / HEIGHT_MM, base.PLOT_WIDTH_MM / WIDTH_MM,
                               base.PLOT_HEIGHT_MM / HEIGHT_MM], label=letter)
            valid = fields['valid_mask'][col]
            values = fields[quantity][col, ci]
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
                vertices = np.stack([np.column_stack([x_edges[ix], z_edges[iz]]),
                                     np.column_stack([x_edges[ix + 1], z_edges[iz]]),
                                     np.column_stack([x_edges[ix + 1], z_edges[iz + 1]]),
                                     np.column_stack([x_edges[ix], z_edges[iz + 1]])], axis=1)
                patch = PathPatch(MplPath.make_compound_path_from_polys(vertices), facecolor=cmap(band),
                                  edgecolor='none', linewidth=0, antialiased=False)
                ax.add_patch(patch)
                count += len(iz)
            assert count == int(valid.sum())
            rendered_cells.append({'panel': letter, 'cells': count, 'internal_edges': False, 'interpolation': False})
            ax.contour(x, z, valid.astype(float), levels=[.5], colors='black', linewidths=.5)
            ax.set(xlim=(0, 172), ylim=(0, 100), aspect='equal', xticks=[0, 40, 80, 120, 172], yticks=[0, 25, 50, 75, 100])
            ax.set_xlabel(r'$x$ (m)', labelpad=2.)
            ax.set_ylabel(r'Aligned $z$ (m)', labelpad=2.)
            ax.annotate(f'({letter})', xy=(0, 1), xycoords='axes fraction', xytext=(-2.5, 3.8), textcoords='offset points',
                        ha='left', va='bottom', fontsize=9.8, fontweight='bold', annotation_clip=False)
            axes.append(ax)
            representatives[ci] = artist
            records.append({'panel': letter, 'row': row + 1, 'column': col + 1, 'case_id': base.CASES[col],
                'role': base.ROLES[col], 'component': base.COMPONENTS[ci], 'quantity': quantity, 'points': count,
                'values_sha256': hashlib.sha256(values[valid].tobytes()).hexdigest(),
                'limits': [scales[ci]['raw_boundaries'][0], scales[ci]['raw_boundaries'][-1]], 'unit': 'm', 'no_clipping': True})
        center = bottom + base.PLOT_HEIGHT_MM / 2
        symbol = r'$u_z$' if ci == 0 else r'$u_x$'
        if quantity == 'reference':
            fig.text(9.7 / WIDTH_MM, (center + 2.2) / HEIGHT_MM, symbol, ha='center', va='center', fontsize=10.)
            fig.text(9.7 / WIDTH_MM, (center - 2.2) / HEIGHT_MM, 'Reference', ha='center', va='center', fontsize=9., fontweight='bold')
        else:
            fig.text(9.7 / WIDTH_MM, (center + 4.0) / HEIGHT_MM, symbol, ha='center', va='center', fontsize=10.)
            fig.text(9.7 / WIDTH_MM, center / HEIGHT_MM, 'Prediction', ha='center', va='center', fontsize=9.)
            fig.text(9.7 / WIDTH_MM, (center - 4.0) / HEIGHT_MM, '(PCA + MLP)', ha='center', va='center', fontsize=8.3)
    for left, role, case, pressure in zip(base.LEFTS_MM, base.ROLES, base.CASES, base.PRESSURES):
        center = (left + base.PLOT_WIDTH_MM / 2) / WIDTH_MM
        fig.text(center, 138.5 / HEIGHT_MM, role, ha='center', va='center', fontsize=10.6, fontweight='bold')
        fig.text(center, 132.3 / HEIGHT_MM, f'{case[4:]}, {pressure:.1f} MPa', ha='center', va='center', fontsize=8.8)
    for ci, (top_row, bottom_row) in enumerate([(0, 1), (2, 3)]):
        bottom = base.BOTTOMS_MM[bottom_row]
        height = base.BOTTOMS_MM[top_row] + base.PLOT_HEIGHT_MM - bottom
        cax = fig.add_axes([198. / WIDTH_MM, bottom / HEIGHT_MM, 2.5 / WIDTH_MM, height / HEIGHT_MM])
        ticks = scales[ci]['raw_ticks']
        bar = fig.colorbar(representatives[ci], cax=cax, ticks=ticks, spacing='proportional', drawedges=False)
        bar.set_ticks(ticks, labels=[f'{v:g}' for v in ticks])
        bar.ax.tick_params(labelsize=7.5, direction='in', length=2., width=.5, pad=2.)
        bar.minorticks_off()
        bar.outline.set_linewidth(.5)
        bar.set_label((r'$u_z$' if ci == 0 else r'$u_x$') + ' (m)', fontsize=9.2, labelpad=4.)
    fig.text(.5, 5. / HEIGHT_MM, base.FIGURE_TITLE, ha='center', va='center',
             fontproperties=font_manager.FontProperties(family=['Times New Roman', 'SimSun'], size=10.))
    fig.canvas.draw()
    issues = base.audit_layout(fig)
    base.js(QA / 'visual_qa.json', {'issues': issues})
    assert not any(level == 'FAIL' for level, message in issues), issues
    require_matplotlib_panel_alignment(fig, axes=axes, panel_ids=list('abcdefghijklmnop'),
        row_groups=[list('abcd'), list('efgh'), list('ijkl'), list('mnop')],
        column_groups=[list('aeim'), list('bfjn'), list('cgko'), list('dhlp')],
        json_out=QA / 'alignment.json', overlay_svg=QA / 'alignment.svg', tolerance_pt=1.5,
        gutter_tolerance_pt=1.5, require_panel_labels=True, strict=True)
    rectangles = [[ax.get_position().x0 * WIDTH_MM, ax.get_position().y0 * HEIGHT_MM,
                   ax.get_position().width * WIDTH_MM, ax.get_position().height * HEIGHT_MM] for ax in axes]
    np.testing.assert_allclose(rectangles, original_layout['axes_rectangles_mm'], rtol=0, atol=1e-10)
    assert records == original_records
    base.js(QA / 'plotted_field_checks.json', records)
    base.js(QA / 'rendered_cell_checks.json', rendered_cells)
    fig.savefig(OUT / f'{NAME}_preview.png', dpi=300, bbox_inches=None)
    fig.savefig(QA / f'{NAME}.pdf', bbox_inches=None)
    buffer = BytesIO()
    fig.savefig(buffer, format='png', dpi=600, bbox_inches=None)
    buffer.seek(0)
    with Image.open(buffer) as image:
        image.convert('RGB').save(OUT / f'{NAME}.tiff', compression='tiff_lzw', dpi=(600, 600))
    plt.close(fig)
    assert protected == {str(path): base.sha(path) for path in paths}
    report = {'variant': 'no_grid', 'data_hashes_and_color_limits_identical_to_scatter_version': True,
              'axes_geometry_identical_to_scatter_version': True, 'width_mm': WIDTH_MM, 'height_mm': HEIGHT_MM,
              'representation': 'contiguous original grid cells grouped by the existing 12 color bands',
              'internal_grid_lines': False, 'interpolation': False, 'smoothing': False, 'downsampling': False,
              'opening_contours_preserved': True, 'original_scatter_version_preserved': True, 'protected_inputs': protected}
    base.js(QA / 'variant_provenance.json', report)
    print(json.dumps({'TIFF': str(OUT / f'{NAME}.tiff'), 'panels': 16,
                      'data_and_geometry_unchanged': True, 'internal_grid_lines': False}, ensure_ascii=True))


if __name__ == '__main__':
    main()
