from pathlib import Path

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA

from data.data_utils import DataGenerator
from scripts import methods
from scripts.methods import run_wfa1
from scripts.plot_style import PaperPlotStyle


def reference_rotation():
    """Reference rotation used by the original double-exponential visualization."""
    return np.linalg.svd([[1, -2], [-3, 1]])[0]


def generate_rotated_exponential_data(n_samples=5000, seed=42):
    """Generate 2D exponential data aligned to the reference frame."""
    np.random.seed(seed)
    samples = DataGenerator(n_samples, 2).exponential_factor_analysis_data(apply_reference_rotation=False)
    return samples @ reference_rotation()

def run_pca_factor_varimax(data, pca):
    """Apply R stats::varimax to PCA factor scores and return original-space frame columns."""
    methods.ensure_r_varimax_deps()
    factor_scores = pca.transform(data)
    score_varimax = methods.ro.r(
        """
        function(factors) {
          vm <- stats::varimax(as.matrix(factors), normalize=FALSE, eps=1e-5)
          list(
            rotated = unclass(vm$loadings),
            rotmat = unclass(vm$rotmat)
          )
        }
        """
    )
    res = score_varimax(methods.np_to_r(factor_scores))
    rotmat = methods.r_to_np(res.rx2('rotmat'))
    return pca.components_.T @ rotmat


def orient_frame_columns(frame):
    """Fix signs deterministically so frame arrows point to the upper half-plane when possible."""
    frame = np.asarray(frame, dtype=float).copy()
    for j in range(frame.shape[1]):
        col = frame[:, j]
        if col[1] < 0 or (np.isclose(col[1], 0.0) and col[0] < 0):
            frame[:, j] = -col
    return frame


def boundary_extent_for_direction(direction, axis_limit):
    direction = np.asarray(direction, dtype=float)
    nonzero = np.abs(direction) > np.finfo(float).eps
    return float(np.min(axis_limit / np.abs(direction[nonzero])))


def draw_true_cross(ax, frame, color, axis_limit, linewidth, zorder):
    """Draw the full sign-symmetric reference frame as a translucent cross."""
    for j in range(frame.shape[1]):
        direction = frame[:, j] / np.linalg.norm(frame[:, j])
        length = boundary_extent_for_direction(direction, axis_limit)
        endpoint = length * direction
        ax.plot(
            [-endpoint[0], endpoint[0]],
            [-endpoint[1], endpoint[1]],
            color=color,
            linewidth=linewidth,
            linestyle='-',
            alpha=0.24,
            solid_capstyle='round',
            zorder=zorder,
        )


def draw_frame(ax, frame, color, linestyle, marker, label, axis_limit, linewidth, zorder):
    """Draw the two frame directions as arrows from the origin to the square boundary."""
    frame = orient_frame_columns(frame)
    for j in range(frame.shape[1]):
        direction = frame[:, j] / np.linalg.norm(frame[:, j])
        length = 0.88 * boundary_extent_for_direction(direction, axis_limit)
        endpoint = length * direction
        ax.annotate(
            '',
            xy=endpoint,
            xytext=(0.0, 0.0),
            arrowprops={
                'arrowstyle': '-|>',
                'color': color,
                'lw': linewidth,
                'linestyle': linestyle,
                'mutation_scale': 16,
                'shrinkA': 0,
                'shrinkB': 0,
            },
            zorder=zorder,
        )
        ax.plot(
            endpoint[0],
            endpoint[1],
            marker=marker,
            color=color,
            markersize=plt.rcParams['lines.markersize'] + 1.5,
            linestyle='None',
            zorder=zorder + 0.1,
        )


def plot_exponential_frames(
    n_samples=200,
    seed=42,
    max_iterations=100,
    figure_dir=Path('figures'),
):
    """Plot PCA, R PCA+Varimax, and WFA frames on one rotated exponential cloud."""
    style = PaperPlotStyle()
    style.apply()

    data = generate_rotated_exponential_data(n_samples=n_samples, seed=seed)

    pca = PCA(n_components=2)
    pca.fit(data)
    pca_frame = pca.components_.T

    try:
        pca_varimax_frame = run_pca_factor_varimax(data, pca)
    except ImportError as exc:
        raise ImportError(
            'The exponential-frame figure requires the R-based PCA+Varimax wrapper. '
            'Install R and rpy2 in the active environment, then rerun this script.'
        ) from exc

    wfa_frame, _, _, _ = run_wfa1(data, 2, initial_point=pca_frame, max_iterations=max_iterations)

    fig, ax = plt.subplots(figsize=(7.2, 6.4))

    point_cmap = LinearSegmentedColormap.from_list(
        'sample_orange', ['#8c2d04', style.palette[1], '#ffb474', '#ffe3c7']
    )
    radii = np.linalg.norm(data, axis=1)
    color_values = radii / max(np.max(radii), np.finfo(float).eps)
    ax.scatter(
        data[:, 0],
        data[:, 1],
        s=13,
        c=color_values,
        cmap=point_cmap,
        vmin=0.0,
        vmax=1.0,
        alpha=0.62,
        marker=style.markers[0],
        edgecolors='none',
        zorder=1,
    )

    axis_limit = 4.0
    true_frame = reference_rotation().T
    true_color = style.palette[2]
    draw_true_cross(
        ax,
        true_frame,
        true_color,
        axis_limit,
        linewidth=1.15 * plt.rcParams['lines.linewidth'],
        zorder=2,
    )

    frame_specs = [
        {
            'label': 'PCA',
            'frame': pca_frame,
            'color': style.palette[4],
            'linestyle': ':',
            'marker': 's',
            'linewidth': 0.82 * plt.rcParams['lines.linewidth'],
            'zorder': 3,
        },
        {
            'label': 'PCA+Varimax',
            'frame': pca_varimax_frame,
            'color': style.method_meta['pca_varimax']['color'],
            'linestyle': '--',
            'marker': style.method_meta['pca_varimax']['marker'],
            'linewidth': 1.18 * plt.rcParams['lines.linewidth'],
            'zorder': 6,
        },
        {
            'label': 'Ours',
            'frame': wfa_frame,
            'color': style.method_meta['wfa1_manifold_optimization']['color'],
            'linestyle': '-',
            'marker': style.method_meta['wfa1_manifold_optimization']['marker'],
            'linewidth': 1.08 * plt.rcParams['lines.linewidth'],
            'zorder': 5,
        },
    ]
    for spec in frame_specs:
        draw_frame(
            ax,
            spec['frame'],
            spec['color'],
            spec['linestyle'],
            spec['marker'],
            spec['label'],
            axis_limit,
            spec['linewidth'],
            spec['zorder'],
        )

    ax.scatter([0.0], [0.0], s=42, color='0.12', zorder=6)
    ax.set_xlim(-axis_limit, axis_limit)
    ax.set_ylim(-axis_limit, axis_limit)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.grid(True, which='major', ls=':', lw=0.5, alpha=0.55)
    ax.minorticks_on()
    ax.tick_params(direction='in', top=True, right=True)

    handles = [
        Line2D(
            [0],
            [0],
            color='none',
            marker=style.markers[0],
            markerfacecolor=style.palette[1],
            markeredgecolor='none',
            markersize=plt.rcParams['lines.markersize'],
            label='Data',
        ),
        Line2D(
            [0],
            [0],
            color=true_color,
            linestyle='-',
            linewidth=1.15 * plt.rcParams['lines.linewidth'],
            alpha=0.34,
            label='True frame',
        ),
    ]
    handles.extend([
        Line2D(
            [0],
            [0],
            color=spec['color'],
            linestyle=spec['linestyle'],
            marker=spec['marker'],
            linewidth=spec['linewidth'],
            markersize=plt.rcParams['lines.markersize'],
            label=spec['label'],
        )
        for spec in frame_specs
    ])
    ax.legend(
        handles=handles,
        loc='upper center',
        bbox_to_anchor=(0.5, -0.075),
        ncol=5,
        frameon=True,
        framealpha=0.9,
        edgecolor='0.8',
        borderpad=0.25,
        handlelength=1.7,
        handletextpad=0.45,
        columnspacing=0.75,
    )

    figure_dir = Path(figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_dir / 'fig_exponential_frames.pdf', bbox_inches='tight', pad_inches=0.04)
    fig.savefig(figure_dir / 'fig_exponential_frames.png', bbox_inches='tight', pad_inches=0.04)
    plt.close(fig)
    print('Saved figures/fig_exponential_frames.{pdf,png}')


if __name__ == '__main__':
    plot_exponential_frames()
