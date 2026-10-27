from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

from scripts import methods
from scripts.plot_style import PaperPlotStyle


def select_rohe_components(data, k, which_plot='u'):
    key = which_plot.upper()
    if key not in data:
        raise ValueError("which_plot must select an array present in the npz, usually 'u' or 'v'.")
    loc_key = f'loc_{key}_stat'
    if loc_key not in data:
        keep = np.arange(k)
    else:
        localization = np.asarray(data[loc_key], dtype=float) ** 0.25
        keep = np.flatnonzero(localization <= 0.15)[:k]
        if keep.size < k:
            raise ValueError(f'Only found {keep.size} non-localized {key} components; requested k={k}.')
    return keep


def load_selected_factors(path, k, which_plot='u'):
    """Load Rohe-style selected PCA factors from the cached NYTimes decomposition."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f'Missing {path}. Copy nytimes_pca.npz into data/ before running this figure.'
        )

    data = np.load(path)
    key = which_plot.upper()
    keep = select_rohe_components(data, k, which_plot)
    return data[key][:, keep], data, keep


def rohe_leverage_sample_indices(X, n_points=5000, lev_samp=True, seed=0):
    n = X.shape[0]
    rng = np.random.default_rng(seed)
    if n_points >= n:
        return np.arange(n)
    if not lev_samp:
        return rng.choice(n, n_points, replace=False)
    leverage = np.linalg.norm(X, axis=1)
    probabilities = leverage / leverage.sum()
    return rng.choice(n, n_points, replace=False, p=probabilities)


def run_global_gfa_rotation(X, max_iterations=1000, verbosity=0):
    """Run one global WFA/GFA optimization over SO(k)."""
    methods.ensure_optimization_deps()
    k = X.shape[1]
    manifold = methods.manifolds.SpecialOrthogonalGroup(k)
    solver = methods.optimizers.SteepestDescent(
        verbosity=verbosity,
        max_iterations=max_iterations,
        min_gradient_norm=1e-10,
        min_step_size=1e-12,
    )
    problem = methods.pymanopt.Problem(
        manifold=manifold,
        cost=methods.wfa1_cost_factory(manifold, X),
    )
    result = solver.run(problem, initial_point=np.identity(k))
    return np.asarray(result.point), result


def run_global_varimax_rotation(X, max_iterations=1000, verbosity=0):
    """Run one global Varimax rotation with R stats::varimax on PCA factors."""
    methods.ensure_r_varimax_deps()
    varimax_on_factors = methods.ro.r(
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
    res = varimax_on_factors(methods.np_to_r(X))
    rotation = methods.r_to_np(res.rx2('rotmat'))
    result = SimpleNamespace(
        iterations=None,
        gradient_norm=None,
        stopping_criterion='R stats::varimax',
        max_iterations=max_iterations,
        verbosity=verbosity,
    )
    return rotation, result


def orient_components_to_positive_skew(X):
    """Flip component signs so every rotated coordinate has nonnegative skewness."""
    centered = X - np.mean(X, axis=0)
    skewness = np.mean(centered ** 3, axis=0)
    signs = np.where(skewness < 0, -1.0, 1.0)
    return X * signs, signs


def full_component_limits(X):
    """Use the full sampled coordinate range, matching base R pairs behavior."""
    lower = np.min(X, axis=0)
    upper = np.max(X, axis=0)
    padding = 0.04 * np.maximum(upper - lower, np.finfo(float).eps)
    return [(float(lo - pad), float(hi + pad)) for lo, hi, pad in zip(lower, upper, padding)]


def component_localization_scores(X):
    """Kurtosis-like localization score shown on diagonal panels."""
    X_centered = X - np.mean(X, axis=0)
    variance = np.mean(X_centered ** 2, axis=0)
    fourth = np.mean(X_centered ** 4, axis=0)
    scores = fourth / np.maximum(variance ** 2, np.finfo(float).eps)
    return np.rint(scores).astype(int)


def compact_tick_label(value, _position):
    """Short tick labels for dense pairplot axes."""
    if abs(value) < 5e-5:
        return '0'
    return f'{value:.3g}'


def draw_projected_axis(ax, direction, xlim, ylim, color, linewidth, linestyle='-'):
    """Draw a line through the origin clipped to the current panel limits."""
    if np.linalg.norm(direction) <= np.finfo(float).eps:
        return
    direction = direction / np.linalg.norm(direction)
    candidates = []
    if abs(direction[0]) > np.finfo(float).eps:
        candidates.extend([xlim[0] / direction[0], xlim[1] / direction[0]])
    if abs(direction[1]) > np.finfo(float).eps:
        candidates.extend([ylim[0] / direction[1], ylim[1] / direction[1]])
    valid = [
        t for t in candidates
        if xlim[0] <= t * direction[0] <= xlim[1]
        and ylim[0] <= t * direction[1] <= ylim[1]
    ]
    if len(valid) < 2:
        return
    t_min, t_max = min(valid), max(valid)
    ax.plot(
        [t_min * direction[0], t_max * direction[0]],
        [t_min * direction[1], t_max * direction[1]],
        color=color,
        linewidth=linewidth,
        linestyle=linestyle,
        zorder=5,
    )


def configure_pair_axis(ax, i, j, k, xlim, ylim, label_prefix):
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.grid(False)
    ax.xaxis.set_major_formatter(FuncFormatter(compact_tick_label))
    ax.yaxis.set_major_formatter(FuncFormatter(compact_tick_label))
    ax.tick_params(
        axis='both',
        which='major',
        labelsize=10,
        length=2.0,
        width=0.6,
        pad=1.5,
        direction='out',
        top=True,
        right=True,
        labeltop=i == 0,
        labelright=j == k - 1,
        labelbottom=i == k - 1,
        labelleft=j == 0,
    )
    if i == k - 1:
        ax.set_xlabel(rf'${label_prefix}_{{{j + 1}}}$', labelpad=1)
    if j == 0:
        ax.set_ylabel(rf'${label_prefix}_{{{i + 1}}}$', labelpad=1)


def plot_pairplot_block(
    fig,
    gridspec,
    X,
    limits,
    diagonal_scores,
    style,
    label_prefix,
    rotation=None,
    show_pca_axes=False,
    show_rotation_axes=True,
):
    k = X.shape[1]
    point_color = style.palette[3]
    axis_color = style.palette[1]
    pca_axis_color = style.palette[4]
    axis_linewidth = 0.55 * plt.rcParams['lines.linewidth']
    pca_axis_linewidth = 0.42 * plt.rcParams['lines.linewidth']

    for i in range(k):
        for j in range(k):
            ax = fig.add_subplot(gridspec[i, j])
            if i == j:
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.text(
                    0.5,
                    0.5,
                    f'{diagonal_scores[i]}',
                    ha='center',
                    va='center',
                    fontsize=plt.rcParams['axes.labelsize'] + 2,
                    transform=ax.transAxes,
                )
                continue

            ax.scatter(
                X[:, j],
                X[:, i],
                s=0.9,
                alpha=0.55,
                color=point_color,
                edgecolors='none',
                rasterized=True,
            )
            configure_pair_axis(ax, i, j, k, limits[j], limits[i], label_prefix)

            if show_pca_axes:
                ax.axhline(
                    0,
                    color=pca_axis_color,
                    lw=pca_axis_linewidth,
                    linestyle='--',
                    alpha=0.85,
                    zorder=4,
                )
                ax.axvline(
                    0,
                    color=pca_axis_color,
                    lw=pca_axis_linewidth,
                    linestyle='--',
                    alpha=0.85,
                    zorder=4,
                )

            if show_rotation_axes:
                if rotation is None:
                    ax.axhline(0, color=axis_color, lw=axis_linewidth, linestyle='--', zorder=5)
                    ax.axvline(0, color=axis_color, lw=axis_linewidth, linestyle='--', zorder=5)
                else:
                    draw_projected_axis(
                        ax,
                        np.array([rotation[j, j], rotation[i, j]]),
                        limits[j],
                        limits[i],
                        axis_color,
                        axis_linewidth,
                        linestyle='--',
                    )
                    draw_projected_axis(
                        ax,
                        np.array([rotation[j, i], rotation[i, i]]),
                        limits[j],
                        limits[i],
                        axis_color,
                        axis_linewidth,
                        linestyle='--',
                    )
