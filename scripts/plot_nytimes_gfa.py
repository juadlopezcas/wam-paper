from pathlib import Path

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np

from scripts import methods
from scripts.plot_style import PaperPlotStyle


def load_factors_data(path, k, which_plot='u', n_points=5000, lev_samp=True, seed=0, origin_percentage=0.05):
    """Load and sample PCA factors from the cached NYTimes decomposition."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f'Missing {path}. Copy nytimes_pca.npz into data/ before running this figure.'
        )

    data = np.load(path)
    key = which_plot.upper()
    if key not in data:
        raise ValueError("which_plot must select an array present in the npz, usually 'u' or 'v'.")

    X = data[key][:, :k]
    n = X.shape[0]
    rng = np.random.default_rng(seed)
    if n_points >= n:
        sample_index = np.arange(n)
    elif lev_samp:
        distances = np.linalg.norm(X, axis=1)
        distances = distances / np.max(distances)
        threshold = np.percentile(distances, origin_percentage * 100)
        probabilities = np.where(distances > threshold, distances, threshold / 10)
        probabilities = probabilities / np.sum(probabilities)
        sample_index = rng.choice(n, n_points, replace=False, p=probabilities)
    else:
        sample_index = rng.choice(n, n_points, replace=False)

    return X[sample_index, :k]


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
    """Run one global Varimax rotation using the shared Varimax loss."""
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
        cost=methods.varimax_loss_factory(manifold, X),
    )
    # X is already represented in PCA coordinates, so the identity rotation
    # initializes Varimax at the principal directions.
    principal_directions = np.identity(k)
    result = solver.run(problem, initial_point=principal_directions)
    return np.asarray(result.point), result


def symmetric_component_limits(X, quantile=0.995):
    """Use robust symmetric limits so outliers do not flatten every panel."""
    limits = np.quantile(np.abs(X), quantile, axis=0)
    limits = np.maximum(limits, np.finfo(float).eps)
    return [(-float(limit), float(limit)) for limit in limits]


def orient_components_to_positive_skew(X):
    """Flip component signs so every rotated coordinate has nonnegative skewness."""
    centered = X - np.mean(X, axis=0)
    skewness = np.mean(centered ** 3, axis=0)
    signs = np.where(skewness < 0, -1.0, 1.0)
    return X * signs, signs


def lower_left_component_limits(X, upper_quantile=0.995, lower_fraction=0.22):
    """Set limits so the origin appears near the lower-left corner."""
    upper = np.quantile(X, upper_quantile, axis=0)
    upper = np.maximum(upper, np.quantile(np.abs(X), upper_quantile, axis=0))
    upper = np.maximum(upper, np.finfo(float).eps)
    return [(-float(lower_fraction * value), float(value)) for value in upper]


def component_localization_scores(X):
    """Kurtosis-like localization score shown on diagonal panels."""
    X_centered = X - np.mean(X, axis=0)
    variance = np.mean(X_centered ** 2, axis=0)
    fourth = np.mean(X_centered ** 4, axis=0)
    scores = fourth / np.maximum(variance ** 2, np.finfo(float).eps)
    return np.rint(scores).astype(int)


def draw_projected_axis(ax, direction, xlim, ylim, color, linewidth):
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
        zorder=5,
    )


def configure_pair_axis(ax, i, j, k, xlim, ylim, label_prefix):
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.grid(False)
    ax.tick_params(
        axis='both',
        which='major',
        labelsize=6,
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
                    fontsize=plt.rcParams['axes.labelsize'],
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
                    ax.axhline(0, color=axis_color, lw=axis_linewidth, zorder=5)
                    ax.axvline(0, color=axis_color, lw=axis_linewidth, zorder=5)
                else:
                    draw_projected_axis(
                        ax,
                        np.array([rotation[j, j], rotation[i, j]]),
                        limits[j],
                        limits[i],
                        axis_color,
                        axis_linewidth,
                    )
                    draw_projected_axis(
                        ax,
                        np.array([rotation[j, i], rotation[i, i]]),
                        limits[j],
                        limits[i],
                        axis_color,
                        axis_linewidth,
                    )


def render_nytimes_pairplot(
    X,
    rotation,
    result,
    style,
    figure_dir,
    figure_stem,
    title,
    flip=True,
):
    """Render side-by-side PCA and rotated NYTimes pairplots."""
    X_rotated = X @ rotation
    if flip:
        X_rotated, component_signs = orient_components_to_positive_skew(X_rotated)
        rotation = rotation * component_signs

    k = X.shape[1]
    pca_limits = symmetric_component_limits(X)
    rotated_limits = lower_left_component_limits(X_rotated)
    pca_scores = component_localization_scores(X)
    rotated_scores = component_localization_scores(X_rotated)

    fig = plt.figure(figsize=(14.5, 6.45))
    fig.suptitle(title, y=0.995, fontsize=plt.rcParams['axes.titlesize'])
    outer = fig.add_gridspec(1, 2, left=0.035, right=0.995, bottom=0.055, top=0.935, wspace=0.20)
    left_grid = outer[0].subgridspec(k, k, wspace=0.12, hspace=0.12)
    right_grid = outer[1].subgridspec(k, k, wspace=0.12, hspace=0.12)

    plot_pairplot_block(
        fig,
        left_grid,
        X,
        pca_limits,
        pca_scores,
        style,
        label_prefix='p',
        rotation=rotation,
        show_pca_axes=True,
        show_rotation_axes=True,
    )
    plot_pairplot_block(
        fig,
        right_grid,
        X_rotated,
        rotated_limits,
        rotated_scores,
        style,
        label_prefix='z',
        rotation=None,
    )

    figure_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_dir / f'{figure_stem}.pdf', bbox_inches='tight')
    fig.savefig(figure_dir / f'{figure_stem}.png', bbox_inches='tight', dpi=220)
    print(f'Saved figures/{figure_stem}.{{pdf,png}}')
    print(f'{title} iterations:', getattr(result, 'iterations', None))
    print(f'{title} gradient norm:', getattr(result, 'gradient_norm', None))
    plt.close(fig)


def plot_nytimes_gfa_pairplot(
    path=Path('data/nytimes_pca.npz'),
    k=7,
    which_plot='u',
    n_points=5000,
    lev_samp=True,
    seed=0,
    max_iterations=1000,
    figure_dir=Path('figures'),
    flip=True,
):
    """Create side-by-side PCA and globally GFA-rotated NYTimes pairplots."""
    style = PaperPlotStyle()
    style.apply()

    X = load_factors_data(path, k, which_plot, n_points, lev_samp, seed=seed)
    rotation, result = run_global_gfa_rotation(X, max_iterations=max_iterations)
    render_nytimes_pairplot(
        X,
        rotation,
        result,
        style,
        figure_dir,
        figure_stem='fig_nytimes_gfa_pairplot',
        title='NYTimes PCA Factors: Global GFA Rotation',
        flip=flip,
    )


def plot_nytimes_varimax_pairplot(
    path=Path('data/nytimes_pca.npz'),
    k=7,
    which_plot='u',
    n_points=5000,
    lev_samp=True,
    seed=0,
    max_iterations=1000,
    figure_dir=Path('figures'),
    flip=True,
):
    """Create side-by-side PCA and globally Varimax-rotated NYTimes pairplots."""
    style = PaperPlotStyle()
    style.apply()

    X = load_factors_data(path, k, which_plot, n_points, lev_samp, seed=seed)
    rotation, result = run_global_varimax_rotation(X, max_iterations=max_iterations)
    render_nytimes_pairplot(
        X,
        rotation,
        result,
        style,
        figure_dir,
        figure_stem='fig_nytimes_varimax_pairplot',
        title='NYTimes PCA Factors: Global Varimax Rotation',
        flip=flip,
    )


def plot_nytimes_pairplots(
    path=Path('data/nytimes_pca.npz'),
    k=7,
    which_plot='u',
    n_points=5000,
    lev_samp=True,
    seed=0,
    max_iterations=1000,
    figure_dir=Path('figures'),
    flip=True,
):
    """Create both GFA and Varimax NYTimes comparison figures."""
    plot_nytimes_gfa_pairplot(path, k, which_plot, n_points, lev_samp, seed, max_iterations, figure_dir, flip)
    plot_nytimes_varimax_pairplot(path, k, which_plot, n_points, lev_samp, seed, max_iterations, figure_dir, flip)



def main():
    plot_nytimes_pairplots()


if __name__ == '__main__':
    main()
