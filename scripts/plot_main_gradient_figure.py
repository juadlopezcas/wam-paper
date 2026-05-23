from pathlib import Path

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, PowerNorm
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
import pymanopt as pymanopt
import pymanopt.manifolds as manifolds
import pymanopt.optimizers as optimizers
from scipy.linalg import expm
from scipy.stats import gaussian_kde

from data.data_utils import DataGenerator
from scripts.methods import wfa1_cost_factory
from scripts.plot_style import PaperPlotStyle


THETA_STAR = -np.pi / 4
THETA_PLOT_MAX = np.pi / 4
THETA_AXIS_MAX = np.pi / 2


def sample_gaussian_mixture_measure(n_samples, seed=None):
    """Sample the 2D Gaussian mixture used by the main gradient figure."""
    if seed is not None:
        np.random.seed(seed)
    return DataGenerator(n_samples, 2).gaussian_factor_analysis_data(main_figure_mixture=True)


def orthonormal_frame(theta):
    """Construct the two frame axes as rows from a rotation angle."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, s], [-s, c]])


def random_orthogonal_frame(dim=2, seed=None):
    """Generate a deterministic random SO(dim) frame via a skew-exponential map."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((dim, dim))
    A = A - A.T
    return expm(A)


def estimate_wfa1_frame_pymanopt(samples, initial_frame):
    """Estimate the empirical minimizer over SO(2)."""
    manifold = manifolds.SpecialOrthogonalGroup(2)
    solver = optimizers.TrustRegions(
        verbosity=0,
        min_step_size=1e-299,
        min_gradient_norm=1e-10,
        max_iterations=2000,
        max_time=5000,
    )
    problem = pymanopt.Problem(manifold=manifold, cost=wfa1_cost_factory(manifold, samples))
    result = solver.run(problem, initial_point=initial_frame.T)
    frame = np.asarray(result.point).T
    theta = float(np.arctan2(frame[0, 1], frame[0, 0]))
    return frame, theta, result


def pi_tick_labels():
    ticks = np.array([
        -np.pi / 2,
        -3 * np.pi / 8,
        -np.pi / 4,
        -np.pi / 8,
        0.0,
        np.pi / 8,
        np.pi / 4,
        3 * np.pi / 8,
        np.pi / 2,
    ])
    labels = [
        r'$-\pi/2$',
        r'$-3\pi/8$',
        r'$-\pi/4$',
        r'$-\pi/8$',
        r'$0$',
        r'$\pi/8$',
        r'$\pi/4$',
        r'$3\pi/8$',
        r'$\pi/2$',
    ]
    return ticks, labels


def empirical_axis_switch_angles(samples, theta_min=-0.5 * np.pi, theta_max=0.5 * np.pi):
    """Angles where an empirical point is equidistant from both frame axes."""
    angles = []
    for point in samples:
        if np.linalg.norm(point) <= np.finfo(float).eps:
            continue
        point_angle = np.arctan2(point[1], point[0])
        for k in range(-3, 4):
            theta = point_angle - np.pi / 4 + k * np.pi / 2
            if theta_min <= theta <= theta_max:
                angles.append(theta)
    return np.array(sorted(angles))


def boundary_extent_for_direction(vec, x_min, x_max, y_min, y_max):
    """Distance from the origin to the square boundary along vec."""
    candidates = []
    if abs(vec[0]) > np.finfo(float).eps:
        candidates.extend([x_min / vec[0], x_max / vec[0]])
    if abs(vec[1]) > np.finfo(float).eps:
        candidates.extend([y_min / vec[1], y_max / vec[1]])
    valid = [
        t for t in candidates
        if t > 0
        and x_min <= t * vec[0] <= x_max
        and y_min <= t * vec[1] <= y_max
    ]
    return min(valid) if valid else 0.0


def draw_frame_arrows(ax, frame, color, x_min, x_max, y_min, y_max, linestyle='-', alpha=1.0, zorder=4):
    """Draw frame axes as double-ended arrows touching the square boundary."""
    for vec in frame:
        vec = vec / np.linalg.norm(vec)
        for sign in (-1.0, 1.0):
            direction = sign * vec
            line_extent = boundary_extent_for_direction(direction, x_min, x_max, y_min, y_max)
            end = line_extent * direction
            ax.annotate(
                '',
                xy=(end[0], end[1]),
                xytext=(0.0, 0.0),
                arrowprops={
                    'arrowstyle': '-|>',
                    'color': color,
                    'lw': plt.rcParams['lines.linewidth'],
                    'linestyle': linestyle,
                    'alpha': alpha,
                    'mutation_scale': 11,
                    'shrinkA': 0,
                    'shrinkB': 0,
                },
                zorder=zorder,
            )


def panel_population_density_orange(ax, samples, style, axis_limit=2.5):
    """Top-left panel: orange heat map of the population measure."""
    x_min = y_min = -axis_limit
    x_max = y_max = axis_limit

    grid_x, grid_y = np.meshgrid(
        np.linspace(x_min, x_max, 220),
        np.linspace(y_min, y_max, 220),
    )
    kde = gaussian_kde(samples.T, bw_method=0.20)
    density = kde(np.vstack([grid_x.ravel(), grid_y.ravel()])).reshape(grid_x.shape)

    density_cmap = LinearSegmentedColormap.from_list(
        'density_orange', ['#ffffff', '#ffe3c7', '#ffb474', style.palette[1], '#8c2d04']
    )
    density_cmap.set_bad('#ffffff')
    density_floor = np.percentile(density, 15)
    density_visible = np.ma.masked_less(density, density_floor)
    ax.imshow(
        density_visible,
        origin='lower',
        extent=[x_min, x_max, y_min, y_max],
        cmap=density_cmap,
        norm=PowerNorm(gamma=0.35),
        aspect='equal',
        alpha=0.98,
    )

    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_box_aspect(1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)


def panel_empirical_scatter_arrows(ax, samples, population_frame, estimated_frame, style, axis_limit=2.5):
    """Bottom-left panel: sample cloud with arrow-ended population and empirical minimizers."""
    x_min = y_min = -axis_limit
    x_max = y_max = axis_limit
    point_cmap = LinearSegmentedColormap.from_list(
        'sample_orange', ['#8c2d04', style.palette[1], '#ffb474', '#ffe3c7']
    )
    radii = np.linalg.norm(samples, axis=1)
    color_values = radii / max(np.max(radii), np.finfo(float).eps)
    ax.scatter(
        samples[:, 0],
        samples[:, 1],
        s=16,
        c=color_values,
        cmap=point_cmap,
        vmin=0.0,
        vmax=1.0,
        alpha=0.72,
        marker=style.markers[0],
        edgecolors='none',
        zorder=2,
    )

    draw_frame_arrows(
        ax,
        population_frame,
        color=style.palette[4],
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        linestyle='-',
        alpha=0.82,
        zorder=3,
    )
    draw_frame_arrows(
        ax,
        estimated_frame,
        color=style.palette[2],
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        linestyle='--',
        alpha=0.98,
        zorder=4,
    )

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_box_aspect(1)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(True, which='major', ls=':', lw=0.45, alpha=0.35)
    ax.legend(
        handles=[
            Line2D(
                [0],
                [0],
                color=style.palette[4],
                linestyle='-',
                linewidth=plt.rcParams['lines.linewidth'],
                label='Population minimizer',
            ),
            Line2D(
                [0],
                [0],
                color=style.palette[2],
                linestyle='--',
                linewidth=plt.rcParams['lines.linewidth'],
                label='Empirical minimizer',
            ),
        ],
        loc='upper center',
        bbox_to_anchor=(0.5, -0.105),
        ncol=1,
        frameon=True,
        framealpha=0.92,
        edgecolor='0.8',
        borderpad=0.25,
        handlelength=1.35,
        handletextpad=0.4,
        labelspacing=0.25,
    )


def frame_columns(theta):
    """Return the two columns of U(theta)."""
    c, s = np.cos(theta), np.sin(theta)
    u1 = np.array([c, s])
    u2 = np.array([-s, c])
    u1_prime = np.array([-s, c])
    u2_prime = np.array([-c, -s])
    return u1, u2, u1_prime, u2_prime


def empirical_subgradient(samples, theta):
    """Scalar derivative/subgradient of the empirical WFA1 risk at theta."""
    u1, u2, u1_prime, u2_prime = frame_columns(theta)
    proj1 = samples @ u1
    proj2 = samples @ u2
    deriv1 = -2.0 * proj1 * (samples @ u1_prime)
    deriv2 = -2.0 * proj2 * (samples @ u2_prime)
    active_first = np.abs(proj1) >= np.abs(proj2)
    return float(np.mean(np.where(active_first, deriv1, deriv2)))


def gradient_curve(samples, theta_grid):
    return np.array([empirical_subgradient(samples, theta) for theta in theta_grid])


def empirical_jump_segments(theta_grid, values, switch_angles, theta_min=None, theta_max=None):
    """Return vertical segments showing left/right empirical jumps at switches."""
    jumps = []
    for switch in switch_angles:
        if theta_min is not None and switch < theta_min:
            continue
        if theta_max is not None and switch > theta_max:
            continue
        idx = np.searchsorted(theta_grid, switch)
        if 0 < idx < values.size:
            left_value = values[idx - 1]
            right_value = values[idx]
            if np.isfinite(left_value) and np.isfinite(right_value):
                jumps.append((switch, left_value, right_value))
    return jumps


def draw_empirical_jump_segments(ax, theta_grid, values, switch_angles, color, theta_min=None, theta_max=None):
    """Draw translucent vertical empirical jump segments at switch locations."""
    for switch, y_left, y_right in empirical_jump_segments(
        theta_grid,
        values,
        switch_angles,
        theta_min=theta_min,
        theta_max=theta_max,
    ):
        ax.vlines(
            switch,
            y_left,
            y_right,
            color=color,
            linewidth=0.75 * plt.rcParams['lines.linewidth'],
            alpha=0.20,
            zorder=3,
        )


def plot_empirical_continuity_intervals(
    ax,
    theta_grid,
    values,
    switch_angles,
    color,
    theta_min,
    theta_max,
    label=None,
):
    """Plot empirical subgradient only on continuity intervals."""
    switches = np.sort(switch_angles[(switch_angles > theta_min) & (switch_angles < theta_max)])
    interval_edges = np.concatenate([[theta_min], switches, [theta_max]])
    first_interval = True
    for left, right in zip(interval_edges[:-1], interval_edges[1:]):
        mask = (theta_grid >= left) & (theta_grid <= right)
        if np.count_nonzero(mask) < 2:
            continue
        ax.plot(
            theta_grid[mask],
            values[mask],
            color=color,
            linewidth=plt.rcParams['lines.linewidth'],
            alpha=0.95,
            linestyle='-',
            label=label if first_interval else None,
        )
        first_interval = False


def add_gradient_zoom_inset(
    ax,
    theta_grid,
    population_gradient,
    empirical_gradient,
    switch_angles,
    style,
):
    """Add a linked inset around theta* for the gradient curves."""
    half_width = np.pi / 10
    x_min = THETA_STAR - half_width
    x_max = THETA_STAR + half_width
    mask = (theta_grid >= x_min) & (theta_grid <= x_max)
    zoom_values = np.concatenate([population_gradient[mask], empirical_gradient[mask]])
    if zoom_values.size == 0:
        return

    value_range = np.ptp(zoom_values)
    y_pad = 0.065 * value_range if value_range > 0 else 0.04 * max(abs(zoom_values[0]), 1.0)
    y_min = float(np.min(zoom_values) - y_pad)
    y_max = float(np.max(zoom_values) + y_pad)
    if y_min > 0:
        y_min = -0.05 * y_max
    if y_max < 0:
        y_max = -0.05 * y_min

    left, bottom, width = 0.705, 0.55, 0.33
    axis_x_min, axis_x_max = ax.get_xlim()
    axis_y_min, axis_y_max = ax.get_ylim()
    region_aspect = ((x_max - x_min) / (axis_x_max - axis_x_min)) / (
        (y_max - y_min) / (axis_y_max - axis_y_min)
    )
    height = width / region_aspect
    if bottom + height > 0.98:
        bottom = 0.98 - height
    if bottom < 0.05:
        bottom = 0.05

    inset = ax.inset_axes((left, bottom, width, height))
    inset.plot(
        theta_grid,
        population_gradient,
        color=style.palette[4],
        linewidth=plt.rcParams['lines.linewidth'],
    )
    switches_in_window = switch_angles[(switch_angles >= x_min) & (switch_angles <= x_max)]
    plot_empirical_continuity_intervals(
        inset,
        theta_grid,
        empirical_gradient,
        switches_in_window,
        style.palette[2],
        theta_min=x_min,
        theta_max=x_max,
    )
    draw_empirical_jump_segments(
        inset,
        theta_grid,
        empirical_gradient,
        switches_in_window,
        style.palette[2],
        theta_min=x_min,
        theta_max=x_max,
    )
    inset.axhline(0.0, color='0.25', linestyle='--', linewidth=0.7 * plt.rcParams['axes.linewidth'], alpha=0.85)
    inset.axvline(THETA_STAR, color='0.25', linestyle='--', linewidth=0.7 * plt.rcParams['axes.linewidth'], alpha=0.85)

    if switches_in_window.size:
        inset.vlines(
            switches_in_window,
            y_min,
            y_min + 0.10 * (y_max - y_min),
            color=style.palette[1],
            linewidth=0.8,
            alpha=0.48,
            zorder=1,
        )

    inset.set_xlim(x_min, x_max)
    inset.set_ylim(y_min, y_max)
    inset.grid(True, which='major', ls=':', lw=0.4, alpha=0.45)
    inset.tick_params(length=2.0, width=0.7, labelsize=max(10, plt.rcParams['xtick.labelsize'] - 2))
    inset.tick_params(axis='y', labelleft=False)

    ticks, labels = pi_tick_labels()
    tick_mask = (ticks >= x_min) & (ticks <= x_max) & ~np.isclose(ticks, THETA_STAR)
    inset.set_xticks(ticks[tick_mask])
    inset.set_xticklabels([label for label, keep in zip(labels, tick_mask) if keep])

    mark_inset(ax, inset, loc1=2, loc2=4, fc='none', ec='0.45', lw=0.8, alpha=0.85)


def panel_gradients(
    ax,
    theta_grid,
    population_gradient,
    empirical_gradient,
    empirical_samples,
    style,
):
    """Right panel: population gradient and empirical subgradient."""
    plot_mask = theta_grid <= THETA_PLOT_MAX
    switch_angles = empirical_axis_switch_angles(empirical_samples)
    switch_angles = switch_angles[switch_angles <= THETA_PLOT_MAX]
    ax.plot(
        theta_grid[plot_mask],
        population_gradient[plot_mask],
        color=style.palette[4],
        linewidth=plt.rcParams['lines.linewidth'],
        label='Population gradient',
    )
    plot_empirical_continuity_intervals(
        ax,
        theta_grid,
        empirical_gradient,
        switch_angles,
        style.palette[2],
        theta_min=-0.5 * np.pi,
        theta_max=THETA_PLOT_MAX,
        label='Empirical subgradient',
    )

    ax.set_xlim(-0.5 * np.pi, THETA_AXIS_MAX)
    ax.axvspan(THETA_PLOT_MAX, THETA_AXIS_MAX, color='white', zorder=0)
    ax.axhline(0.0, color='0.25', linestyle='--', linewidth=plt.rcParams['axes.linewidth'], alpha=0.85)
    ax.axvline(THETA_STAR, color='0.25', linestyle='--', linewidth=plt.rcParams['axes.linewidth'], alpha=0.85)

    y_min, y_max = ax.get_ylim()
    draw_empirical_jump_segments(
        ax,
        theta_grid,
        empirical_gradient,
        switch_angles,
        style.palette[2],
        theta_min=-0.5 * np.pi,
        theta_max=THETA_PLOT_MAX,
    )
    if switch_angles.size:
        ax.vlines(
            switch_angles,
            y_min,
            y_min + 0.065 * (y_max - y_min),
            color=style.palette[1],
            linewidth=0.8,
            alpha=0.45,
            label='Axis switches',
            zorder=1,
        )
        ax.set_ylim(y_min, y_max)

    ticks, labels = pi_tick_labels()
    tick_mask = ticks <= THETA_PLOT_MAX
    ax.set_xticks(ticks[tick_mask])
    ax.set_xticklabels([label for label, keep in zip(labels, tick_mask) if keep])
    ax.set_xlabel(r'Rotation angle $\theta$', labelpad=6)
    ax.set_ylabel('Gradient value', labelpad=7)
    ax.grid(True, which='major', ls=':', lw=0.5, alpha=0.6)

    add_gradient_zoom_inset(
        ax,
        theta_grid,
        population_gradient,
        empirical_gradient,
        switch_angles,
        style,
    )

    ax.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, -0.145),
        ncol=3,
        frameon=True,
        framealpha=0.9,
        edgecolor='0.8',
        borderpad=0.25,
        handlelength=1.35,
        handletextpad=0.4,
        labelspacing=0.25,
    )


def create_main_gradient_figure(
    n_population=100000,
    n_empirical=55,
    seed_density=7,
    seed_population_gradient=17,
    seed_empirical=31,
    seed_initialization=101,
    figure_dir=Path('figures'),
):
    """Create the alternative main figure showing gradients/subgradients."""
    style = PaperPlotStyle()
    style.apply()

    density_samples = sample_gaussian_mixture_measure(n_population, seed=seed_density)
    population_gradient_samples = sample_gaussian_mixture_measure(
        n_population,
        seed=seed_population_gradient,
    )

    rng_empirical = np.random.default_rng(seed_empirical)
    empirical_indices = rng_empirical.choice(density_samples.shape[0], size=n_empirical, replace=False)
    empirical_samples = density_samples[empirical_indices]

    theta_grid = np.linspace(-0.5 * np.pi, THETA_AXIS_MAX, 2001)
    population_gradient = gradient_curve(population_gradient_samples, theta_grid)
    empirical_gradient = gradient_curve(empirical_samples, theta_grid)

    true_frame = orthonormal_frame(THETA_STAR)
    initial_frame = random_orthogonal_frame(dim=2, seed=seed_initialization)
    estimated_frame, est_theta, opt_result = estimate_wfa1_frame_pymanopt(
        empirical_samples,
        initial_frame=initial_frame,
    )

    fig = plt.figure(figsize=(10.4, 5.0))
    left_panel_size = 0.205
    left_panel_height = left_panel_size * (10.4 / 5.0)
    left_x = 0.025
    ax_density = fig.add_axes([left_x, 0.545, left_panel_size, left_panel_height])
    ax_empirical = fig.add_axes([left_x, 0.095, left_panel_size, left_panel_height])
    ax_gradient = fig.add_axes([0.320, 0.135, 0.660, 0.785])

    panel_population_density_orange(ax_density, density_samples, style)
    panel_empirical_scatter_arrows(ax_empirical, empirical_samples, true_frame, estimated_frame, style)
    panel_gradients(
        ax_gradient,
        theta_grid,
        population_gradient,
        empirical_gradient,
        empirical_samples,
        style,
    )

    for ax in (ax_density, ax_empirical, ax_gradient):
        ax.tick_params(length=3.0, width=0.8)

    figure_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_dir / 'fig_main_gradient.pdf', bbox_inches='tight', pad_inches=0.04)
    fig.savefig(figure_dir / 'fig_main_gradient.png', bbox_inches='tight', pad_inches=0.04)
    print('Saved figures/fig_main_gradient.{pdf,png}')
    print('  - pymanopt U_n^* angle (deg): {:.2f}'.format(np.degrees(est_theta)))
    print('  - pymanopt iterations:        {}'.format(opt_result.iterations))
    print('  - pymanopt gradient norm:     {:.3e}'.format(opt_result.gradient_norm))


def main():
    create_main_gradient_figure()


if __name__ == '__main__':
    main()
