from pathlib import Path

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
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


def true_frame_gaussian_mixture():
    """Return the population-optimal frame for the specified 2D Gaussian mixture."""
    return orthonormal_frame(np.pi / 4.0)


def sample_gaussian_mixture_measure(n_samples, seed=None):
    """Sample DataGenerator's 2D Gaussian mixture."""
    if seed is not None:
        np.random.seed(seed)
    return DataGenerator(n_samples, 2).gaussian_factor_analysis_data()



def sample_double_exponential_measure(n_samples, rotation_angle=np.pi / 4.0, seed=None):
    """Sample DataGenerator's 2D double-exponential data and rotate it."""
    if seed is not None:
        np.random.seed(seed)
    samples = DataGenerator(n_samples, 2).exponential_factor_analysis_data()
    return samples @ orthonormal_frame(rotation_angle)


def sample_student_t_measure(n_samples, rotation_angle=np.pi / 4.0, seed=None):
    """Sample DataGenerator's 2D Student-t data and rotate it."""
    if seed is not None:
        np.random.seed(seed)
    samples = DataGenerator(n_samples, 2).student_t_data()
    return samples @ orthonormal_frame(rotation_angle)

def orthonormal_frame(theta):
    """Construct a 2D orthonormal frame from a rotation angle."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, s], [-s, c]])


def random_orthogonal_frame(dim=2, seed=None):
    """Generate a deterministic random SO(dim) frame via a skew-exponential map."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((dim, dim))
    A = A - A.T
    return expm(A)


def objective_for_frame(samples, frame):
    """Evaluate the WFA1 objective for a given frame on data samples."""
    norms_sq = np.sum(samples * samples, axis=1)
    proj_sq = np.max(np.abs(samples @ frame.T) ** 2, axis=1)
    return np.mean(norms_sq - proj_sq)


def objective_curve(samples, theta_grid):
    """Evaluate objective values over a grid of frame rotation angles."""
    values = np.empty_like(theta_grid)
    for i, theta in enumerate(theta_grid):
        values[i] = objective_for_frame(samples, orthonormal_frame(theta))
    return values


def estimate_wfa1_frame_pymanopt(samples, initial_frame):
    """Estimate U_n^* via pymanopt optimization over SO(2), as in main.py."""
    manifold = manifolds.SpecialOrthogonalGroup(2)
    solver = optimizers.TrustRegions(
        verbosity=0,
        min_step_size=1e-299,
        min_gradient_norm=1e-10,
        max_iterations=2000,
        max_time=5000,
    )
    # The shared cost treats frame axes as columns; this script draws and
    # evaluates frames as rows, so we transpose at the interface.
    problem = pymanopt.Problem(manifold=manifold, cost=wfa1_cost_factory(manifold, samples))
    result = solver.run(problem, initial_point=initial_frame.T)
    frame = np.asarray(result.point).T
    theta = float(np.arctan2(frame[0, 1], frame[0, 0]))
    return frame, theta, result


def draw_frame(ax, frame, color_a, color_b, line_extent, linestyle='-'):
    """Draw two orthonormal frame axes through the origin."""
    for vec, color in zip(frame, [color_a, color_b]):
        line = np.vstack([-line_extent * vec, line_extent * vec])
        ax.plot(line[:, 0], line[:, 1], linestyle=linestyle, color=color, linewidth=plt.rcParams['lines.linewidth'])


def panel_population_density(ax, samples, style, axis_limit=2.5):
    """Top-left panel: smooth heat map of the population measure."""
    x_min = y_min = -axis_limit
    x_max = y_max = axis_limit

    grid_x, grid_y = np.meshgrid(
        np.linspace(x_min, x_max, 220),
        np.linspace(y_min, y_max, 220),
    )
    kde = gaussian_kde(samples.T, bw_method=0.20)
    density = kde(np.vstack([grid_x.ravel(), grid_y.ravel()])).reshape(grid_x.shape)

    density_cmap = LinearSegmentedColormap.from_list(
        'ibm_density', ['#ffffff'] + list(reversed(style.palette))
    )
    ax.imshow(
        density,
        origin='lower',
        extent=[x_min, x_max, y_min, y_max],
        cmap=density_cmap,
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


def panel_empirical_scatter(ax, samples, population_frame, estimated_frame, style, axis_limit=2.5):
    """Bottom-left panel: sample cloud with population and empirical minimizers."""
    ax.scatter(
        samples[:, 0],
        samples[:, 1],
        s=16,
        color=style.palette[4],
        alpha=0.62,
        marker=style.markers[0],
        edgecolors='none',
    )

    x_min = y_min = -axis_limit
    x_max = y_max = axis_limit
    line_extent = 0.48 * max(x_max - x_min, y_max - y_min)

    draw_frame(
        ax,
        population_frame,
        color_a=style.palette[1],
        color_b=style.palette[1],
        line_extent=line_extent,
        linestyle='--',
    )
    draw_frame(
        ax,
        estimated_frame,
        color_a=style.palette[3],
        color_b=style.palette[3],
        line_extent=line_extent,
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
                color=style.palette[1],
                linestyle='--',
                linewidth=plt.rcParams['lines.linewidth'],
                label='Population minimizer',
            ),
            Line2D(
                [0],
                [0],
                color=style.palette[3],
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





def add_objective_zoom_inset(
    ax,
    theta_grid,
    population_values,
    empirical_theta_grid,
    empirical_values,
    switch_angles,
    style,
):
    """Add a linked zoom window around the empirical minimizer."""
    zoom_center = empirical_theta_grid[int(np.argmin(empirical_values))]
    half_width = np.pi / 12
    x_min = max(-0.5 * np.pi, zoom_center - half_width)
    x_max = min(0.5 * np.pi, zoom_center + half_width)
    if x_max - x_min < 2 * half_width:
        if x_min <= -0.5 * np.pi:
            x_max = x_min + 2 * half_width
        else:
            x_min = x_max - 2 * half_width

    population_mask = (theta_grid >= x_min) & (theta_grid <= x_max)
    empirical_mask = (empirical_theta_grid >= x_min) & (empirical_theta_grid <= x_max)
    zoom_values = np.concatenate([population_values[population_mask], empirical_values[empirical_mask]])
    if zoom_values.size == 0:
        return

    value_range = np.ptp(zoom_values)
    y_pad = 0.12 * value_range if value_range > 0 else 0.05 * max(abs(zoom_values[0]), 1.0)
    y_min = max(0.0, float(np.min(zoom_values) - y_pad))
    y_max = float(np.max(zoom_values) + y_pad)

    left, bottom, width = 0.62, 0.67, 0.33
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
        population_values,
        color=style.palette[4],
        linewidth=plt.rcParams['lines.linewidth'],
    )
    inset.plot(
        empirical_theta_grid,
        empirical_values,
        color=style.palette[2],
        linewidth=plt.rcParams['lines.linewidth'],
        alpha=0.95,
    )

    switches_in_window = switch_angles[(switch_angles >= x_min) & (switch_angles <= x_max)]
    if switches_in_window.size:
        switch_values = np.interp(switches_in_window, empirical_theta_grid, empirical_values)
        inset.scatter(
            switches_in_window,
            switch_values,
            s=28,
            marker='x',
            color=style.palette[1],
            linewidths=plt.rcParams['axes.linewidth'],
            alpha=0.9,
            zorder=5,
        )
        inset.vlines(
            switches_in_window,
            y_min,
            y_min + 0.08 * (y_max - y_min),
            color=style.palette[1],
            linewidth=0.8,
            alpha=0.45,
            zorder=1,
        )

    inset.set_xlim(x_min, x_max)
    inset.set_ylim(y_min, y_max)
    inset.grid(True, which='major', ls=':', lw=0.4, alpha=0.45)
    inset.tick_params(length=2.0, width=0.7, labelsize=max(10, plt.rcParams['xtick.labelsize'] - 2))
    inset.tick_params(axis='y', labelleft=False)

    ticks, labels = pi_tick_labels()
    tick_mask = (ticks >= x_min) & (ticks <= x_max)
    inset.set_xticks(ticks[tick_mask])
    inset.set_xticklabels([label for label, keep in zip(labels, tick_mask) if keep])

    mark_inset(ax, inset, loc1=2, loc2=4, fc='none', ec='0.45', lw=0.8, alpha=0.85)

def panel_objectives(
    ax,
    theta_grid,
    population_values,
    empirical_theta_grid,
    empirical_values,
    empirical_samples,
    style,
):
    """Right panel: population and empirical objective over frame angle."""
    ax.plot(
        theta_grid,
        population_values,
        color=style.palette[4],
        linewidth=plt.rcParams['lines.linewidth'],
        label='Population objective',
    )
    ax.plot(
        empirical_theta_grid,
        empirical_values,
        color=style.palette[2],
        linewidth=plt.rcParams['lines.linewidth'],
        alpha=0.95,
        label='Empirical objective',
    )

    switch_angles = empirical_axis_switch_angles(empirical_samples)
    if switch_angles.size:
        y_max = max(np.max(population_values), np.max(empirical_values))
        rug_height = 0.045 * y_max
        ax.vlines(
            switch_angles,
            0.0,
            rug_height,
            color=style.palette[1],
            linewidth=0.8,
            alpha=0.45,
            label='Axis switches',
            zorder=1,
        )
        ax.set_ylim(bottom=0.0)
        switch_values = np.interp(switch_angles, empirical_theta_grid, empirical_values)
        ax.scatter(
            switch_angles,
            switch_values,
            s=22,
            marker='x',
            color=style.palette[1],
            linewidths=plt.rcParams['axes.linewidth'],
            alpha=0.9,
            zorder=5,
        )
    ticks, labels = pi_tick_labels()
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)
    ax.set_xlabel('Rotation angle', labelpad=6)
    ax.set_ylabel('Objective value', labelpad=7)
    ax.grid(True, which='major', ls=':', lw=0.5, alpha=0.6)
    add_objective_zoom_inset(
        ax,
        theta_grid,
        population_values,
        empirical_theta_grid,
        empirical_values,
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

def diagnose_objective_mismatch(theta_grid, population_values, empirical_values):
    """Print diagnostics explaining differences between population and empirical curves."""
    pop_argmin = int(np.argmin(population_values))
    emp_argmin = int(np.argmin(empirical_values))
    pop_theta = float(np.degrees(theta_grid[pop_argmin]))
    emp_theta = float(np.degrees(theta_grid[emp_argmin]))

    raw_rmse = float(np.sqrt(np.mean((population_values - empirical_values) ** 2)))
    pop_centered = population_values - np.min(population_values)
    emp_centered = empirical_values - np.min(empirical_values)
    centered_rmse = float(np.sqrt(np.mean((pop_centered - emp_centered) ** 2)))

    pop_curvature = float(np.mean(np.abs(np.diff(population_values, n=2))))
    emp_curvature = float(np.mean(np.abs(np.diff(empirical_values, n=2))))

    print('\nObjective diagnostics')
    print('  - Population minimizer angle (deg): {:.2f}'.format(pop_theta))
    print('  - Empirical minimizer angle (deg):  {:.2f}'.format(emp_theta))
    print('  - Minimizer gap (deg):              {:.2f}'.format(abs(pop_theta - emp_theta)))
    print('  - Raw RMSE between curves:          {:.4f}'.format(raw_rmse))
    print('  - RMSE after min-centering:         {:.4f}'.format(centered_rmse))
    print('  - Mean abs 2nd-diff (population):   {:.4e}'.format(pop_curvature))
    print('  - Mean abs 2nd-diff (empirical):    {:.4e}'.format(emp_curvature))
    print('  - Interpretation: empirical curve is rougher due to finite-sample noise and the max projection term.')


def create_main_figure(
    n_population=100000,
    n_empirical=200,
    seed_population=7,
    seed_empirical=31,
    seed_initialization=101,
    figure_dir=Path('figures'),
):
    """Create and save the three-panel figure used in the paper."""
    style = PaperPlotStyle()
    style.apply()

    true_frame = true_frame_gaussian_mixture()
    population_samples = sample_gaussian_mixture_measure(n_population, seed=seed_population)
    #population_samples = sample_double_exponential_measure(n_population, seed=seed_population)
    #population_samples = sample_student_t_measure(n_population, seed=seed_population)

    rng_empirical = np.random.default_rng(seed_empirical)
    empirical_indices = rng_empirical.choice(n_population, size=n_empirical, replace=False)
    empirical_samples = population_samples[empirical_indices]

    theta_grid = np.linspace(-0.5 * np.pi, 0.5 * np.pi, 721)
    empirical_theta_grid = np.linspace(-0.5 * np.pi, 0.5 * np.pi, 121)

    initial_frame = random_orthogonal_frame(dim=2, seed=seed_initialization)
    estimated_frame, est_theta, opt_result = estimate_wfa1_frame_pymanopt(
        empirical_samples,
        initial_frame=initial_frame,
    )
    empirical_values = objective_curve(empirical_samples, empirical_theta_grid)
    population_values = objective_curve(population_samples, theta_grid)
    empirical_values_dense = objective_curve(empirical_samples, theta_grid)
    diagnose_objective_mismatch(theta_grid, population_values, empirical_values_dense)
    initial_theta = float(np.arctan2(initial_frame[0, 1], initial_frame[0, 0]))
    print('  - initialization angle (deg):         {:.2f}'.format(np.degrees(initial_theta)))
    print('  - pymanopt U_n^* angle (deg):         {:.2f}'.format(np.degrees(est_theta)))
    print('  - pymanopt iterations:                {}'.format(opt_result.iterations))
    print('  - pymanopt gradient norm:             {:.3e}'.format(opt_result.gradient_norm))

    fig = plt.figure(figsize=(10.4, 5.0))
    # Explicit axes rectangles keep the two left panels exactly equal in
    # physical size. Fractions account for the 10.4 x 5.0 inch figure.
    left_panel_size = 0.205
    left_panel_height = left_panel_size * (10.4 / 5.0)
    left_x = 0.025
    ax_density = fig.add_axes([left_x, 0.545, left_panel_size, left_panel_height])
    ax_empirical = fig.add_axes([left_x, 0.095, left_panel_size, left_panel_height])
    ax_objective = fig.add_axes([0.320, 0.135, 0.660, 0.785])

    panel_population_density(ax_density, population_samples, style)
    panel_empirical_scatter(ax_empirical, empirical_samples, true_frame, estimated_frame, style)
    panel_objectives(
        ax_objective,
        theta_grid,
        population_values,
        empirical_theta_grid,
        empirical_values,
        empirical_samples,
        style,
    )

    for ax in (ax_density, ax_empirical, ax_objective):
        ax.tick_params(length=3.0, width=0.8)

    figure_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_dir / 'fig_main.pdf', bbox_inches='tight', pad_inches=0.04)
    fig.savefig(figure_dir / 'fig_main.png', bbox_inches='tight', pad_inches=0.04)
    print('Saved figures/fig_main.{pdf,png}')


def main():
    create_main_figure()


if __name__ == '__main__':
    main()