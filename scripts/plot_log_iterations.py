import csv
from pathlib import Path

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

from scripts.plot_style import PaperPlotStyle


DATA_TYPE_META = {
    'gaussian': {'label': 'Gaussian mixture', 'ls': '-'},
    'exponential': {'label': 'Double exponential', 'ls': '-'},
    'student_t': {'label': 'Student-t', 'ls': '-'},
}


def log_iterations_path(data_type, n_points, dim, data_dir):
    return data_dir / f'log_iterations_{data_type}_n{n_points}_d{dim}.csv'


def load_iteration_series(csv_path):
    with open(csv_path, 'r') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return np.array([]), np.array([])
    x = np.array([float(row['iteration']) for row in rows])
    y = np.array([float(row['normalized_error']) for row in rows])
    order = np.argsort(x)
    return x[order], np.maximum(y[order], np.finfo(float).tiny)


def plot_panel(ax, series_specs, data_type, title, style, data_dir):
    missing = []
    plotted = 0
    for idx, spec in enumerate(series_specs):
        csv_path = log_iterations_path(data_type, spec['n_points'], spec['dim'], data_dir)
        if not csv_path.exists():
            missing.append(f'{DATA_TYPE_META[data_type]["label"]}, {spec["label"]}')
            continue
        x, y = load_iteration_series(csv_path)
        if len(x) == 0:
            missing.append(f'{DATA_TYPE_META[data_type]["label"]}, {spec["label"]}')
            continue
        meta = style.series_meta(idx)
        ax.plot(
            x,
            y,
            label=spec['label'],
            color=meta['color'],
            marker=meta['marker'],
            ls=DATA_TYPE_META[data_type]['ls'],
            markevery=max(len(x) // 6, 1),
            linewidth=plt.rcParams['lines.linewidth'],
        )
        plotted += 1

    ax.set_title(title)
    ax.set_xlabel('Iterations')
    ax.set_yscale('log')
    ax.xaxis.set_minor_locator(ticker.NullLocator())
    ax.grid(True, which='major', ls=':', lw=0.5, alpha=0.6)
    ax.grid(False, which='minor')
    if not plotted:
        ax.text(0.5, 0.5, 'No generated data', transform=ax.transAxes,
                ha='center', va='center', color='0.35')
    return missing


def plot_log_iterations(
    data_dir=Path('experiments/log_iterations'),
    figure_dir=Path('figures'),
    fixed_dim=15,
    fixed_n_points=5000,
    n_points_list=(2000, 4000, 8000, 16000, 32000),
    dim_list=(12, 25, 50, 75, 100),
):
    style = PaperPlotStyle()
    style.apply()

    n_series = [{'n_points': n, 'dim': fixed_dim, 'label': f'$n={n}$'} for n in n_points_list]
    d_series = [{'n_points': fixed_n_points, 'dim': d, 'label': f'$d={d}$'} for d in dim_list]

    fig = plt.figure(figsize=(9.4, 5.2))
    grid = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.55], right=0.875, wspace=0.08, hspace=0.34)
    ax_gaussian_n = fig.add_subplot(grid[0, 0])
    ax_gaussian_d = fig.add_subplot(grid[1, 0])
    top_right = grid[0, 1].subgridspec(1, 2, wspace=0.36)
    bottom_right = grid[1, 1].subgridspec(1, 2, wspace=0.36)
    ax_exp_n = fig.add_subplot(top_right[0, 0])
    ax_student_n = fig.add_subplot(top_right[0, 1])
    ax_exp_d = fig.add_subplot(bottom_right[0, 0])
    ax_student_d = fig.add_subplot(bottom_right[0, 1])

    panels = [
        (ax_gaussian_n, 'gaussian', n_series, 'Gaussian mixture'),
        (ax_exp_n, 'exponential', n_series, 'Double exponential'),
        (ax_student_n, 'student_t', n_series, 'Student-t'),
        (ax_gaussian_d, 'gaussian', d_series, ''),
        (ax_exp_d, 'exponential', d_series, ''),
        (ax_student_d, 'student_t', d_series, ''),
    ]

    all_missing = []
    for ax, data_type, series_specs, title in panels:
        all_missing.extend(plot_panel(ax, series_specs, data_type, title, style, data_dir))

    ax_gaussian_n.set_ylabel('Error')
    ax_gaussian_d.set_ylabel('Error')
    ax_gaussian_n.set_xlim(left=0, right=25)
    ax_gaussian_d.set_xlim(left=0, right=32)
    ax_exp_n.set_xlim(left=0, right=30)
    ax_exp_d.set_xlim(left=0, right=30)
    ax_student_n.set_xlim(left=0, right=30)
    ax_student_d.set_xlim(left=0, right=40)

    legend_kwargs = {
        'loc': 'center left', 'frameon': True, 'framealpha': 0.9, 'edgecolor': '0.8',
        'borderpad': 0.25, 'handlelength': 1.25, 'handletextpad': 0.35,
        'labelspacing': 0.25, 'borderaxespad': 0.15,
    }
    fig.legend(handles=style.legend_handles(n_series), bbox_to_anchor=(0.895, 0.705), **legend_kwargs)
    fig.legend(handles=style.legend_handles(d_series), bbox_to_anchor=(0.895, 0.285), **legend_kwargs)

    figure_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_dir / 'fig_log_iterations.pdf', bbox_inches='tight')
    fig.savefig(figure_dir / 'fig_log_iterations.png', bbox_inches='tight')
    if all_missing:
        print('Missing generated series: ' + ', '.join(all_missing))
    print('Saved figures/fig_log_iterations.{pdf,png}')


if __name__ == '__main__':
    plot_log_iterations()
