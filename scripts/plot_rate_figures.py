import csv
from pathlib import Path

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

from scripts.plot_style import PaperPlotStyle


def configure_loglog_axis(ax):
    ax.set_xscale('log', base=2)
    ax.set_yscale('log', base=2)
    ax.set_xlabel('Sample size')
    ax.set_ylabel('Relative error')
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.grid(True, which='major', ls=':', lw=0.5, alpha=0.6)
    ax.grid(False, which='minor')


def load_dimension_series(dim, data_dir):
    csv_path = data_dir / f'results_id_dim{dim}.csv'
    n_values = []
    relative_error = []
    with open(csv_path, 'r') as f:
        for row in csv.DictReader(f):
            n_values.append(float(row['Number Points']))
            relative_error.append(np.sqrt(float(row['Normalized Error'])))
    return np.array(n_values), np.array(relative_error)


def add_inverse_sqrt_reference(ax, x_values, y_values):
    mid_idx = len(x_values) // 2
    scale = y_values[mid_idx] * np.sqrt(x_values[mid_idx])
    reference_y = scale / np.sqrt(x_values)
    ax.plot(x_values, reference_y, color='black', ls='--', lw=plt.rcParams['lines.linewidth'], alpha=0.75)
    label_idx = min(mid_idx + 1, len(x_values) - 2)
    x0, y0 = x_values[label_idx], reference_y[label_idx]
    x1, y1 = x_values[label_idx + 1], reference_y[label_idx + 1]
    p0 = ax.transData.transform((x0, y0))
    p1 = ax.transData.transform((x1, y1))
    angle = np.degrees(np.arctan2(p1[1] - p0[1], p1[0] - p0[0]))
    ax.annotate(
        r'$1/\sqrt{n}$', xy=(x0, y0), xytext=(0, 4), textcoords='offset points',
        rotation=angle, rotation_mode='anchor', ha='center', va='bottom',
        fontsize=plt.rcParams['legend.fontsize'], color='black',
    )


def plot_empirical_statistical_rates(ax, style, data_dir, dimensions=(2, 25, 50, 75, 100)):
    reference_x = None
    reference_y = None
    for idx, dim in enumerate(dimensions):
        x, y = load_dimension_series(dim, data_dir)
        meta = style.series_meta(idx)
        ax.plot(x, y, label=f'$d={dim}$', color=meta['color'], marker=meta['marker'],
                linewidth=plt.rcParams['lines.linewidth'])
        if reference_x is None:
            reference_x = x
            reference_y = y
    if reference_x is None:
        raise ValueError('No empirical statistical rate data was loaded.')
    add_inverse_sqrt_reference(ax, reference_x, reference_y)
    configure_loglog_axis(ax)
    ax.autoscale(tight=True)
    ax.legend(loc='best', frameon=True, framealpha=0.9, edgecolor='0.8')


def load_convergence_summary(csv_path):
    with open(csv_path, 'r') as f:
        return list(csv.DictReader(f))


def plot_convergence_rate(ax, style, data_dir, dim=5):
    rows = load_convergence_summary(data_dir / 'convergence_rate_summary_d5_steepest_descent_delta0.1_trials10.csv')
    if not rows:
        raise ValueError('No convergence summary data found.')
    n_values = np.array([float(row['n_points']) for row in rows])
    gaussian_y = None
    for data_type, meta in style.data_type_meta.items():
        mean_col = f'{data_type}_mean_log2_inv_eps'
        std_col = f'{data_type}_std_log2_inv_eps'
        mean_log = np.array([float(row[mean_col]) for row in rows])
        std_log = np.array([float(row[std_col]) for row in rows])
        means = np.sqrt(2.0 ** (-mean_log))
        lower = np.sqrt(2.0 ** (-(mean_log + std_log)))
        upper = np.sqrt(2.0 ** (-(mean_log - std_log)))
        ax.plot(n_values, means, label=meta['label'], marker=meta['marker'], color=meta['color'],
                linewidth=plt.rcParams['lines.linewidth'])
        ax.fill_between(n_values, lower, upper, color=meta['color'], alpha=0.12)
        if data_type == 'gaussian':
            gaussian_y = means
    if gaussian_y is not None:
        add_inverse_sqrt_reference(ax, n_values, gaussian_y)
    configure_loglog_axis(ax)
    ax.legend(loc='best', frameon=True, framealpha=0.9, edgecolor='0.8')


def plot_rate_figures(data_dir=Path('experiments/rates'), figure_dir=Path('figures')):
    style = PaperPlotStyle()
    style.apply()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), sharey=False)
    plot_empirical_statistical_rates(axes[0], style, data_dir)
    plot_convergence_rate(axes[1], style, data_dir)
    axes[1].set_ylabel('')
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(figure_dir / 'fig_rates_combined.pdf', bbox_inches='tight')
    fig.savefig(figure_dir / 'fig_rates_combined.png', bbox_inches='tight')
    print('Saved figures/fig_rates_combined.{pdf,png}')


if __name__ == '__main__':
    plot_rate_figures()
