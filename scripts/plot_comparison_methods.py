import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

from scripts.plot_style import PaperPlotStyle


def load_frame_accuracy_results(csv_path):
    grouped = defaultdict(list)
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            key = (row['method'], int(row['dimension']), row['data_type'], int(row['sample_size']))
            grouped[key].append((float(row['frame_error']), float(row['runtime'])))
    return grouped


def load_noise_robustness_results(csv_path):
    grouped = defaultdict(list)
    noise_levels = set()
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            noise_level = float(row['noise_level'])
            key = (row['method'], int(row['dimension']), noise_level)
            grouped[key].append((float(row['frame_error']), float(row['runtime'])))
            noise_levels.add(noise_level)
    return grouped, sorted(noise_levels)


def stats(vals):
    errs = np.array([v[0] for v in vals])
    rts = np.array([v[1] for v in vals])
    return np.mean(errs), np.std(errs), np.mean(rts), np.std(rts)


def lower_band(means, stds, floor=None):
    if floor is None:
        return np.maximum(means - stds, means * 0.25)
    return np.maximum(means - stds, floor)


def plot_series(ax, x_values, grouped, key_fn, style, metric):
    for method in style.methods:
        meta = style.method_meta[method]
        means, stds, valid_x = [], [], []
        for x in x_values:
            vals = grouped.get(key_fn(method, x), [])
            if not vals:
                continue
            result = stats(vals)
            means.append(result[0] if metric == 'frame_error' else result[2])
            stds.append(result[1] if metric == 'frame_error' else result[3])
            valid_x.append(x)
        if not valid_x:
            continue
        means = np.array(means)
        stds = np.array(stds)
        floor = 1e-6 if metric == 'runtime' else None
        ax.plot(
            valid_x, means, color=meta['color'], marker=meta['marker'], ls=meta['ls'],
            label=meta['label'], zorder=meta['zorder'], linewidth=plt.rcParams['lines.linewidth']
        )
        ax.fill_between(
            valid_x, lower_band(means, stds, floor=floor), means + stds,
            color=meta['color'], alpha=0.12, zorder=meta['zorder'] - 1,
        )


def format_axis(ax, x_values, xlabel, ylabel, x_log=False, y_log=True):
    if x_log:
        ax.set_xscale('log')
        ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    if y_log:
        ax.set_yscale('log')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x_values)
    ax.xaxis.set_minor_locator(ticker.NullLocator())
    ax.grid(True, which='major', ls=':', lw=0.5, alpha=0.6)
    ax.grid(False, which='minor')


def fit_log_y_to_lines(ax, pad_factor=1.35):
    y_values = []
    for line in ax.lines:
        y_data = np.asarray(line.get_ydata(), dtype=float)
        y_values.extend(y_data[np.isfinite(y_data) & (y_data > 0)])
    if not y_values:
        return
    ax.set_ylim(bottom=min(y_values) / pad_factor, top=max(y_values) * pad_factor)


def plot_comparison_methods(data_dir=Path('experiments/comparison_methods'), figure_dir=Path('figures')):
    style = PaperPlotStyle()
    style.apply()

    frame_grouped = load_frame_accuracy_results(data_dir / 'frame_accuracy_results.csv')
    noise_grouped, noise_levels = load_noise_robustness_results(data_dir / 'noise_robustness_results.csv')

    dims = [2, 5, 10, 15, 25, 50]
    data_type = 'gaussian_rotated'
    sample_size = 1000
    noise_dimension = 5

    fig, axes = plt.subplots(2, 2, figsize=(6.8, 5.2), sharex=False, sharey=False)
    ax_error_dim, ax_runtime_dim = axes[0]
    ax_error_noise, ax_runtime_noise = axes[1]

    plot_series(ax_error_dim, dims, frame_grouped,
                lambda method, d: (method, d, data_type, sample_size), style, 'frame_error')
    format_axis(ax_error_dim, dims, 'Dimension', 'Mean frame error', y_log=True)

    plot_series(ax_runtime_dim, dims, frame_grouped,
                lambda method, d: (method, d, data_type, sample_size), style, 'runtime')
    format_axis(ax_runtime_dim, dims, 'Dimension', 'Mean runtime (s)', x_log=True, y_log=True)
    fit_log_y_to_lines(ax_runtime_dim)

    plot_series(ax_error_noise, noise_levels, noise_grouped,
                lambda method, tau: (method, noise_dimension, tau), style, 'frame_error')
    format_axis(ax_error_noise, noise_levels, 'Noise level', 'Mean frame error', y_log=True)

    plot_series(ax_runtime_noise, noise_levels, noise_grouped,
                lambda method, tau: (method, noise_dimension, tau), style, 'runtime')
    format_axis(ax_runtime_noise, noise_levels, 'Noise level', 'Mean runtime (s)', y_log=True)

    handles, labels = ax_error_dim.get_legend_handles_labels()
    fig.legend(
        handles, labels, loc='upper center', ncol=4, bbox_to_anchor=(0.5, 0.995),
        frameon=True, framealpha=0.9, edgecolor='0.8', borderpad=0.45,
        handlelength=1.7, handletextpad=0.5,
    )

    figure_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(figure_dir / 'fig_comparison_completing_methods.pdf', bbox_inches='tight')
    fig.savefig(figure_dir / 'fig_comparison_completing_methods.png', bbox_inches='tight')
    print('Saved figures/fig_comparison_completing_methods.{pdf,png}')


if __name__ == '__main__':
    plot_comparison_methods()
