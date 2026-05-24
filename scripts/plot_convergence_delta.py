import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

from scripts.plot_style import PaperPlotStyle


SOLVER_META = {
    'steepest_descent': {
        'label': 'Steepest descent',
        'color_index': 3,
        'marker': 'o',
        'linestyle': '-',
        'zorder': 3,
    },
    'trust_regions': {
        'label': 'Trust regions',
        'color_index': 2,
        'marker': 's',
        'linestyle': '-',
        'zorder': 4,
    },
}


def load_convergence_delta_trials(csv_path):
    grouped = defaultdict(list)
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            grouped[(row['solver'], float(row['delta']))].append(float(row['error']))
    return grouped


def summarize_by_delta(grouped, solver):
    deltas = sorted(delta for method, delta in grouped if method == solver)
    means, stds = [], []
    for delta in deltas:
        values = np.asarray(grouped[(solver, delta)], dtype=float)
        means.append(np.mean(values))
        stds.append(np.std(values))
    return np.asarray(deltas), np.asarray(means), np.asarray(stds)


def plot_convergence_delta(
    data_dir=Path('experiments/convergence_delta'),
    figure_dir=Path('figures'),
    n_points=2000,
    dim=15,
    n_trials=10,
):
    """Plot optimizer sensitivity to initialization distance delta."""
    style = PaperPlotStyle()
    style.apply()

    csv_path = data_dir / f'convergence_vs_delta_detailed_n{n_points}_d{dim}_trials{n_trials}.csv'
    grouped = load_convergence_delta_trials(csv_path)

    fig, ax = plt.subplots(figsize=(6.8, 4.2))

    for solver, meta in SOLVER_META.items():
        deltas, means, stds = summarize_by_delta(grouped, solver)
        if deltas.size == 0:
            continue
        color = style.palette[meta['color_index']]
        lower = np.maximum(means - stds, means * 0.25)
        upper = means + stds
        ax.plot(
            deltas,
            means,
            color=color,
            marker=meta['marker'],
            linestyle=meta['linestyle'],
            linewidth=plt.rcParams['lines.linewidth'],
            markersize=plt.rcParams['lines.markersize'],
            label=meta['label'],
            zorder=meta['zorder'],
        )
        ax.fill_between(
            deltas,
            lower,
            upper,
            color=color,
            alpha=0.16,
            linewidth=0,
            zorder=meta['zorder'] - 1,
        )

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Initialization perturbation')
    ax.set_ylabel('Squared frame error')
    ax.grid(True, which='major', ls=':', lw=0.5, alpha=0.6)
    ax.grid(False, which='minor')
    ax.xaxis.set_major_formatter(ticker.LogFormatterMathtext())
    ax.xaxis.set_minor_locator(ticker.NullLocator())

    ax.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, 1.14),
        ncol=2,
        frameon=True,
        framealpha=0.9,
        edgecolor='0.8',
        borderpad=0.3,
        handlelength=1.7,
        handletextpad=0.45,
        columnspacing=0.9,
    )

    figure_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(figure_dir / 'fig_convergence_vs_delta.pdf', bbox_inches='tight')
    fig.savefig(figure_dir / 'fig_convergence_vs_delta.png', bbox_inches='tight')
    plt.close(fig)
    print('Saved figures/fig_convergence_vs_delta.{pdf,png}')


if __name__ == '__main__':
    plot_convergence_delta()
