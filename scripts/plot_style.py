import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


class PaperPlotStyle:
    """Centralized Matplotlib style and legend metadata."""

    palette = ['#ffb000', '#fe6100', '#dc267f', '#785ef0', '#648fff']
    markers = ['o', 's', '^', 'D', 'P', 'X']

    data_type_meta = {
        'gaussian': {'label': 'Gaussian mixture', 'color': palette[4], 'marker': 'o', 'ls': '-'},
        'exponential': {'label': 'Double exponential', 'color': palette[1], 'marker': 's', 'ls': '-'},
        'student_t': {'label': 'Student-t', 'color': palette[2], 'marker': '^', 'ls': '-'},
    }

    method_meta = {
        'wfa1_manifold_optimization': {
            'label': 'Ours', 'color': palette[3], 'marker': 'o', 'zorder': 5, 'ls': '-'
        },
        'robust_subspace_clustering': {
            'label': 'RSC', 'color': palette[1], 'marker': 's', 'zorder': 4, 'ls': '-'
        },
        'deflation_varimax_random': {
            'label': 'Deflation Varimax', 'color': palette[2], 'marker': '^', 'zorder': 3, 'ls': '--'
        },
        'pca_varimax': {
            'label': 'PCA+Varimax', 'color': palette[0], 'marker': 'D', 'zorder': 2, 'ls': '--'
        },
    }

    rc_params = {
        'font.family': 'cmr10',
        'mathtext.fontset': 'cm',
        'axes.formatter.use_mathtext': True,
        'font.size': 16,
        'axes.labelsize': 17,
        'axes.titlesize': 17,
        'legend.fontsize': 13,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'axes.linewidth': 1.4,
        'lines.linewidth': 3.5,
        'lines.markersize': 5.8,
        'figure.dpi': 150,
    }

    legend_kwargs = {
        'frameon': True,
        'framealpha': 0.9,
        'edgecolor': '0.8',
    }

    def apply(self):
        plt.rcParams.update(self.rc_params)

    @property
    def methods(self):
        return list(self.method_meta.keys())

    def series_meta(self, index):
        return {
            'color': self.palette[index % len(self.palette)],
            'marker': self.markers[index % len(self.markers)],
        }

    def legend_handles(self, series_specs):
        handles = []
        for idx, spec in enumerate(series_specs):
            meta = self.series_meta(idx)
            handles.append(
                Line2D(
                    [0],
                    [0],
                    color=meta['color'],
                    marker=meta['marker'],
                    linewidth=plt.rcParams['lines.linewidth'],
                    markersize=plt.rcParams['lines.markersize'],
                    label=spec['label'],
                )
            )
        return handles
