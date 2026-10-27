from pathlib import Path
import time

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import autograd.numpy as np
import pandas as pd

from scripts.plot_nytimes_gfa import (
    component_localization_scores,
    full_component_limits,
    load_selected_factors,
    orient_components_to_positive_skew,
    plot_pairplot_block,
    rohe_leverage_sample_indices,
    run_global_gfa_rotation,
    run_global_varimax_rotation,
)
from scripts.plot_style import PaperPlotStyle


DEFAULT_DATA_PATH = Path('data/nytimes_pca.npz')
DEFAULT_RESULTS_PATH = Path('experiments/nytimes_interpretability/full_data_results.csv')
DEFAULT_ROTATIONS_PATH = Path('experiments/nytimes_interpretability/full_data_rotations.npz')
DEFAULT_FIGURE_DIR = Path('figures')


def interpretability_metrics(matrix):
    """Scale-free row-sparsity and marginal-kurtosis summaries."""
    squared = np.asarray(matrix, dtype=float) ** 2
    energy = squared.sum(axis=1)
    nonzero = energy > np.finfo(float).tiny
    squared = squared[nonzero]
    energy = energy[nonzero]
    proportions = squared / energy[:, None]

    axis_concentration = proportions.max(axis=1)
    effective_support = 1.0 / np.sum(proportions ** 2, axis=1)
    l1_over_l2 = np.sum(np.sqrt(squared), axis=1) / np.sqrt(energy)
    dimension = matrix.shape[1]
    hoyer = (np.sqrt(dimension) - l1_over_l2) / (np.sqrt(dimension) - 1.0)

    second_moment = np.mean(np.asarray(matrix, dtype=float) ** 2, axis=0)
    fourth_moment = np.mean(np.asarray(matrix, dtype=float) ** 4, axis=0)
    kurtosis = fourth_moment / np.maximum(second_moment ** 2, np.finfo(float).tiny)

    return {
        'axis_concentration_weighted': np.max(squared, axis=1).sum() / energy.sum(),
        'axis_concentration_median': np.median(axis_concentration),
        'effective_support_median': np.median(effective_support),
        'hoyer_median': np.median(hoyer),
        'mean_raw_kurtosis': np.mean(kurtosis),
    }


def run_nytimes_interpretability_experiment(
    data_path=DEFAULT_DATA_PATH,
    results_path=DEFAULT_RESULTS_PATH,
    rotations_path=DEFAULT_ROTATIONS_PATH,
    k=8,
    max_iterations=1000,
):
    """Fit full-data GFA and R Varimax rotations on both sides of the NYTimes SVD."""
    U, data, keep = load_selected_factors(data_path, k, which_plot='u')
    V = np.asarray(data['V'][:, keep], dtype=float)
    singular_values = np.asarray(data['scree'][keep], dtype=float)
    n_documents = U.shape[0]

    # This scaling gives L_k = C_0 D_0^T for the centered, normalized NYTimes matrix.
    coefficients = np.sqrt(n_documents) * U
    loadings = V * (singular_values / np.sqrt(n_documents))[None, :]
    targets = (
        ('dictionary_learning', 'Document coefficients', U, coefficients),
        ('factor_analysis', 'Word loadings', V, loadings),
    )

    rows = []
    saved_rotations = {'keep': keep}
    for application, target_label, rotation_target, evaluation_matrix in targets:
        rotations = {'PCA': np.eye(k)}
        metadata = {'PCA': {}}

        started = time.time()
        rotations['GFA'], result = run_global_gfa_rotation(
            rotation_target,
            max_iterations=max_iterations,
            verbosity=0,
        )
        metadata['GFA'] = {
            'runtime_seconds': time.time() - started,
            'iterations': getattr(result, 'iterations', None),
            'gradient_norm': getattr(result, 'gradient_norm', None),
            'stopping_criterion': getattr(result, 'stopping_criterion', None),
        }

        started = time.time()
        rotations['Varimax'], result = run_global_varimax_rotation(rotation_target)
        metadata['Varimax'] = {
            'runtime_seconds': time.time() - started,
            'iterations': getattr(result, 'iterations', None),
            'gradient_norm': getattr(result, 'gradient_norm', None),
            'stopping_criterion': getattr(result, 'stopping_criterion', None),
        }
        saved_rotations[f'{application}_gfa'] = rotations['GFA']
        saved_rotations[f'{application}_varimax'] = rotations['Varimax']

        for method, rotation in rotations.items():
            row = {
                'application': application,
                'target': target_label,
                'method': method,
                'n_rows': rotation_target.shape[0],
                'rank': k,
                'components_zero_based': ','.join(map(str, keep)),
                'orthogonality_error': np.linalg.norm(rotation.T @ rotation - np.eye(k)),
                **metadata[method],
                **interpretability_metrics(evaluation_matrix @ rotation),
            }
            rows.append(row)

    results = pd.DataFrame(rows)
    results_path = Path(results_path)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(results_path, index=False)
    rotations_path = Path(rotations_path)
    rotations_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(rotations_path, **saved_rotations)
    print(f'Saved {results_path}')
    print(f'Saved {rotations_path}')
    return results


def plot_nytimes_gfa_interpretations_pairplot(
    data_path=DEFAULT_DATA_PATH,
    rotations_path=DEFAULT_ROTATIONS_PATH,
    figure_dir=DEFAULT_FIGURE_DIR,
    k=8,
    n_points=5000,
    seed=0,
):
    """Pairplots for GFA sparsity of document codes and normalized word loadings."""
    U, data, keep = load_selected_factors(data_path, k, which_plot='u')
    V = np.asarray(data['V'][:, keep], dtype=float)
    saved = np.load(rotations_path)
    if not np.array_equal(np.asarray(saved['keep']), keep):
        raise ValueError('Saved NYTimes rotations do not match the selected components; rerun with --generate.')

    rows = (
        (
            'Dictionary learning\n(document coefficients)',
            U,
            np.asarray(saved['dictionary_learning_gfa']),
            'c',
            'z',
            seed,
        ),
        (
            'Factor analysis\n(normalized word loadings)',
            V,
            np.asarray(saved['factor_analysis_gfa']),
            r'\lambda',
            r'\widetilde{\lambda}',
            seed + 1,
        ),
    )

    style = PaperPlotStyle()
    style.apply()
    fig = plt.figure(figsize=(15.2, 12.3))
    outer = fig.add_gridspec(
        2,
        2,
        left=0.115,
        right=0.995,
        bottom=0.045,
        top=0.94,
        wspace=0.20,
        hspace=0.28,
    )

    for row_index, (row_title, target, rotation, left_prefix, right_prefix, row_seed) in enumerate(rows):
        rotated, signs = orient_components_to_positive_skew(target @ rotation)
        rotation = rotation * signs
        sample_index = rohe_leverage_sample_indices(target, n_points, True, row_seed)
        sampled = target[sample_index]
        sampled_rotated = rotated[sample_index]
        left_grid = outer[row_index, 0].subgridspec(k, k, wspace=0.12, hspace=0.12)
        right_grid = outer[row_index, 1].subgridspec(k, k, wspace=0.12, hspace=0.12)
        plot_pairplot_block(
            fig,
            left_grid,
            sampled,
            full_component_limits(sampled),
            component_localization_scores(sampled),
            style,
            label_prefix=left_prefix,
            rotation=rotation,
            show_pca_axes=True,
            show_rotation_axes=False,
        )
        plot_pairplot_block(
            fig,
            right_grid,
            sampled_rotated,
            full_component_limits(sampled_rotated),
            component_localization_scores(sampled_rotated),
            style,
            label_prefix=right_prefix,
            rotation=None,
        )
        row_y = 0.935 if row_index == 0 else 0.485
        fig.text(0.035, row_y - 0.205, row_title, rotation=90, ha='center', va='center', fontsize=23)

    fig.text(0.33, 0.992, 'PCA coordinates', ha='center', va='top', fontsize=23)
    fig.text(0.79, 0.992, 'GFA-rotated coordinates', ha='center', va='top', fontsize=23)
    figure_dir = Path(figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)
    stem = figure_dir / 'fig_nytimes_gfa_two_interpretations_pairplot'
    fig.savefig(stem.with_suffix('.pdf'), bbox_inches='tight')
    fig.savefig(stem.with_suffix('.png'), bbox_inches='tight', dpi=220)
    plt.close(fig)
    print(f'Saved {stem}.{{pdf,png}}')


if __name__ == '__main__':
    plot_nytimes_gfa_interpretations_pairplot()
