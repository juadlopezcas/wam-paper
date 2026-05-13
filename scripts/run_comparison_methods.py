import argparse
import csv
from pathlib import Path

import numpy as np

from data.data_utils import DataGenerator
from scripts.methods import run_trial

METHODS = [
    'deflation_varimax_random',
    'robust_subspace_clustering',
    'pca_varimax',
    'wfa1_manifold_optimization',
]

FRAME_HEADER = [
    'method', 'data_type', 'dimension', 'sample_size', 'noise_level', 'trial', 'frame_error',
    'runtime', 'success', 'initialization', 'gradient_norm', 'iterations', 'step_size', 'stopping_criterion'
]
NOISE_HEADER = [
    'method', 'dimension', 'noise_level', 'trial', 'frame_error', 'runtime', 'success',
    'gradient_norm', 'iterations', 'step_size', 'stopping_criterion'
]

def random_orthogonal(dim):
    Q, R = np.linalg.qr(np.random.randn(dim, dim))
    signs = np.sign(np.diag(R))
    signs[signs == 0] = 1
    return Q * signs

def generate_data(data_type, n_samples, dim, noise_level):
    generator = DataGenerator(n_samples, dim)

    rotated = data_type.endswith('_rotated')
    base_type = data_type.replace('_rotated', '')

    if base_type == 'gaussian':
        data = generator.gaussian_factor_analysis_data(t=noise_level)
    elif base_type == 'exponential':
        data = generator.exponential_factor_analysis_data()
    else:
        raise ValueError(f'Unknown data type: {data_type}')

    true_frames = generator.matrix_axis

    if rotated:
        A = random_orthogonal(dim)
        data = data @ A.T
        true_frames = A @ true_frames

    return data, true_frames


def run_single(method, data_type, dim, n_samples, noise_level, trial, seed=42):
    np.random.seed(seed + trial * 1000)
    data, true_frames = generate_data(data_type, n_samples, dim, noise_level)
    assignment_error, frame_error, runtime, success, details = run_trial(method, data, dim, true_frames)
    return {
        'method': method,
        'data_type': data_type,
        'dimension': dim,
        'sample_size': n_samples,
        'noise_level': noise_level,
        'trial': trial,
        'frame_error': frame_error,
        'runtime': runtime,
        'success': success,
        'initialization': 'random' if method == 'deflation_varimax_random' else 'N/A',
        'gradient_norm': details.get('gradient_norm'),
        'iterations': details.get('iterations'),
        'step_size': details.get('step_size'),
        'stopping_criterion': details.get('stopping_criterion'),
    }


def run_frame_accuracy(output_dir, dimensions=(2, 5, 10, 15, 25, 50), sample_sizes=(1000,),
                       data_types=('gaussian_rotated',), noise_level=0.05, n_trials=10, methods=METHODS):
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / 'frame_accuracy_results.csv'
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FRAME_HEADER)
        writer.writeheader()
        for method in methods:
            for data_type in data_types:
                for dim in dimensions:
                    for n_samples in sample_sizes:
                        if n_samples < dim * 5:
                            continue
                        print(f'Frame accuracy: {method}, {data_type}, d={dim}, n={n_samples}')
                        for trial in range(n_trials):
                            row = run_single(method, data_type, dim, n_samples, noise_level, trial)
                            writer.writerow(row)
                            f.flush()
    return path


def run_noise_robustness(output_dir, dimensions=(5,), noise_levels=(0.01, 0.05, 0.1, 0.2, 0.3),
                         n_samples=1000, n_trials=10, methods=METHODS):
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / 'noise_robustness_results.csv'
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=NOISE_HEADER)
        writer.writeheader()
        for method in methods:
            for dim in dimensions:
                for noise_level in noise_levels:
                    print(f'Noise robustness: {method}, d={dim}, noise={noise_level}')
                    for trial in range(n_trials):
                        result = run_single(method, 'gaussian_rotated', dim, n_samples, noise_level, trial)
                        writer.writerow({key: result[key] for key in NOISE_HEADER})
                        f.flush()
    return path


def run_comparison_suite(output_dir=Path('experiments/comparison_methods'), n_trials=10):
    return [
        run_frame_accuracy(output_dir, n_trials=n_trials),
        run_noise_robustness(output_dir, n_trials=n_trials),
    ]


def parse_args():
    parser = argparse.ArgumentParser(description='Regenerate method-comparison CSV files.')
    parser.add_argument('--output-dir', type=Path, default=Path('experiments/comparison_methods'))
    parser.add_argument('--frame-only', action='store_true')
    parser.add_argument('--noise-only', action='store_true')
    parser.add_argument('--trials', type=int, default=10)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.frame_only:
        paths = [run_frame_accuracy(args.output_dir, n_trials=args.trials)]
    elif args.noise_only:
        paths = [run_noise_robustness(args.output_dir, n_trials=args.trials)]
    else:
        paths = run_comparison_suite(args.output_dir, n_trials=args.trials)
    for path in paths:
        print(f'Saved {path}')


if __name__ == '__main__':
    main()
