import argparse
import csv
from pathlib import Path

import numpy as np

from data.data_utils import DataGenerator
from scripts.methods import ensure_optimization_deps, run_wfa1
from scripts.metrics import normalized_frame_error

DATA_TYPES = ('gaussian', 'exponential', 'student_t')


def generate_points(generator, data_type):
    if data_type == 'gaussian':
        return generator.gaussian_factor_analysis_data()
    if data_type == 'exponential':
        return generator.exponential_factor_analysis_data()
    if data_type == 'student_t':
        return generator.student_t_data()
    raise ValueError(f'Unknown data type: {data_type}')


def run_dimension_rate(dim, output_dir, sample_sizes=(100, 200, 400, 800, 1600, 3200, 6400), seed=42):
    ensure_optimization_deps()
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f'results_id_dim{dim}.csv'
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Dimension', 'Number Points', 'Assignment Error', 'Normalized Error', 'Iterations',
                         'Stopping Criterion', 'Step Size', 'Gradient Norm'])
        for idx, n_points in enumerate(sample_sizes):
            np.random.seed(seed + dim * 1000 + idx)
            generator = DataGenerator(n_points, dim)
            data = generator.gaussian_factor_analysis_data()
            estimated, _, _, details = run_wfa1(data, dim)
            assignment_error, normalized_error = normalized_frame_error(estimated, generator.matrix_axis)
            writer.writerow([
                dim,
                n_points,
                assignment_error,
                normalized_error,
                details.get('iterations'),
                details.get('stopping_criterion'),
                details.get('step_size'),
                details.get('gradient_norm'),
            ])
    return path


def run_convergence_rate_summary(
    output_dir,
    dim=5,
    sample_sizes=(200, 400, 800, 1600, 3200),
    n_trials=10,
    delta=0.1,
    seed=42,
):
    ensure_optimization_deps()
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f'convergence_rate_summary_d{dim}_steepest_descent_delta{delta}_trials{n_trials}.csv'
    with open(path, 'w', newline='') as f:
        fieldnames = ['n_points', 'log2_n']
        for data_type in DATA_TYPES:
            fieldnames.extend([f'{data_type}_mean_log2_inv_eps', f'{data_type}_std_log2_inv_eps'])
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for n_points in sample_sizes:
            row = {'n_points': n_points, 'log2_n': np.log2(n_points)}
            for data_type in DATA_TYPES:
                log_errors = []
                for trial in range(n_trials):
                    np.random.seed(seed + trial * 1000 + n_points + 17 * dim)
                    generator = DataGenerator(n_points, dim)
                    data = generate_points(generator, data_type)
                    estimated, _, _, _ = run_wfa1(data, dim)
                    _, normalized_error = normalized_frame_error(estimated, generator.matrix_axis)
                    log_errors.append(np.log2(1.0 / max(normalized_error, np.finfo(float).tiny)))
                row[f'{data_type}_mean_log2_inv_eps'] = float(np.mean(log_errors))
                row[f'{data_type}_std_log2_inv_eps'] = float(np.std(log_errors))
            writer.writerow(row)
    return path


def run_rates_suite(output_dir=Path('experiments/rates'), dimensions=(2, 25, 50, 75, 100), n_trials=10):
    paths = []
    for dim in dimensions:
        print(f'Running empirical rate experiment for d={dim}')
        paths.append(run_dimension_rate(dim, output_dir))
    print('Running convergence-rate summary for d=5')
    paths.append(run_convergence_rate_summary(output_dir, n_trials=n_trials))
    return paths


def parse_args():
    parser = argparse.ArgumentParser(description='Regenerate statistical-rate CSV files.')
    parser.add_argument('--output-dir', type=Path, default=Path('experiments/rates'))
    parser.add_argument('--single-dim', type=int, default=None, help='Only regenerate results_id_dim{d}.csv.')
    parser.add_argument('--summary-only', action='store_true')
    parser.add_argument('--trials', type=int, default=10)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.single_dim is not None:
        paths = [run_dimension_rate(args.single_dim, args.output_dir)]
    elif args.summary_only:
        paths = [run_convergence_rate_summary(args.output_dir, n_trials=args.trials)]
    else:
        paths = run_rates_suite(args.output_dir, n_trials=args.trials)
    for path in paths:
        print(f'Saved {path}')


if __name__ == '__main__':
    main()
