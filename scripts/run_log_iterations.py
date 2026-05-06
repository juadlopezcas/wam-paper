import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.linalg import expm

from data.data_utils import DataGenerator
from scripts.methods import ensure_optimization_deps, run_wfa1
from scripts.metrics import approx_metric_matching

DATA_GENERATORS = {
    'gaussian': 'gaussian_factor_analysis_data',
    'exponential': 'exponential_factor_analysis_data',
    'student_t': 'student_t_data',
}


def generate_points(generator, data_type):
    return getattr(generator, DATA_GENERATORS[data_type])()


def perturb_identity_matrix(dim, delta):
    A = np.random.randn(dim, dim)
    A = A - A.T
    return expm(delta * A)


def log_iterations_path(output_dir, data_type, n_points, dim):
    return output_dir / f'log_iterations_{data_type}_n{n_points}_d{dim}.csv'


def save_log_to_csv(log_dict, n_points, dim, data_type, path):
    iterations = log_dict['iterations']['iteration']
    points = log_dict['iterations']['point']
    costs = log_dict['iterations']['cost']
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['data_type', 'n_points', 'dimension', 'iteration', 'cost', 'error', 'normalized_error'])
        for i in range(len(iterations)):
            error = approx_metric_matching(points[i], np.identity(dim))
            normalized_error = error / np.linalg.norm(np.identity(dim), ord='fro') ** 2
            writer.writerow([data_type, n_points, dim, iterations[i], costs[i], error, normalized_error])


def run_single_log_iteration(n_points, dim, data_type, output_dir, delta=0.1, seed=None):
    ensure_optimization_deps()
    if seed is not None:
        np.random.seed(seed)
    generator = DataGenerator(n_points, dim)
    data = generate_points(generator, data_type)
    initial_point = perturb_identity_matrix(dim, delta)
    _, _, _, details = run_wfa1(
        data,
        dim,
        initial_point=initial_point,
        log_verbosity=2,
        max_iterations=1000,
    )
    output_path = log_iterations_path(output_dir, data_type, n_points, dim)
    save_log_to_csv(details['log'], n_points, dim, data_type, output_path)
    return output_path


def run_log_iteration_suite(
    output_dir=Path('experiments/log_iterations'),
    fixed_dim=15,
    fixed_n_points=5000,
    n_points_list=(2000, 4000, 8000, 16000, 32000),
    dim_list=(12, 25, 50, 75, 100),
    data_types=('gaussian', 'exponential', 'student_t'),
    delta=0.1,
):
    paths = []
    for data_type in data_types:
        for n_points in n_points_list:
            print(f'Running {data_type}, n={n_points}, d={fixed_dim}')
            paths.append(run_single_log_iteration(n_points, fixed_dim, data_type, output_dir, delta=delta))
    for data_type in data_types:
        for dim in dim_list:
            print(f'Running {data_type}, n={fixed_n_points}, d={dim}')
            paths.append(run_single_log_iteration(fixed_n_points, dim, data_type, output_dir, delta=delta))
    return paths


def parse_args():
    parser = argparse.ArgumentParser(description='Regenerate log-iteration CSV files.')
    parser.add_argument('--output-dir', type=Path, default=Path('experiments/log_iterations'))
    parser.add_argument('--single', action='store_true', help='Run one configuration instead of the full suite.')
    parser.add_argument('--data-type', choices=sorted(DATA_GENERATORS), default='gaussian')
    parser.add_argument('--n-points', type=int, default=2000)
    parser.add_argument('--dim', type=int, default=15)
    parser.add_argument('--delta', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.single:
        paths = [run_single_log_iteration(args.n_points, args.dim, args.data_type, args.output_dir, args.delta, args.seed)]
    else:
        paths = run_log_iteration_suite(output_dir=args.output_dir, delta=args.delta)
    for path in paths:
        print(f'Saved {path}')


if __name__ == '__main__':
    main()
