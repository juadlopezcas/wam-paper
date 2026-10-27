import argparse

from scripts.plot_comparison_methods import plot_comparison_methods
from scripts.plot_convergence_delta import plot_convergence_delta
from scripts.plot_exponential_frames import plot_exponential_frames
from scripts.plot_log_iterations import plot_log_iterations
from scripts.plot_main_gradient_figure import create_main_gradient_figure
from scripts.plot_nytimes_interpretability import (
    plot_nytimes_gfa_interpretations_pairplot,
)
from scripts.plot_rate_figures import plot_rate_figures


def parse_args():
    parser = argparse.ArgumentParser(description='Reproduce WAM paper figures.')
    parser.add_argument('--all', action='store_true', help='Generate all retained figures from saved inputs.')
    parser.add_argument('--log-iterations', action='store_true', help='Generate fig_log_iterations.')
    parser.add_argument('--comparison-methods', action='store_true', help='Generate fig_comparison_completing_methods.')
    parser.add_argument('--convergence-delta', action='store_true', help='Generate fig_convergence_vs_delta.')
    parser.add_argument('--rates', action='store_true', help='Generate fig_rates_combined.')
    parser.add_argument('--main-gradient', action='store_true', help='Generate fig_main_gradient.')
    parser.add_argument('--exponential-frames', action='store_true', help='Generate fig_exponential_frames. Requires R and rpy2.')
    parser.add_argument('--nytimes', action='store_true', help='Generate fig_nytimes_gfa_two_interpretations_pairplot from saved rotations.')
    parser.add_argument('--prepare-nytimes', action='store_true', help='Download and preprocess the NYTimes data cache.')
    parser.add_argument('--generate', action='store_true', help='Regenerate CSVs before plotting selected figures.')
    parser.add_argument('--trials', type=int, default=10, help='Trials for regenerated comparison or rate summaries.')
    return parser.parse_args()


def main():
    args = parse_args()
    run_all = args.all or not (
        args.log_iterations
        or args.comparison_methods
        or args.convergence_delta
        or args.rates
        or args.main_gradient
        or args.exponential_frames
        or args.nytimes
        or args.prepare_nytimes
    )

    if args.prepare_nytimes:
        from scripts.prepare_nytimes_data import prepare_nytimes_data
        prepare_nytimes_data(force_download=args.generate)

    if run_all or args.main_gradient:
        create_main_gradient_figure()

    if run_all or args.nytimes:
        if args.generate:
            from scripts.plot_nytimes_interpretability import run_nytimes_interpretability_experiment
            run_nytimes_interpretability_experiment()
        plot_nytimes_gfa_interpretations_pairplot()

    if run_all or args.exponential_frames:
        try:
            plot_exponential_frames()
        except ImportError as exc:
            raise SystemExit(str(exc)) from exc

    if run_all or args.log_iterations:
        if args.generate:
            from scripts.run_log_iterations import run_log_iteration_suite
            run_log_iteration_suite()
        plot_log_iterations()

    if run_all or args.comparison_methods:
        if args.generate:
            from scripts.run_comparison_methods import run_comparison_suite
            run_comparison_suite(n_trials=args.trials)
        plot_comparison_methods()

    if run_all or args.convergence_delta:
        plot_convergence_delta()

    if run_all or args.rates:
        if args.generate:
            from scripts.run_rates import run_rates_suite
            run_rates_suite(n_trials=args.trials)
        plot_rate_figures()


if __name__ == '__main__':
    main()
