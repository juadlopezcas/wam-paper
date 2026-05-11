import argparse

from scripts.plot_comparison_methods import plot_comparison_methods
from scripts.plot_log_iterations import plot_log_iterations
from scripts.plot_main_figure import create_main_figure
from scripts.plot_nytimes_gfa import plot_nytimes_pairplots
from scripts.plot_rate_figures import plot_rate_figures


def parse_args():
    parser = argparse.ArgumentParser(description='Reproduce WAM paper figures.')
    parser.add_argument('--all', action='store_true', help='Generate all figures from existing CSVs.')
    parser.add_argument('--log-iterations', action='store_true', help='Generate fig_log_iterations.')
    parser.add_argument('--comparison-methods', action='store_true', help='Generate fig_comparison_completing_methods.')
    parser.add_argument('--rates', action='store_true', help='Generate fig_rates_combined.')
    parser.add_argument('--main-figure', action='store_true', help='Generate fig_main.')
    parser.add_argument('--nytimes', action='store_true', help='Generate fig_nytimes_gfa_pairplot from data/nytimes_pca.npz.')
    parser.add_argument('--generate', action='store_true', help='Regenerate CSVs before plotting selected figures.')
    parser.add_argument('--trials', type=int, default=10, help='Trials for regenerated comparison/rate summaries.')
    return parser.parse_args()


def main():
    args = parse_args()
    run_all = args.all or not (args.log_iterations or args.comparison_methods or args.rates or args.main_figure or args.nytimes)

    if run_all or args.main_figure:
        create_main_figure()

    if args.nytimes:
        plot_nytimes_pairplots()

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

    if run_all or args.rates:
        if args.generate:
            from scripts.run_rates import run_rates_suite
            run_rates_suite(n_trials=args.trials)
        plot_rate_figures()


if __name__ == '__main__':
    main()
