# WAM Paper Figures

This repository contains the code and experiment outputs needed to reproduce the paper figures:

- `figures/fig_log_iterations.pdf`
- `figures/fig_comparison_completing_methods.pdf`
- `figures/fig_rates_combined.pdf`

The repository is designed so that figures can be regenerated quickly from the saved CSV files, while the scripts that regenerate those CSV files are also included for full reproducibility.

## Repository Layout

```text
data/                          Synthetic data generators used by the experiments
experiments/log_iterations/     CSV files for optimization error-vs-iteration plots
experiments/comparison_methods/ CSV files for method comparison and noise robustness plots
experiments/rates/              CSV files for statistical rate plots
figures/                        Generated PDF and PNG figures
scripts/                        Plotting scripts, experiment scripts, metrics, and shared style
main.py                         Command-line entrypoint
```

The shared Matplotlib style used by all figures is defined in:

```text
scripts/plot_style.py
```

## Environment

Create and activate the tested conda environment:

```bash
conda env create -f environment.yml
conda activate wamfigs
```

The environment pins the package versions used to smoke-test the plotting and CSV-regeneration scripts.

## Reproduce Figures From Saved CSVs

To regenerate all figures from the CSV files already included in `experiments/`, run:

```bash
python main.py --all
```

Equivalently, running `python main.py` with no flags regenerates all figures.

To regenerate one figure at a time:

```bash
python main.py --log-iterations
python main.py --comparison-methods
python main.py --rates
```

The output files are written to `figures/` as both PDF and PNG files.

## Figure Contents

### `fig_log_iterations.pdf`

This figure shows how the normalized error evolves over optimization iterations for three synthetic data distributions:

```text
Gaussian mixture
Double exponential
Student-t
```

The top row fixes the dimension at `d = 15` and varies the number of samples:

```text
n = 2000, 4000, 8000, 16000, 32000
```

The bottom row fixes the sample size at `n = 5000` and varies the dimension:

```text
d = 12, 25, 50, 75, 100
```

The source CSV files are stored in:

```text
experiments/log_iterations/
```

### `fig_comparison_completing_methods.pdf`

This figure compares frame recovery methods by mean frame error and runtime. The methods are:

```text
Ours
RSC
Deflation Varimax
PCA+Varimax
```

The dimension-scaling panels use Gaussian data with:

```text
sample_size = 1000
noise_level = 0.05
dimensions = 2, 5, 10, 15, 25, 50
```

The noise-robustness panels use:

```text
dimension = 5
noise_levels = 0.01, 0.05, 0.1, 0.2, 0.3
```

The source CSV files are stored in:

```text
experiments/comparison_methods/
```

### `fig_rates_combined.pdf`

This figure summarizes statistical rate behavior. Both panels use log-log axes with base 2.

The left panel plots relative error for dimensions:

```text
d = 2, 25, 50, 75, 100
```

and sample sizes:

```text
n = 100, 200, 400, 800, 1600, 3200, 6400
```

The right panel fixes `d = 5` and compares:

```text
Gaussian mixture
Double exponential
Student-t
```

using steepest descent with `delta = 0.1` and `10` trials. The plotted y-axis is relative error, computed as `sqrt(Normalized Error)` where applicable.

The source CSV files are stored in:

```text
experiments/rates/
```

## Regenerate CSV Files

The included CSV files are enough to reproduce the figures. Regenerating them is optional and can be computationally expensive, especially for the method-comparison experiments involving RSC.

To regenerate CSVs for one figure and then plot that figure:

```bash
python main.py --log-iterations --generate
python main.py --comparison-methods --generate
python main.py --rates --generate
```

To regenerate all CSVs and all figures:

```bash
python main.py --all --generate
```

For quick checks, use smaller trial counts where supported:

```bash
python main.py --comparison-methods --generate --trials 1
python main.py --rates --generate --trials 1
```

The experiment scripts can also be run directly:

```bash
python -m scripts.run_log_iterations --single --data-type gaussian --n-points 2000 --dim 15
python -m scripts.run_comparison_methods --frame-only --trials 1
python -m scripts.run_rates --single-dim 2
python -m scripts.run_rates --summary-only --trials 1
```

These commands write CSV files under the corresponding `experiments/` subfolder.
