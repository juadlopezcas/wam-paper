# WMA Paper Figures

Code and saved experiment outputs for the retained figures from **Wasserstein
approximation of measures via orthogonal frames**.

## Setup

```bash
conda env create -f environment.yml
conda activate wamfigs
```

The environment includes Python, R, and `rpy2`; R is used for the
`stats::varimax` comparisons.

## Reproduce the retained figures

The default command redraws every retained figure from the existing experiment
outputs and cached NYTimes rotations:

```bash
python main.py
# equivalent to: python main.py --all
```

This plotting-only command does **not** regenerate any CSV file. The retained
figures and their individual commands are:

| Figure | Command | Saved input |
| --- | --- | --- |
| `fig_comparison_completing_methods.pdf` | `python main.py --comparison-methods` | `experiments/comparison_methods/*.csv` |
| `fig_log_iterations.pdf` | `python main.py --log-iterations` | `experiments/log_iterations/*.csv` |
| `fig_convergence_vs_delta.pdf` | `python main.py --convergence-delta` | `experiments/convergence_delta/*.csv` |
| `fig_exponential_frames.pdf` | `python main.py --exponential-frames` | deterministic synthetic sample |
| `fig_main_gradient.pdf` | `python main.py --main-gradient` | deterministic synthetic sample |
| `fig_nytimes_gfa_two_interpretations_pairplot.pdf` | `python main.py --nytimes` | `data/nytimes_pca.npz` and `experiments/nytimes_interpretability/full_data_rotations.npz` |
| `fig_rates_combined.pdf` | `python main.py --rates` | `experiments/rates/*.csv` |

Each command writes a PDF and PNG under `figures/`.

## Regenerate selected experiments

Saved CSVs are the default and remain in the repository because several
experiments are expensive. Add `--generate` only when you intentionally want to
rerun the experiment behind a selected figure:

```bash
python main.py --log-iterations --generate
python main.py --comparison-methods --generate --trials 10
python main.py --rates --generate --trials 10
python main.py --nytimes --generate
```

The last command refits the full-data GFA and R Varimax rotations for both
interpretations of the NYTimes decomposition:

- document coefficients, interpreted as sparse codes in dictionary learning;
- normalized word loadings, interpreted as sparse factors/loadings.

The comparison-method experiment can update only the saved alternating
minimization rows before redrawing the figure:

```bash
python -m scripts.run_comparison_methods --methods qdla --update-existing --trials 10
python main.py --comparison-methods
```

To rebuild the NYTimes PCA cache from the UCI bag-of-words data, run:

```bash
python main.py --prepare-nytimes
```

Pass `--generate` as well only to force a fresh download of the raw files.

## Tests

```bash
python -m unittest discover -s tests -v
```

The retained tests cover the Q-DLA hard-thresholding and alternating-minimization
implementation used in `fig_comparison_completing_methods.pdf`.

## Layout

```text
data/          synthetic generators and cached NYTimes decomposition
experiments/   saved experiment outputs (all historical CSVs are preserved)
figures/       generated PDF and PNG figures
scripts/       retained plotting and experiment code
tests/         checks for methods used by retained figures
main.py        command-line entry point
```

Historical experiment outputs may outlive their standalone plotting scripts;
this is intentional so time-consuming results are not lost. Shared plotting
style is defined in `scripts/plot_style.py`.
