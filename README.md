# WAM Paper Figures

Code and saved experiment outputs to reproduce figures from **Wasserstein approximation of measures via orthogonal frames**.

## Setup

```bash
conda env create -f environment.yml
conda activate wamfigs
```

## Reproduce Figures

Regenerate all figures from the saved CSV files:

```bash
python main.py
```

or equivalently:

```bash
python main.py --all
```

Generate individual figures:

```bash
python main.py --main-figure
python main.py --log-iterations
python main.py --comparison-methods
python main.py --rates
python main.py --nytimes  # requires data/nytimes_pca.npz
```

Figures are saved in `figures/` as PDF and PNG files, including `fig_main.pdf` for the introductory problem figure. The NYTimes figure is optional because it requires the cached PCA factor file `data/nytimes_pca.npz`.

## Regenerate Experiment CSVs

The CSV files needed for the figures are already included under `experiments/`. To rerun the experiments and then recreate the figures:

```bash
python main.py --all --generate
```

or for one figure at a time:

```bash
python main.py --log-iterations --generate
python main.py --comparison-methods --generate
python main.py --rates --generate
```

Some experiments, especially the comparison methods using RSC, can take a long time.

## Layout

```text
data/          synthetic data generators
experiments/   saved CSV files and regenerated experiment outputs
figures/       generated figures
scripts/       plotting and experiment scripts
main.py        command-line entrypoint
```

The shared plotting style is in `scripts/plot_style.py`.
