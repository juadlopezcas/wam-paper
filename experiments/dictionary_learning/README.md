# Synthetic orthogonal dictionary-learning benchmark

> **Archived experiment.** The CSV results and configuration are intentionally
> retained because they were expensive to produce. The standalone runner and
> plotting scripts were removed during repository cleanup; the Q-DLA method
> itself remains in `scripts/methods.py` because it is used by the retained
> comparison-method figure.

## Model and paired protocol

Rows are observations. For a Haar-distributed orthogonal dictionary Q*, generate
Y = X Q*^T + (sigma / sqrt(k)) W. Every row of X has exactly s uniformly chosen
nonzero entries, with independent Gaussian amplitudes scaled by 1/sqrt(s).
Thus population mean signal energy per observation is one and sigma is the
population RMS noise-to-signal ratio. Both s=1 and s=3 are tested. This is an
exact-support model, not an independent Bernoulli-Gaussian model.

Train on 200k observations; test on max(10000, 200k) independent observations.
One global RMS scale, estimated from training observations, is applied to all
splits and all methods. No empirical centering, whitening, PCA, per-row normalization, or
ground-truth information is used for fitting. Population covariance is already
isotropic. Unrotated means Q=I, not an empirically fitted PCA dictionary.

Every method receives the same training data and the same random orthogonal
initialization. Different trials are independent; method comparisons are paired.
Configuration, seed, numerical-package versions, and thread count are recorded
in configuration.json. Dependencies are imported before timings; BLAS uses one
thread. No NYTimes inputs or outputs are used.

## Methods

- GFA: repository run_wfa1, minimizing mean squared distance to the nearest axis.
- MSP: polar projection of Y^T (YQ)^3, the complete orthogonal fourth-power
  dictionary-learning iteration of Zhai et al. (JMLR 2020):
  https://jmlr.org/papers/v21/19-755.html . Stops at normalized iterate change
  below 1e-8, or the iteration cap.
- SmoothL1: full-matrix objective mean sqrt((YQ)^2 + epsilon^2), optimized with
  the repository's Pymanopt steepest-descent machinery. Epsilon is 0.01 times
  training entry RMS. This is an explicitly smoothed L1 baseline, NOT a
  reproduction of the published columnwise nonsmooth subgradient algorithm.
- Procrustes: alternate A=H_s(YQ) and Q=polar(Y^T A). Stops at relative
  reconstruction-loss improvement below 1e-8 or the iteration cap. In recovery
  experiments it receives the generating support s, an explicitly favorable
  known-sparsity assumption. In representation plots it is refitted separately
  for every evaluated coding support.

All fitted dictionaries stay in O(k). The GFA one-sparse criterion and the
s=1 Procrustes criterion are identical; their comparison tests numerical
optimization rather than different statistical models. GFA remains trained on
one-sparse approximation even for three-sparse generating data. MSP and
SmoothL1 are not given the generating support.

This initial suite evaluates single-start performance. It does not yet compare
equal-time restart schedules or multiple amplitude distributions.

## Metrics and references

Matched dictionary error is mean_j ||qhat_j - sign_j q*_pi(j)||^2, minimized by
one-to-one Hungarian matching of absolute atom correlations. No atom may be
matched to more than one ground-truth atom. Recovery can disagree with GFA's
objective on multi-sparse data; that is objective mismatch, not automatically
solver failure.

Relative reconstruction error is ||Y-H_b(YQ)Q^T||_F^2 / ||Y||_F^2. Hard
thresholding is exact optimal b-sparse coding for an orthogonal dictionary.
Both noisy test error and clean test error are stored; the plotted primary
metric uses independent noisy observations. This is representation error, not
denoising error against clean targets. The generating dictionary is an
evaluation-only reference, not a certified optimal finite-sample rotation.

For representation settings (k=10, sigma=.05), construct a best-found rotation
for each coding support b=1,...,6. Seed the common hard-threshold/Procrustes
objective with every fitted method, the budget-matched Procrustes solution,
and 50 random orthogonal starts; retain the lowest training loss, including
the original candidates before refinement. Never use the true dictionary
or test data to choose it. Independent refinements run on four worker processes,
each limited to one BLAS thread. Ordinary method fits remain single-threaded.
Compatible completed sequential or batched checkpoints may be retained on
resume; their reference wall times are not comparable to the parallel reference
times and are not used in method runtime figures. This is NOT a global-optimality
certificate.
Report training loss minus best-found training loss, not a test 'optimality
gap'. The expensive reference runtime is stored separately. It is possible
for its held-out error to exceed another method's error.

Plots show means and normal-approximation 95% intervals across independent
datasets (1.96 standard errors). Log plots clip display values at 1e-10;
unclipped metrics are retained in CSVs. The configuration file specifies actual
trial counts. Runtime intervals are descriptive rather than a claim of normal
runtime distributions.

## Files and figures

- recovery_results.csv: per-trial recovery, runtime, test reconstruction,
  orthogonality, iteration counts, and termination diagnostics.
- representation_results.csv: per-trial, per-support train/test errors and
  best-found reference gaps.
- configuration.json: reproducibility parameters and versions.
- fig_dictionary_learning_comparison: recovery/runtime versus dimension/noise
  for one-sparse generating signals.
- fig_dictionary_learning_comparison_multisparse: same for three-sparse signals.
- fig_dictionary_learning_reconstruction: held-out error versus coding support.
- fig_dictionary_learning_optimization_gap: training-loss gaps to best found.

CSV checkpoints are written after completed trials. By default the runner
replaces these files. Add `--resume` to retain compatible completed trials:

```bash
python -m scripts.run_dictionary_learning --trials 20 --resume
```

Resume checks the numerical experiment configuration before continuing. An
interrupted, unfinished trial is recomputed, rather than appended partially.
