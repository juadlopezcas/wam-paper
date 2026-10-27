# Synthetic orthogonal dictionary-learning results

Full run: 20 independent paired trials per setting, 50 additional starts for
the best-found reference. See README.md and configuration.json for the protocol.
All methods use orthogonal dictionaries; these results do not benchmark
unrestricted or overcomplete dictionary learning.

At dimension 10 and noise level 0.05, mean results are:

| Method | Dictionary error, 1-sparse codes | Dictionary error, 3-sparse codes | Held-out error, budget 1 on 1-sparse codes | Held-out error, budget 3 on 3-sparse codes |
| --- | ---: | ---: | ---: | ---: |
| GFA | 0.00000573 | 0.00256 | 0.002222 | 0.003548 |
| MSP | 0.0000102 | 0.00195 | 0.002227 | 0.003100 |
| Smoothed L1 | 0.00000821 | 0.0000110 | 0.002225 | 0.001685 |
| Procrustes | 0.00000573 | 0.00000830 | 0.002222 | 0.001683 |
| Unrotated | 0.813 | 0.797 | 0.6231 | 0.2560 |

GFA matches the best-found training objective for one-sparse data to numerical
precision (mean gap 1.33e-17). On three-sparse data, its mean one-sparse training
gap is 7.34e-6. This is evidence for effective optimization of its intended
one-sparse objective, not a global optimality certificate.

GFA remains trained for one-sparse reconstruction throughout. Its poorer
three-sparse dictionary recovery and budget-three reconstruction should not be
hidden: learning a good union of axes and recovering a dictionary generating
multi-sparse observations are different objectives. Procrustes is given the
generating support for recovery and the evaluated support for reconstruction.
Smoothed L1 is the documented smooth orthogonal baseline, not a reproduction
of the published nonsmooth subgradient algorithm.

Single-start timings do not establish equal-compute superiority. At dimension
80 on one-sparse data, mean fit times were GFA 3.03 s, MSP 0.321 s, Procrustes
0.227 s, and Smoothed L1 20.3 s. GFA therefore is not the fastest baseline here.

The reconstruction plot uses a new single-start Procrustes fit at each coding
budget. Some larger-budget fits reach inferior local minima, even when a
one-sparse-trained rotation gives a better representation. The best-found
reference includes those native rotations and multi-start refinements, selects
by training loss only, and is not a held-out oracle. Its search time is excluded
from ordinary fit timings.

The figures report means and approximate 95% confidence intervals across
independent datasets. Raw recovery and representation results are retained in
the adjacent CSV files.
