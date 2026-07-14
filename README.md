## Folder structure

- `ecf/` - the ECF detector: `known_k.py` (single known change) and
  `divisive.py` (unknown number of changes, binary segmentation +
  permutation test).
- `midast/` - `engine.py` (MIDAST calibration + KS detection, wraps
  `vendored_midast/`) and `cvm_fast.py` (vectorized, R-free re-implementation
  of MIDAST's Cramer-von-Mises test).
- `baselines/` - `e_divisive.py`, real e-Divisive (`ecp::e.divisive`) via
  rpy2.
- `common/` - data generators (sub-Gaussian, Student-t, Lomax) and MIDAST's
  Algorithm 1/2 calibration helpers.
- `vendored_midast/` - MIDAST source.
- `experiments/known_k/` - ECF-vs-MIDAST grid experiments for a single known
  change-point, and the script that plots them.
- `experiments/unknown_k/` - the unknown-number-of-changes benchmark
  (ECF-Divisive vs MIDAST[KS] vs MIDAST[CvM] vs e-Divisive) and the
  three-way comparison script.
- `comparative_plots/` - generated figures.

## Setup

```
pip install -r requirements.txt
```

`baselines/e_divisive.py` also needs R installed, with the `ecp` package:

```r
install.packages("ecp")
```

## ECF: cosine-only by default

`ecf.known_k.ECF` and `ecf.divisive.ECFCore` use cosine (real-part) random
Fourier features by default. The sine (imaginary) part carries no signal for
distributions symmetric about their centre, and an ablation confirmed
cosine-only matches or improves detection while roughly halving localization
error on the symmetric families we test (sub-Gaussian, Student-t). Lomax is
the one asymmetric family where the sine part helps; `run_pareto.py`
constructs `ECF(..., feature="cossin")` explicitly for this reason. Pass
`feature="cossin"` to either class to use the original real+imaginary
fingerprint elsewhere.

## MIDAST[CvM], R-free

MIDAST's `CramerTest` calls R's `cramer.test` through rpy2 for every sliding
window, recomputing the full pairwise distance matrix for every bootstrap
replicate. `midast/cvm_fast.py` precomputes the distance matrix once per
window and evaluates all bootstrap replicates as a single batched matrix
product, giving the same statistic (verified against R to 5+ decimal places)
roughly two orders of magnitude faster, with no R dependency. It also fixes
MIDAST[KS]'s power collapse at higher dimensions, since Cramer-von-Mises does
not carry KS's dimension-dependent critical-value penalty.

`midast.cvm_fast.FastCramerTest` is a drop-in replacement for
`CramerTest` in `vendored_midast/multivariate_tests_from_R.py` (same
constructor and `conduct_test` signature); `fast_cramer_scan` is a drop-in
replacement for `ChangeDetector(test_name="CramerTest").fit(...)`.

## Running

Known-k=1, per family (`--dim` in `{2, 10}`):

```
python experiments/known_k/run_subgaussian.py --dim 2
python experiments/known_k/run_pareto.py --dim 2
python experiments/known_k/run_studentt.py --dim 2
python experiments/known_k/make_comparative_plots.py
```

Unknown-k:

```
python experiments/unknown_k/benchmark.py --trials 50
python experiments/unknown_k/three_way_comparison.py --dim 2 --trials 50
```

All grid scripts write to `results/` at the project root, are checkpointed
(safe to interrupt and re-run), and are seeded, so re-running reproduces the
same results.
