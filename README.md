# msid — Markov-Switching Statistical Identification for SVAR/SVECM models

[![CI](https://github.com/jdr501/msid/actions/workflows/ci.yml/badge.svg)](https://github.com/jdr501/msid/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/msid.svg)](https://pypi.org/project/msid/)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://pypi.org/project/msid/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

`msid` implements the identification-through-heteroskedasticity framework of
Herwartz & Lütkepohl (2014, *Journal of Econometrics* 183, "HL") with the
estimation improvements and additional tests developed in Rajapaksa & Shao
(2026). Distinct volatility regimes, modeled by a first-order Markov chain,
statistically identify the structural impact matrix **B** without exclusion
restrictions — and make conventional (zero, sign, long-run) restrictions
*testable*.

Supports **SVAR and SVECM**, any dimension **K ≥ 2**, and **2–4 volatility
regimes**. Pure Python (no compiled extensions), Python ≥ 3.10, MIT license.

## Install

```bash
pip install msid
```

Or from source:

```bash
git clone https://github.com/jdr501/msid.git
cd msid
pip install -e .
```

An optional `msid[jax]` extra is declared for a future JAX autodiff backend;
the current release uses `autograd` throughout (exact reverse-mode gradients
for the M-step and the OPG scores — no finite differences anywhere).

## Quickstart

```python
import msid

y = msid.load_example()                       # bundled simulated daily data

model = msid.MSVECM(y, lags=1, coint_rank=1,  # Johansen beta, held fixed
                    n_regimes=2, deterministic="ct")
res = model.fit(n_starts=50, random_state=0)  # multi-start EM

res.summary()                                 # estimates + identification tests

# test an overidentifying restriction set against the statistically
# identified model
R = msid.Restrictions(K=3)
R.b_zeros([(2, 0), (2, 1)])                   # b31 = b32 = 0 (0-indexed)
lr_table, res_r = res.test_restrictions(R)
print(lr_table)

# same test against a SIMULATED null instead of chi-squared -- worth the
# compute whenever the conclusion rests on a non-rejection
boot, res_r = res.test_restrictions_bootstrap(R, n_boot=499)
print(boot)

irf = res.irf(horizon=30, ci=0.68, n_boot=1000)   # wild-bootstrap bands
irf.plot()

res.plot_regimes(series=y.columns[0], threshold=0.7)
```

### Shock ordering

B is identified only up to column permutation and sign, so the order in
which shocks come out of the optimizer is arbitrary — and every LR
statistic is *invariant* to that order, so no test can validate a
labeling. Fix it first, by a rule stated in advance:

```python
# Recommended: standardize B by variable scale, score every assignment of
# shocks to variables by total own-variable impact, take the maximizer,
# orient signs so each own impact is positive.
res = model.fit(n_starts=50, random_state=0, shock_order="max_own_impact")
print(res.labeling_report_)   # best vs second-best score, gap, own shares

# Legacy Hungarian form of the same rule: identical permutation, no report
res = model.fit(n_starts=50, random_state=0, shock_order="variables")

# HL volatility labeling: sort shocks by their regime-2 relative
# variance lambda_2j, largest first
res = model.fit(n_starts=50, random_state=0, shock_order="lambda_desc")

# or reorder after the fact
res.order_shocks_by_variables()      # same as shock_order="variables"
res.sort_shocks(regime=2)            # same as "lambda_desc"
res.reorder_shocks([1, 2, 0])        # any explicit permutation
```

`"max_own_impact"` labels shocks the way conventional-SVAR readers expect
and reports how sharply it did so: the assignment score runs from about 1
(shocks spread evenly over the variables) to K (a clean one-to-one match),
and the best-minus-second-best gap is the credibility statistic for the
labels. When that gap falls below `min_gap` the labeling is flagged
ambiguous — the declared fallback is `"lambda_desc"`, reporting the shocks
as statistical objects and dropping variable-indexed restrictions such as
`b_32 = 0`. `"lambda_desc"` is regime-based and also useful as an
independent robustness check: agreement between the two orderings is
stronger evidence than either alone. Both are only as credible as the
statistics behind them — check the identification block in `summary()`
(distinct λ's) too. See `THEORY.md`, Section 5.

## What's in the box

| Area | Where |
| --- | --- |
| MSVECM / MSVAR estimation (EM, multi-start, log-Λ parameterization, autograd gradients) | `msid.model`, `msid.estimation` |
| Zero / sign / long-run restrictions on B and Ξ | `msid.Restrictions` |
| Identification Wald tests for λ distinctness (auto in `summary()`) | `msid.inference.wald_lambda` |
| Deterministic shock labeling + best/second-best diagnostics | `msid.labeling`, `results.order_shocks_by_max_own_impact()` |
| LR tests of economic restrictions, AIC/SC model tables | `msid.compare_models`, `msid.model_table` |
| Bootstrap null distribution for those LR tests, with nesting repair | `msid.bootstrap_lr_test`, `results.test_restrictions_bootstrap(...)` |
| State-invariance LR test (M ≥ 3), bootstrap overidentification J-test | `results.test_b_invariance()`, `results.test_overidentification()` |
| Overlapping-window Wald test for temporal stability of B | `results.test_b_stability(...)` |
| Rolling-window coefficient stability diagnostics | `results.rolling_stability(...)` |
| Fixed-design wild bootstrap IRF bands, conditional FEVDs | `results.irf(...)`, `results.fevd(...)` |
| OPG standard errors (autograd scores, block-diagonal) | `results.std_errors_` |
| Johansen/lag-order/ARCH-LM/White pre-tests | `msid.pretest` |

See `THEORY.md` for the identification logic and conventions, and
`notebooks/` for one executable tutorial per workflow.

## Development

```bash
pip install -e ".[dev]"
pytest -m "not slow"      # fast suite (runs in CI on 3.10–3.12)
ruff check msid && black --check msid
```

## Citing

If you use `msid` in academic work, please cite the software and the paper
whose methodology it implements (see also `CITATION.cff`):

```bibtex
@software{rajapaksa2026msid,
  author  = {Rajapaksa, Danusha},
  title   = {msid: Markov-Switching Statistical Identification for
             Structural VAR/VECM Models},
  year    = {2026},
  version = {0.1.0},
  url     = {https://github.com/jdr501/msid}
}

@unpublished{rajapaksa2026tether,
  author = {Rajapaksa, Danusha and Shao, Enchuan},
  title  = {The Microstructure of Stablecoin Stability: Evidence from Tether},
  year   = {2026},
  note   = {Working paper}
}
```

## References

- Herwartz, H., Lütkepohl, H. (2014). Structural vector autoregressions with
  Markov switching: Combining conventional with statistical identification of
  shocks. *Journal of Econometrics* 183, 104–116.
- Lanne, M., Lütkepohl, H., Maciejowska, K. (2010). Structural vector
  autoregressions with Markov switching. *JEDC* 34, 121–131.
- Rajapaksa, D., Shao, E. (2026). The Microstructure of Stablecoin Stability:
  Evidence from Tether. Working paper.
