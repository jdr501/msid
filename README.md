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

irf = res.irf(horizon=30, ci=0.68, n_boot=1000)   # wild-bootstrap bands
irf.plot()

res.plot_regimes(series=y.columns[0], threshold=0.7)
```

### Shock ordering

B is identified only up to column permutation and sign, so the order in
which shocks come out of the optimizer is arbitrary. Two conventions are
available for fixing it (plus explicit permutations):

```python
# Cholesky-style: shock j = the shock of variable j (assignment by
# scale-normalized impact shares; warns if the matching is ambiguous)
res = model.fit(n_starts=50, random_state=0, shock_order="variables")

# HL volatility labeling: sort shocks by their regime-2 relative
# variance lambda_2j, largest first
res = model.fit(n_starts=50, random_state=0, shock_order="lambda_desc")

# or reorder after the fact
res.order_shocks_by_variables()      # same as shock_order="variables"
res.sort_shocks(regime=2)            # same as "lambda_desc"
res.reorder_shocks([1, 2, 0])        # any explicit permutation
```

`"variables"` matches how conventional-SVAR readers expect shocks to be
labeled and is the natural display convention; `"lambda_desc"` is
regime-based and useful as a robustness/diagnostic device. Both are only
as credible as the statistics behind them — check the identification block
in `summary()` (distinct λ's) and heed the ambiguity warning. See
`THEORY.md`, Section 5.

## What's in the box

| Area | Where |
| --- | --- |
| MSVECM / MSVAR estimation (EM, multi-start, log-Λ parameterization, autograd gradients) | `msid.model`, `msid.estimation` |
| Zero / sign / long-run restrictions on B and Ξ | `msid.Restrictions` |
| Identification Wald tests for λ distinctness (auto in `summary()`) | `msid.inference.wald_lambda` |
| LR tests of economic restrictions, AIC/SC model tables | `msid.compare_models`, `msid.model_table` |
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
