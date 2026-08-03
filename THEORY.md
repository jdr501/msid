# Theory notes for `msid`

This note summarizes the identification logic behind the package and the
conventions that a user needs to interpret results correctly. Equation
references are to Herwartz & Lütkepohl (2014, *J. Econometrics* 183; "HL")
and to the appendices of Rajapaksa & Shao (2026; "RS").

## 1. Identification through Markov-switching heteroskedasticity

The reduced-form errors follow `u_t | s_t ~ N(0, Σ_{s_t})` (HL Eq. 4) with a
first-order M-state Markov chain `s_t` and

```
Σ_1 = B B',    Σ_m = B Λ_m B',   m = 2, …, M          (HL Eq. 5)
```

where `Λ_m = diag(λ_m1, …, λ_mK)` is positive and `Λ_1 ≡ I_K` normalizes the
structural shocks `ε_t = B^{-1} u_t` to unit variance in state 1. The
`λ_mi` are the *relative variances* of the shocks in state m versus state 1.

**Uniqueness conditions.**

- **M = 2:** the decomposition is unique — up to changes of column signs and
  simultaneous permutations of the `λ_2i` and the columns of B — iff the
  `λ_2i` are *all distinct* (HL Section 3.1).
- **M ≥ 3:** uniqueness (up to sign) requires that for every pair of
  subscripts `k ≠ l` there is *some* regime `j ∈ {2,…,M}` with
  `λ_jk ≠ λ_jl` (Lanne, Lütkepohl & Maciejowska 2010, Proposition 1).

These are testable conditions: `results.summary()` always prints an
"Identification status" block with the corresponding Wald tests — pairwise
`λ_2i = λ_2j` for M = 2 (HL Table 4 layout), joint pairwise tests across
regimes plus per-regime all-equal tests for M ≥ 3 (HL Table 5 layout). If
equality cannot be rejected, B is only **set-identified** and p-values of
restricted-model LR tests should be read as *upper bounds*: were some λ's
equal, the LR degrees of freedom would shrink and true p-values would be
even smaller (HL Section 4.2). Rejections therefore survive; non-rejections
need caution.

Because any admissible B fits the data equally well up to column
permutation and sign, *economic labels* for the shocks must come from
outside the statistical model — from volatility timing (which regime is
turbulent for which shock) or from the impulse responses themselves.

## 2. Why restrictions become testable

In a homoskedastic SVAR, `Σ = BB'` has K(K+1)/2 free elements against K²
parameters in B: exclusion restrictions are *just*-identifying and cannot be
tested. With M ≥ 2 volatility states, B is already (statistically)
identified, so **any** zero restriction on B or on the long-run matrix

```
Ξ = β⊥ [α'⊥ (I_K − ΣΓ_i) β⊥]^{-1} α'⊥ B     (VECM; HL Eq. 2)
Ξ = (I_K − ΣA_i)^{-1} B                      (stationary levels VAR)
```

is *over*-identifying and testable by LR: `LR = 2(log L_T − log L^r_T)`,
asymptotically χ² with df = number of zero restrictions.

**Why sign restrictions carry no degrees of freedom.** A sign restriction
like `b_31 > 0` selects among the 2^K observationally equivalent column-sign
configurations of an already-identified B; it removes no free parameter and
changes no likelihood value. `msid` therefore implements signs as post-hoc
column flips (never as constrained optimization) and counts df from zero
restrictions only. (The Tether paper's Table 3 prints df = 2 for both its
triangular — one sign + one zero — and conventional — two zeros — models;
`compare_models(df_override=…)` reproduces any printed convention.)

## 3. Regime-invariance of B

The maintained assumption is that regimes shift shock *intensities* (Λ_m),
not the transmission mechanism (B). Three checks:

1. **LR test (M ≥ 3 only):** the unstructured MS model with free Σ_m nests
   the common-B decomposition; df = ½MK(K+1) − K² − (M−1)K (HL Eq. 6). With
   M = 2 the decomposition is exact — nothing to test.
2. **Bootstrap J-test (RS App. F.1):** with M ≥ 3 the extra covariances
   yield q = (M−2)K(K−1)/2 overidentifying restrictions
   `vec_off(B^{-1}Σ_m B^{-1'}) = 0, m ≥ 3`; Ω is estimated by parametric
   bootstrap under the null.
3. **Overlapping-window Wald test (RS App. F.2):** B is re-estimated in two
   calendar windows straddling a candidate transition date;
   `W = Δ'V[Δ]^{-1}Δ` with V from a parametric bootstrap drawing
   `u*_t ~ N(0, Σ_m p_{t|T}(m) Σ_m)`. Windows overlap by design so each
   contains observations from both regimes — B is not identified inside a
   single-regime window, and `msid` refuses to run if a window lacks
   high-probability observations from every regime.

## 4. Estimation conventions

- **EM algorithm** (HL Appendix; RS Online App. A, Algorithm 1): Hamilton
  filter forward, Kim smoother backward, closed-form updates for P and θ
  (ξ-weighted GLS), numerical minimization of `l(B, Λ_2…Λ_M)` for the
  structural block.
- **Log-λ parameterization** (RS improvement): optimization runs over
  `log λ_mi`, replacing HL's hard 0.01 lower bound; positivity holds at
  every line-search point, and gradients are exact autograd derivatives
  (no finite differences). Ill-conditioned Σ_m steps (condition number
  > 1e12) are rejected.
- **Randomized Λ⁰_m starts** (RS improvement): identity starts tend to
  collapse to the single-regime OLS solution; `msid` draws
  LogUniform(0.1, 10) diagonals by default and runs a multi-start manager
  (`n_starts`, all converged log-likelihoods kept in
  `results.loglik_starts_`).
- **Two-step β:** for the MSVECM, β is estimated once by Johansen reduced-
  rank regression (or supplied by the user) and **held fixed** during EM,
  as in RS.
- **Long-run zeros:** given θ, `Ξ = C(θ)B` is linear in B, so Ξ-zeros enter
  the M-step as a quadratic penalty whose weight escalates across EM
  iterations until the constraint holds to 1e-8 (verified at convergence,
  error otherwise).

## 5. Label-switching conventions

The states of a Markov mixture are exchangeable; two rules pin them down:

- `label_order="lambda_sort"` (default): states ordered by total state
  variance `tr(Σ_m) = tr(BΛ_mB')`, ascending. This is deterministic and —
  unlike sorting on tr(Λ_m) — invariant to which regime an EM run happened
  to use as its Λ_1 = I baseline; when the ordering changes, the
  normalization is restored via `B ← B·diag(Λ_new1)^{1/2}`,
  `Λ_m ← Λ_m / Λ_new1`.
- `label_order="terminal_prob"`: HL's rule `ξ_{iT|T} ≤ ξ_{jT|T}` for i < j.

Column signs of B are normalized by the user's sign restrictions when
present, else to a positive diagonal.

### Shock (column) ordering

Distinct from *state* labels, the **columns of B** — the shocks — are also
only identified up to permutation: nothing in the likelihood ties column j
to variable j. The estimator therefore keeps whatever arrangement the
optimizer converged to (`shock_order=None`), and two explicit conventions
are offered for pinning it down:

- **`shock_order="variables"`** — Cholesky-style labels. Each column is
  assigned to the variable it impacts most on impact, using impact *shares*
  normalized by each variable's residual standard deviation (so the
  assignment is unit-free) and a Hungarian best-assignment across columns.
  After reordering, "shock j" is the shock of variable j, which is the
  convention readers of conventional SVARs expect and the one used in the
  Tether paper's Figure 7. This is only meaningful when normalized B is
  roughly diagonal-dominant — each statistical shock loading mainly on one
  distinct variable. When the winning assignment beats the best alternative
  by a thin margin (< 0.1 in impact share by default), a warning is raised:
  the statistical shocks are then likely mixtures of economic shocks and
  variable labels should not be trusted (HL Sec. 3.1's caution).
- **`shock_order="lambda_desc"` / `"lambda_asc"`** — HL's volatility
  labeling. Shocks are sorted by their relative variance λ_2j in regime 2
  (largest or smallest first). This never suffers from assignment
  ambiguity, because it sorts on an identified scalar per shock — but the
  resulting order is regime-specific and carries no variable meaning by
  itself; it is most useful as a robustness/diagnostic device, and only
  credible when the λ's are statistically distinct (check the
  identification block first).

An explicit permutation (e.g. `shock_order=[1, 2, 0]`) is also accepted,
and the same operations are available post-fit as
`results.order_shocks_by_variables()`, `results.sort_shocks()` and
`results.reorder_shocks()`. All of them permute B's columns, the λ's, the
structural shocks and cached standard errors consistently; the Σ_m are
invariant by construction. On a restricted fit, note that positions in the
`Restrictions` object refer to the original column order.

## 6. Why Λ and P are held fixed in the bootstrap

The IRF bootstrap (fixed-design wild bootstrap, Rademacher weights; HL
Eq. 7) re-estimates only θ* and B* from each pseudo-sample, warm-started at
the ML estimates, with `Λ̂_m` and `P̂` frozen. Both papers do this because
(i) the IRFs are functions of (θ, B) only — Λ and P enter only through the
weighting of observations; (ii) re-estimating the full MS structure per
replication risks label switching, boundary solutions and local optima
that would contaminate the bands with parameterization changes rather than
sampling uncertainty; and (iii) it keeps 1000 replications computationally
feasible. Sign normalization is enforced against the original B̂ in every
replication so column flips cannot widen the bands artificially.

## 7. Model selection

Standard tests for the *number of states* are non-regular (parameters
unidentified under the null), so `msid` follows HL in comparing AIC/SC
across M (`msid.model_table`). Note that for under-identified models the
criteria mechanically prefer restricted versions — check the identification
block before reading restriction-selection from AIC/SC (HL Section 3.3).
