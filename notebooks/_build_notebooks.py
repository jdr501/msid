"""Generate the five tutorial notebooks (spec Section 13).

Run once:  python notebooks/_build_notebooks.py
Each notebook is executable top-to-bottom on the bundled example data.
"""

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def nb(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}


def code(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": src.splitlines(keepends=True)}


NOTEBOOKS = {
    "01_msvecm_johansen_beta.ipynb": [
        md("# MSVECM with Johansen $\\beta$\n\n"
           "Fits a 2-regime Markov-switching SVECM on the bundled example data, "
           "with the cointegration vector estimated by Johansen's procedure and "
           "held fixed through the EM iterations (the two-step approach of "
           "Rajapaksa & Shao 2026)."),
        code("import numpy as np\nimport msid\n\ny = msid.load_example()\ny.plot(subplots=True, figsize=(9, 6));"),
        md("## Pre-tests\nJohansen cointegration test and residual heteroskedasticity "
           "diagnostics that motivate the MS specification."),
        code("print(msid.johansen(y, det_order=1, k_ar_diff=1))\n"
             "msid.justify_ms(y, lags=1);"),
        md("## Fit\n`coint_rank=None` would select the rank by trace test; here we fix r=1."),
        code("model = msid.MSVECM(y, lags=1, coint_rank=1, n_regimes=2, deterministic='ct')\n"
             "res = model.fit(n_starts=20, random_state=0)\n"
             "res.summary();"),
        md("## Regimes and impulse responses"),
        code("res.plot_regimes(series=y.columns[0], threshold=0.7);"),
        code("irf = res.irf(horizon=30, ci=0.68, n_boot=200, random_state=1)\n"
             "irf.plot();"),
        code("res.fevd(horizon=12, by_state=True).query('horizon == 12')"),
    ],
    "02_msvecm_user_beta.ipynb": [
        md("# MSVECM with user-supplied $\\beta$\n\n"
           "Theory-imposed cointegration vectors skip Johansen estimation entirely "
           "-- e.g. HL (2014) impose $\\beta = (0,0,0,1)'$ for a stationary "
           "interest rate. Here we impose the (approximate) true vector of the "
           "example DGP."),
        code("import numpy as np\nimport msid\n\ny = msid.load_example()\n"
             "beta = np.array([[1.0], [-0.5], [-0.2]])\n"
             "model = msid.MSVECM(y, lags=1, beta=beta, n_regimes=2)\n"
             "print(model.beta_source)   # 'user'\n"
             "res = model.fit(n_starts=20, random_state=0)\n"
             "res.summary();"),
        md("Compare with the Johansen-based fit: the log-likelihoods should be "
           "very close when the imposed vector is (nearly) correct."),
        code("res_j = msid.MSVECM(y, lags=1, coint_rank=1, n_regimes=2).fit(\n"
             "    n_starts=20, random_state=0)\n"
             "print(f'user beta: {res.loglik_:.2f}   Johansen beta: {res_j.loglik_:.2f}')"),
    ],
    "03_msvar.ipynb": [
        md("# MSVAR in levels\n\nThe levels VAR is a first-class specification: "
           "same EM loop, likelihood, bootstrap and tests as the VECM, with "
           "$\\Xi = (I - \\sum A_i)^{-1} B$ as the long-run matrix (stationarity "
           "required)."),
        code("import numpy as np\nimport msid\n"
             "from msid.datasets.simulate import simulate_msvar\n\n"
             "A = [np.array([[0.5, 0.1], [0.0, 0.4]])]\n"
             "B = np.array([[1.0, 0.0], [0.4, 0.8]])\n"
             "Lambdas = [np.ones(2), np.array([8.0, 0.2])]\n"
             "P = np.array([[0.95, 0.1], [0.05, 0.9]])\n"
             "y, states = simulate_msvar(700, A, B, Lambdas, P, random_state=11)\n"
             "model = msid.MSVAR(y, lags=1, n_regimes=2, deterministic='c')\n"
             "res = model.fit(n_starts=20, random_state=0)\n"
             "res.summary();"),
        code("print('long-run matrix Xi:')\nprint(np.round(res.Xi_, 3))\n"
             "res.irf(horizon=15, n_boot=200, random_state=2).plot();"),
    ],
    "04_testing_restrictions.ipynb": [
        md("# Testing competing restriction sets\n\nWith MS-identification any "
           "zero restriction on $B$ or $\\Xi$ is overidentifying and testable. "
           "Sign restrictions are normalizations (0 df)."),
        code("import numpy as np\nimport msid\n\ny = msid.load_example()\n"
             "model = msid.MSVECM(y, lags=1, coint_rank=1, n_regimes=2)\n"
             "res_u = model.fit(n_starts=20, random_state=0)"),
        md("## Identification diagnostics first\nLR tests are only as good as "
           "the distinctness of the $\\lambda$'s."),
        code("print(res_u.identification_)"),
        md("## Two competing hypotheses\n(Tether-paper style: a 'triangular' set "
           "with one sign + one zero, and a 'conventional' set with two zeros.)"),
        code("R_tri = msid.Restrictions(3).b_zeros([(2, 1)]).b_signs({(2, 0): '+'})\n"
             "R_conv = msid.Restrictions(3).b_zeros([(2, 0), (2, 1)])\n"
             "t1, res_tri = res_u.test_restrictions(R_tri, n_starts=10, random_state=1)\n"
             "t2, res_conv = res_u.test_restrictions(R_conv, n_starts=10, random_state=1)\n"
             "import pandas as pd\npd.concat([t1, t2], ignore_index=True)"),
        md("## Model comparison table (HL Table 1 layout)"),
        code("msid.model_table([res_u, res_tri, res_conv],\n"
             "                 names=['unrestricted', 'triangular', 'conventional'])"),
    ],
    "05_invariance_stability.ipynb": [
        md("# Invariance and stability testing\n\nThe maintained assumption is a "
           "regime-invariant impact matrix $B$: regimes shift shock variances, "
           "not the transmission mechanism. Three checks (spec Section 6)."),
        code("import numpy as np\nimport msid\n"
             "from msid.datasets.simulate import default_msvecm_params, simulate_msvecm\n\n"
             "pars = default_msvecm_params(M=3)\n"
             "y, _ = simulate_msvecm(T=900, alpha=pars['alpha'], beta=pars['beta'],\n"
             "                       Gammas=pars['Gammas'], B=pars['B'],\n"
             "                       Lambdas=pars['Lambdas'], P=pars['P'], random_state=7)\n"
             "res3 = msid.MSVECM(y, lags=1, coint_rank=1, n_regimes=3).fit(\n"
             "    n_starts=10, random_state=0)"),
        md("## 1. LR test for state-invariant B (M >= 3)"),
        code("print(res3.test_b_invariance())"),
        md("## 2. Bootstrap overidentification J-test (Tether App. F.1)"),
        code("print(res3.test_overidentification(n_boot=500, random_state=1))"),
        md("## 3. Overlapping-window Wald test (Tether App. F.2) and rolling "
           "coefficient diagnostics (App. E)"),
        code("res2 = msid.MSVECM(msid.load_example(), lags=1, coint_rank=1,\n"
             "                   n_regimes=2).fit(n_starts=10, random_state=0)\n"
             "print(res2.test_b_stability('2020-05-15', window_before=(8, 3),\n"
             "                            window_after=(3, 8), n_boot=200,\n"
             "                            min_regime_obs=10, random_state=2))"),
        code("roll = res2.rolling_stability(window=300, lags=2)\n"
             "roll.plot(equation=res2.model.y.columns[0]);"),
    ],
}


def main():
    for name, cells in NOTEBOOKS.items():
        path = os.path.join(HERE, name)
        with open(path, "w") as fh:
            json.dump(nb(cells), fh, indent=1)
        print("wrote", path)


if __name__ == "__main__":
    main()
