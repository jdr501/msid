"""Nested LR tests of economic restrictions and model-comparison tables.

``LR = 2 |log L_T - log L^r_T|`` with degrees of freedom equal to the number
of **zero** restrictions only: sign restrictions are pure normalizations and
carry 0 df (spec Section 7).

Note on the df convention: the Tether paper reports df = 2 for BOTH its
triangular model (b31 > 0, b32 = 0 -- one sign + one zero) and its
conventional model (b31 = b32 = 0 -- two zeros), i.e. it follows the printed
convention of its Table 3.  By default this module counts only zero
restrictions (df = 1 and 2 respectively for those models); pass
``df_override`` to reproduce a paper's printed df exactly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

__all__ = ["compare_models", "model_table"]


def _describe_h0(res) -> str:
    R = res.restrictions
    parts = [f"b{i + 1}{j + 1}=0" for (i, j) in R.b_zero_indices]
    parts += [f"xi{i + 1}{j + 1}=0" for (i, j) in R.xi_zero_indices]
    parts += [
        f"b{i + 1}{j + 1}{'>' if s > 0 else '<'}0" for (i, j), s in R.sign_restrictions.items()
    ]
    return ", ".join(parts) if parts else "unrestricted"


def compare_models(unrestricted, restricted, df_override=None, names=None) -> pd.DataFrame:
    """LR tests of each restricted fit against the unrestricted fit.

    Parameters
    ----------
    unrestricted : results object (H1).
    restricted : results object or list of results objects (H0).
    df_override : int, list of int, or None
        Override the automatic df (= number of zero restrictions in H0 minus
        those in H1).  See the module docstring for the sign-restriction df
        subtlety in the Tether paper's Table 3.
    names : list of str, optional row labels.

    Returns
    -------
    DataFrame in the layout of Tether Table 3 / HL Table 6:
    model, H0, H1, LR, df, p-value.
    """
    if not isinstance(restricted, (list, tuple)):
        restricted = [restricted]
    if df_override is not None and not isinstance(df_override, (list, tuple)):
        df_override = [df_override] * len(restricted)
    h1_desc = _describe_h0(unrestricted)
    rows = []
    for i, res in enumerate(restricted):
        lr = 2.0 * abs(unrestricted.loglik_ - res.loglik_)
        df = res.restrictions.n_zero_restrictions - unrestricted.restrictions.n_zero_restrictions
        if df_override is not None:
            df = int(df_override[i])
        if df <= 0:
            p = np.nan
        else:
            p = float(stats.chi2.sf(lr, df))
        rows.append(
            {
                "model": names[i] if names else f"restricted {i + 1}",
                "H0": _describe_h0(res),
                "H1": h1_desc,
                "LR": lr,
                "df": df,
                "p-value": p,
            }
        )
    return pd.DataFrame(rows)


def model_table(fits, names=None) -> pd.DataFrame:
    """Model comparison: log L, free parameters, AIC, SC (HL Table 1 layout).

    AIC = -2 log L + 2 k;  SC = -2 log L + log(T) k, with free-parameter
    counts k taken from each fit's Restrictions object.
    """
    rows = []
    for i, res in enumerate(fits):
        rows.append(
            {
                "model": names[i] if names else _describe_h0(res),
                "logL": res.loglik_,
                "k": res.n_free_params_,
                "AIC": res.aic_,
                "SC": res.sc_,
            }
        )
    return pd.DataFrame(rows)
