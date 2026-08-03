"""Bundled example data and the Tether-series download stub."""

from __future__ import annotations

import os

import pandas as pd

from .simulate import default_msvecm_params, simulate_msvar, simulate_msvecm

__all__ = [
    "default_msvecm_params",
    "load_example",
    "load_tether",
    "simulate_msvar",
    "simulate_msvecm",
]

_HERE = os.path.dirname(__file__)
_EXAMPLE_CSV = os.path.join(_HERE, "example_msvecm.csv")


def load_example() -> pd.DataFrame:
    """Simulated K=3, M=2 MSVECM daily series (T=1000, fixed seed).

    Generated from :func:`default_msvecm_params`; regenerated on first call
    if the bundled CSV is missing.
    """
    if not os.path.exists(_EXAMPLE_CSV):
        pars = default_msvecm_params()
        y, _ = simulate_msvecm(
            T=1000,
            alpha=pars["alpha"],
            beta=pars["beta"],
            Gammas=pars["Gammas"],
            B=pars["B"],
            Lambdas=pars["Lambdas"],
            P=pars["P"],
            random_state=20260101,
        )
        y.to_csv(_EXAMPLE_CSV, index_label="date")
        return y
    return pd.read_csv(_EXAMPLE_CSV, index_col="date", parse_dates=True)


def load_tether(path: str | None = None) -> pd.DataFrame:
    """Load the Tether replication dataset (daily p_usdt, q_usdt, p_btc).

    The raw series (Tether price, Tether circulating supply, Bitcoin price;
    glassnode.com, 2019-01-02 to 2023-11-08) are not redistributable with
    the package.  Supply ``path`` to a CSV with columns
    ``date, p_usdt, q_usdt, p_btc`` (raw units); the loader applies the
    paper's Appendix B rescaling: price in cents, BTC in $10,000s, supply
    in 100 millions.
    """
    if path is None:
        raise FileNotFoundError(
            "The Tether dataset is not bundled (data license). Download the "
            "daily USDT price, USDT circulating supply and BTC price series "
            "from glassnode.com (2019-01-02 to 2023-11-08), save them as a "
            "CSV with columns date, p_usdt, q_usdt, p_btc, and pass its path: "
            "load_tether('/path/to/tether.csv')."
        )
    df = pd.read_csv(path, parse_dates=["date"]).set_index("date")
    need = {"p_usdt", "q_usdt", "p_btc"}
    if not need.issubset(df.columns):
        raise ValueError(f"CSV must contain columns {sorted(need)}")
    out = pd.DataFrame(index=df.index)
    out["p_usdt"] = df["p_usdt"] * 100.0  # dollars -> cents
    out["q_usdt"] = df["q_usdt"] / 1e8  # units -> 100 millions
    out["p_btc"] = df["p_btc"] / 1e4  # dollars -> $10,000s
    return out


def _regenerate_example() -> pd.DataFrame:  # pragma: no cover - maintenance
    if os.path.exists(_EXAMPLE_CSV):
        os.remove(_EXAMPLE_CSV)
    return load_example()
