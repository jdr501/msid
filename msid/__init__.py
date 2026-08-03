"""msid -- Markov-Switching Statistical Identification for SVAR/SVECM models.

Implements the identification-through-heteroskedasticity framework of
Herwartz & Luetkepohl (2014, J. Econometrics 183) with the estimation
improvements and additional tests of Rajapaksa & Shao (2026).

Quick start
-----------
>>> import msid
>>> y = msid.load_example()
>>> model = msid.MSVECM(y, lags=1, coint_rank=1, n_regimes=2)
>>> res = model.fit(n_starts=20, random_state=0)
>>> res.summary()
>>> res.irf(horizon=30, n_boot=500).plot()
"""

from .datasets import load_example, load_tether
from .inference.lr_tests import compare_models, model_table
from .model import MSVAR, MSVECM
from .pretest import arch_lm, johansen, justify_ms, select_lags
from .restrictions import Restrictions

__version__ = "0.1.0"

__all__ = [
    "MSVAR",
    "MSVECM",
    "Restrictions",
    "__version__",
    "arch_lm",
    "compare_models",
    "johansen",
    "justify_ms",
    "load_example",
    "load_tether",
    "model_table",
    "select_lags",
]
