"""Helpers specific to probabilistic (quantile) forecasting — Week 5.

LightGBM's `quantile` objective trains each alpha level as a fully
independent model, so nothing stops e.g. the P10 prediction from exceeding
P50 at a given row — a real, if infrequent, failure mode, not a hypothetical
one (this project's own `1d` model crosses on ~2.6% of `VALIDATION` rows).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def enforce_monotonic_quantiles(predictions: dict[float, pd.Series]) -> dict[float, pd.Series]:
    """Rearrange independently-trained quantile predictions into a monotonic stack.

    For each row, sorts the predicted values across alpha levels into
    ascending order and reassigns them to the correspondingly-ordered
    alphas — the standard "quantile rearrangement" fix (Chernozhukov et al.
    2010). This removes the logical impossibility of P10 > P50 without
    changing the *set* of predicted values at each row, just which alpha
    they're attributed to.
    """
    alphas = sorted(predictions)
    stacked = pd.concat([predictions[a] for a in alphas], axis=1)
    sorted_values = pd.DataFrame(
        np.sort(stacked.to_numpy(), axis=1), index=stacked.index, columns=alphas
    )
    return {alpha: sorted_values[alpha] for alpha in alphas}


def quantile_crossing_rate(predictions: dict[float, pd.Series]) -> float:
    """Fraction of rows where predictions aren't monotonically non-decreasing in alpha."""
    alphas = sorted(predictions)
    stacked = pd.concat([predictions[a] for a in alphas], axis=1).to_numpy()
    violations = (np.diff(stacked, axis=1) < 0).any(axis=1)
    return float(violations.mean())
