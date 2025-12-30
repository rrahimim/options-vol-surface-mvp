from __future__ import annotations

import numpy as np
import pandas as pd

from src.mvp.iv import implied_vol_call


def add_iv_column(
    calls: pd.DataFrame,
    price_col: str = "price",
    max_rows: int | None = None,
) -> pd.DataFrame:
    """
    Compute implied vol for each row (calls only).
    Returns a copy with an 'iv' column.
    """
    d = calls.copy()

    # optional speed cap (useful for Streamlit)
    if max_rows is not None and len(d) > max_rows:
        d = d.sample(max_rows, random_state=0).copy()

    d["iv"] = d.apply(
        lambda row: implied_vol_call(
            price=float(row[price_col]),
            S=float(row["S"]),
            K=float(row["K"]),
            r=float(row["r"]),
            q=float(row["q"]),
            T=float(row["T"]),
        ),
        axis=1,
    )

    # clean obviously bad IVs
    d = d[(d["iv"] > 0.01) & (d["iv"] < 3.0)].copy()
    return d


def build_smoothed_surface_grid(
    calls_iv: pd.DataFrame,
    expiries_top_n: int = 8,
    m_min: float = 0.75,
    m_max: float = 1.25,
    n_m: int = 40,
    n_t: int = 25,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    d = calls_iv.copy()

    # choose most liquid expiries
    top_exp = d["EXPIRE_DATE"].value_counts().head(expiries_top_n).index
    d = d[d["EXPIRE_DATE"].isin(top_exp)].copy()

    # moneyness & maturity
    d["moneyness"] = d["K"] / d["S"]
    d["tau"] = d["T"]

    # domain (tighter = smoother)
    d = d[(d["moneyness"] >= m_min) & (d["moneyness"] <= m_max)].copy()
    d = d[(d["tau"] >= 14/365) & (d["tau"] <= 1.0)].copy()  # 2w to 1y

    # tighter IV sanity filter (kills spikes)
    d = d[(d["iv"] >= 0.05) & (d["iv"] <= 1.5)].copy()

    if d.empty:
        raise ValueError("No data left after filtering. Loosen moneyness/tau/iv ranges.")

    # regular grid
    m_axis = np.linspace(m_min, m_max, n_m)
    t_axis = np.linspace(d["tau"].min(), d["tau"].max(), n_t)

    mi = np.digitize(d["moneyness"], m_axis) - 1
    ti = np.digitize(d["tau"], t_axis) - 1
    mask = (mi >= 0) & (mi < len(m_axis)) & (ti >= 0) & (ti < len(t_axis))
    d = d[mask].copy()
    mi = mi[mask]
    ti = ti[mask]

    Z = np.full((len(t_axis), len(m_axis)), np.nan)
    C = np.zeros_like(Z)

    for i, j, v in zip(ti, mi, d["iv"].values):
        if np.isnan(Z[i, j]):
            Z[i, j] = v
        else:
            Z[i, j] += v
        C[i, j] += 1

    Z = Z / np.where(C == 0, np.nan, C)

    # interpolate NaNs
    Zdf = pd.DataFrame(Z).interpolate(axis=1, limit_direction="both")
    Zdf = Zdf.interpolate(axis=0, limit_direction="both")
    Z = Zdf.values

    M, T = np.meshgrid(m_axis, t_axis)
    return M, T, Z
