from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

import plotly.graph_objects as go

from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter

# Your project modules (works when you run: PYTHONPATH=src streamlit run app.py)
from src.mvp.data import load_options_data, most_liquid_quote_date, calls_for_date
from src.mvp.bs import bs_call_price
from src.mvp.iv import implied_vol_call
from src.mvp.bs import bs_call_price, bs_put_price, bs_greeks
from src.mvp.mc import mc_european_price

# ----------------------------
# Streamlit page config
# ----------------------------
st.set_page_config(
    page_title="Options Lab",
    layout="wide",
)

st.title("Options Lab")

# ----------------------------
# Helpers / caching
# ----------------------------
@st.cache_data(show_spinner=False)
def cached_load_df() -> pd.DataFrame:
    return load_options_data()


@st.cache_data(show_spinner=False)
def cached_quote_dates(df: pd.DataFrame, top_n: int) -> list[str]:
    # "most liquid" = most rows for that date
    counts = df["QUOTE_DATE"].value_counts()
    return counts.head(top_n).index.tolist()


@st.cache_data(show_spinner=False)
def cached_calls_for_date(df: pd.DataFrame, quote_date: str) -> pd.DataFrame:
    return calls_for_date(df, quote_date)


@st.cache_data(show_spinner=False)
def cached_calls_with_iv(
    df: pd.DataFrame,
    quote_date: str,
    use_mid: bool,
    r: float,
    q: float,
    max_rows: int,
    min_price: float,
    m_min: float,
    m_max: float,
) -> pd.DataFrame:
    """
    Returns calls for the date with columns:
    S, K, T, price, iv, moneyness (K/S)
    """
    calls = cached_calls_for_date(df, quote_date).copy()

    # Decide price used
    if use_mid and {"C_BID", "C_ASK"}.issubset(calls.columns):
        calls["price"] = 0.5 * (calls["C_BID"] + calls["C_ASK"])
    else:
        calls["price"] = calls["C_LAST"]

    # Basic filters
    calls = calls.dropna(subset=["UNDERLYING_LAST", "STRIKE", "DTE", "price"])
    calls = calls[calls["price"] > min_price]
    calls = calls[calls["DTE"] > 0]

    calls["S"] = calls["UNDERLYING_LAST"].astype(float)
    calls["K"] = calls["STRIKE"].astype(float)
    calls["T"] = calls["DTE"].astype(float) / 365.0
    calls["moneyness"] = calls["K"] / calls["S"]

    calls = calls[(calls["moneyness"] >= m_min) & (calls["moneyness"] <= m_max)]

    # Subsample for speed (keep most "reasonable" near ATM first)
    calls["atm_dist"] = np.abs(calls["moneyness"] - 1.0)
    calls = calls.sort_values(["DTE", "atm_dist"])
    if len(calls) > max_rows:
        calls = calls.head(max_rows)

    # Compute IV row-wise (robust but slower)
    def _iv(row) -> float:
        try:
            return float(
                implied_vol_call(
                    price=float(row["price"]),
                    S=float(row["S"]),
                    K=float(row["K"]),
                    r=float(r),
                    q=float(q),
                    T=float(row["T"]),
                )
            )
        except Exception:
            return np.nan

    calls["iv"] = calls.apply(_iv, axis=1)

    # Drop nonsense IVs
    calls = calls.dropna(subset=["iv"])
    calls = calls[(calls["iv"] > 0.001) & (calls["iv"] < 5.0)]

    return calls


@st.cache_data(show_spinner=False)
def cached_surface_grid(
    calls_iv: pd.DataFrame,
    top_expiries: int,
    m_min: float,
    m_max: float,
    nm: int,
    nt: int,
    interp_method: str,
    smooth_sigma: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a smoothed IV surface grid from calls_iv.
    Returns (M_grid, T_grid, IV_grid).
    """
    df = calls_iv.copy()

    # keep top expiries (by count)
    topT = (
        df.groupby("EXPIRE_DATE")["iv"]
        .count()
        .sort_values(ascending=False)
        .head(top_expiries)
        .index
    )
    df = df[df["EXPIRE_DATE"].isin(topT)]

    # Use moneyness M = K/S, and T
    M = df["moneyness"].to_numpy(float)
    T = df["T"].to_numpy(float)
    IV = df["iv"].to_numpy(float)

    # Grid
    m_grid = np.linspace(m_min, m_max, nm)
    t_grid = np.linspace(max(0.001, T.min()), T.max(), nt)
    M_grid, T_grid = np.meshgrid(m_grid, t_grid)

    # Interpolate scattered points onto grid
    points = np.column_stack([M, T])

    method = interp_method
    if method not in {"linear", "nearest", "cubic"}:
        method = "linear"

    IV_grid = griddata(points, IV, (M_grid, T_grid), method=method)

    # Fill missing areas with nearest (so surface isn't full of holes)
    if np.isnan(IV_grid).any():
        IV_nn = griddata(points, IV, (M_grid, T_grid), method="nearest")
        IV_grid = np.where(np.isnan(IV_grid), IV_nn, IV_grid)

    # Smooth
    if smooth_sigma > 0:
        IV_grid = gaussian_filter(IV_grid, sigma=float(smooth_sigma))

    return M_grid, T_grid, IV_grid


def bs_put_from_parity(call: float, S: float, K: float, r: float, q: float, T: float) -> float:
    # Put-Call parity (European, continuous yields)
    return float(call - S * np.exp(-q * T) + K * np.exp(-r * T))


def mc_european_price(
    S: float,
    K: float,
    r: float,
    q: float,
    T: float,
    sigma: float,
    option: str,
    n_paths: int,
    n_steps: int = 1,               # kept for API compatibility; we use exact terminal sim
    antithetic: bool = True,
    seed: int | None = 42,
) -> dict:
    """
    Risk-neutral GBM Monte Carlo for European call/put.
    Uses *exact terminal distribution* (no time stepping needed for European payoff).
    Returns price, stderr, CI, and terminal samples for plotting.
    """
    option = option.lower().strip()
    if option not in {"call", "put"}:
        raise ValueError("option must be 'call' or 'put'")

    if T <= 0:
        raise ValueError("T must be > 0")
    if sigma <= 0:
        raise ValueError("sigma must be > 0")
    if n_paths < 1000:
        raise ValueError("n_paths should be >= 1000 for a meaningful CI")

    rng = np.random.default_rng(seed)

    # Antithetic: generate half Z, mirror it
    if antithetic:
        n_half = (n_paths + 1) // 2
        Z = rng.standard_normal(n_half)
        Z = np.concatenate([Z, -Z])[:n_paths]
    else:
        Z = rng.standard_normal(n_paths)

    mu = (r - q - 0.5 * sigma**2) * T
    vol = sigma * np.sqrt(T)

    # Exact terminal GBM
    ST = S * np.exp(mu + vol * Z)

    if option == "call":
        payoff = np.maximum(ST - K, 0.0)
    else:
        payoff = np.maximum(K - ST, 0.0)

    disc = np.exp(-r * T)
    pv = disc * payoff

    price = float(pv.mean())
    # sample std / sqrt(n)
    stderr = float(pv.std(ddof=1) / np.sqrt(n_paths))
    ci_low = price - 1.96 * stderr
    ci_high = price + 1.96 * stderr

    return {
        "price": price,
        "stderr": stderr,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "ST": ST,
        "payoff": payoff,
    }
def _simulate_gbm_paths(S, r, q, sigma, T, n_steps, n_paths, seed=42):
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    drift = (r - q - 0.5 * sigma**2) * dt
    vol = sigma * np.sqrt(dt)

    Z = rng.standard_normal((n_paths, n_steps))
    logS = np.log(S) + np.cumsum(drift + vol * Z, axis=1)
    paths = np.exp(logS)
    paths = np.column_stack([np.full(n_paths, S), paths])  # include S0 at t=0
    tgrid = np.linspace(0.0, T, n_steps + 1)
    return tgrid, paths

def _discounted_payoff(ST, K, r, T, option="call"):
    if option == "call":
        payoff = np.maximum(ST - K, 0.0)
    else:
        payoff = np.maximum(K - ST, 0.0)
    return np.exp(-r * T) * payoff

def _mc_convergence(discounted_payoffs, batch=10_000):
    # returns arrays (n_used, mean, ci_low, ci_high)
    x = discounted_payoffs
    n = len(x)
    n_used, means, lo, hi = [], [], [], []
    for end in range(batch, n + 1, batch):
        s = x[:end]
        m = float(np.mean(s))
        se = float(np.std(s, ddof=1) / np.sqrt(end))
        ci = 1.96 * se
        n_used.append(end)
        means.append(m)
        lo.append(m - ci)
        hi.append(m + ci)
    return np.array(n_used), np.array(means), np.array(lo), np.array(hi)

def _paths_fig(tgrid, paths, title="Sample GBM Paths"):
    # plot first ~20 paths
    max_show = min(paths.shape[0], 20)
    fig = go.Figure()
    for i in range(max_show):
        fig.add_trace(go.Scatter(x=tgrid, y=paths[i], mode="lines", line=dict(width=1), showlegend=False))
    fig.update_layout(
        title=title,
        xaxis_title="Time (years)",
        yaxis_title="Underlying price",
        height=420,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig

def _convergence_fig(n_used, mean, lo, hi, title="MC Convergence (95% CI)"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=n_used, y=mean, mode="lines+markers", name="Estimate", line=dict(width=2)))
    fig.add_trace(go.Scatter(x=n_used, y=hi, mode="lines", name="95% CI High", line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(
        x=n_used, y=lo, mode="lines", name="95% CI Low",
        fill="tonexty", fillcolor="rgba(255,80,80,0.15)", line=dict(width=0), showlegend=False
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Number of paths",
        yaxis_title="Option price",
        height=380,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig
def binomial_crr_price(
    S: float,
    K: float,
    r: float,
    q: float,
    T: float,
    sigma: float,
    n_steps: int,
    option: str = "call",
    style: str = "european",
) -> float:
    """
    Cox-Ross-Rubinstein (CRR) binomial tree pricer.
    Supports European/American call/put with continuous dividend yield q.

    Returns: option price (float)
    """
    option = option.lower().strip()
    style = style.lower().strip()
    if option not in {"call", "put"}:
        raise ValueError("option must be 'call' or 'put'")
    if style not in {"european", "american"}:
        raise ValueError("style must be 'european' or 'american'")
    if n_steps < 1:
        raise ValueError("n_steps must be >= 1")
    if T <= 0:
        raise ValueError("T must be > 0")
    if sigma <= 0:
        raise ValueError("sigma must be > 0")
    if S <= 0 or K <= 0:
        raise ValueError("S and K must be > 0")

    dt = T / n_steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u

    # Risk-neutral probability with dividend yield
    disc = np.exp(-r * dt)
    a = np.exp((r - q) * dt)
    p = (a - d) / (u - d)

    # Guard against numerical weirdness for extreme params
    if not (0.0 <= p <= 1.0):
        # Clamp slightly to avoid blowing up; also signals params/steps are extreme
        p = float(np.clip(p, 0.0, 1.0))

    # Terminal stock prices: S * u^j * d^(n-j), j=0..n
    j = np.arange(n_steps + 1)
    ST = S * (u ** j) * (d ** (n_steps - j))

    if option == "call":
        values = np.maximum(ST - K, 0.0)
    else:
        values = np.maximum(K - ST, 0.0)

    # Backward induction
    for step in range(n_steps - 1, -1, -1):
        # Continuation value
        values = disc * (p * values[1:] + (1.0 - p) * values[:-1])

        if style == "american":
            # Intrinsic value at this node/time
            j = np.arange(step + 1)
            S_nodes = S * (u ** j) * (d ** (step - j))
            if option == "call":
                intrinsic = np.maximum(S_nodes - K, 0.0)
            else:
                intrinsic = np.maximum(K - S_nodes, 0.0)

            values = np.maximum(values, intrinsic)

    return float(values[0])

def crr_binomial_tree(
    S: float,
    K: float,
    r: float,
    q: float,
    T: float,
    sigma: float,
    n_steps: int,
    option: str = "call",
    exercise: str = "european",
):
    """
    Cox-Ross-Rubinstein tree.
    Returns:
      stock[i][j]   = underlying at time i with j up moves
      opt[i][j]     = option value at node
      p             = risk-neutral up probability
      dt,u,d,disc
    """
    option = option.lower().strip()
    exercise = exercise.lower().strip()

    dt = T / n_steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    disc = np.exp(-r * dt)
    p = (np.exp((r - q) * dt) - d) / (u - d)

    if not (0.0 < p < 1.0):
        # this can happen with extreme params; still handle, but warn upstream
        p = float(np.clip(p, 1e-6, 1 - 1e-6))

    # stock lattice
    stock = [[0.0] * (i + 1) for i in range(n_steps + 1)]
    for i in range(n_steps + 1):
        for j in range(i + 1):
            stock[i][j] = S * (u ** j) * (d ** (i - j))

    # payoff at maturity
    opt = [[0.0] * (i + 1) for i in range(n_steps + 1)]
    for j in range(n_steps + 1):
        if option == "call":
            opt[n_steps][j] = max(stock[n_steps][j] - K, 0.0)
        else:
            opt[n_steps][j] = max(K - stock[n_steps][j], 0.0)

    # backward induction
    for i in range(n_steps - 1, -1, -1):
        for j in range(i + 1):
            cont = disc * (p * opt[i + 1][j + 1] + (1 - p) * opt[i + 1][j])
            if exercise == "american":
                if option == "call":
                    intr = max(stock[i][j] - K, 0.0)
                else:
                    intr = max(K - stock[i][j], 0.0)
                opt[i][j] = max(cont, intr)
            else:
                opt[i][j] = cont

    return stock, opt, p, dt, u, d, disc


def binomial_tree_preview_fig(stock, opt):
    """
    Plotly preview that looks like a teaching diagram:
    black node + connecting branches + a stacked label tile.
    """
    n_steps = len(stock) - 1

    # positions (spaced like a triangle)
    xs, ys = [], []
    node_map = {}  # (i,j) -> (x,y)
    for i in range(n_steps + 1):
        for j in range(i + 1):
            x = i
            y = (2 * j - i)  # centered
            node_map[(i, j)] = (x, y)
            xs.append(x)
            ys.append(y)

    fig = go.Figure()

    # edges
    ex, ey = [], []
    for i in range(n_steps):
        for j in range(i + 1):
            x0, y0 = node_map[(i, j)]
            x_up, y_up = node_map[(i + 1, j + 1)]
            x_dn, y_dn = node_map[(i + 1, j)]
            ex += [x0, x_up, None, x0, x_dn, None]
            ey += [y0, y_up, None, y0, y_dn, None]

    fig.add_trace(
        go.Scatter(
            x=ex,
            y=ey,
            mode="lines",
            line=dict(width=3, color="rgba(255,255,255,0.18)"),
            hoverinfo="skip",
            showlegend=False,
        )
    )

    # nodes (black dots)
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="markers",
            marker=dict(size=10, color="black", line=dict(width=1, color="rgba(255,255,255,0.25)")),
            hoverinfo="skip",
            showlegend=False,
        )
    )

    # annotations (stacked “tiles” feel)
    # We’ll emulate tiles by using a single annotation with line breaks + semi-opaque bg.
    for i in range(n_steps + 1):
        for j in range(i + 1):
            x, y = node_map[(i, j)]
            Sij = stock[i][j]
            Vij = opt[i][j]

            label = (
                f"<b>{Sij:,.4f}</b><br>"
                f"<span style='color:rgba(255,110,110,1)'><b>{Vij:,.4f}</b></span>"
            )

            fig.add_annotation(
                x=x,
                y=y,
                text=label,
                showarrow=False,
                xanchor="center",
                yanchor="middle",
                align="center",
                font=dict(size=12, color="rgba(230,230,230,1)"),
                bgcolor="rgba(30,35,45,0.75)",
                bordercolor="rgba(255,255,255,0.12)",
                borderwidth=1,
                borderpad=6,
                opacity=1.0,
                yshift=18,  # tile sits above the dot, like your example
            )

    fig.update_layout(
        height=520,
        margin=dict(l=10, r=10, t=20, b=10),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )
    return fig

# ----------------------------
# Load dataset + GLOBAL settings (not in sidebar)
# ----------------------------
with st.spinner("Loading dataset…"):
    df = cached_load_df()

# Default global settings (safe + consistent)
top_dates_default = 50
dates = cached_quote_dates(df, top_dates_default)

default_date = most_liquid_quote_date(df)
quote_date = default_date if default_date in dates else dates[0]

use_mid = True
r = 0.01
q = 0.00
max_rows = 12000
min_price = 0.05

# ----------------------------
# Sidebar: Clean README + LinkedIn
# ----------------------------
with st.sidebar:
    # Header (no "About")
    st.caption("Built by")

    st.markdown(
        """
        <div style="display:flex; align-items:center; gap:10px;">
            <img src="https://cdn-icons-png.flaticon.com/512/174/174857.png"
                 width="22" height="22">
            <a href="https://www.linkedin.com/in/rrahimim/"
               target="_blank"
               style="
                   text-decoration:none;
                   color:white;
                   background:#1f2937;
                   padding:6px 12px;
                   border-radius:6px;
                   font-weight:600;
                   font-size:0.9rem;
               ">
                Reza Rahimi-Motem
            </a>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    st.markdown("### READ ME")
    st.markdown(
        """
This app explores **options pricing**, **implied volatility**, and **risk/scenario analysis** using SPY options data.

**Tabs**
- **Option Pricer** — Black–Scholes pricing, Greeks, heatmaps  
- **Volatility Smile** — Compute IVs + smoothing by expiry  
- **Volatility Surface** — Smoothed 3D implied volatility surface  
- **Monte Carlo** — GBM pricing + distributions + diagnostics  
- **Scenario & Risk** — Multi-leg strategies + stress testing  
- **Binomial Tree** — CRR European/American pricing + educational tree view  

**Tip**
Compute IVs in **Volatility Smile** first for the fastest **Surface** workflow.
"""
    )

# ----------------------------
# Tabs
# ----------------------------
tabs = st.tabs(["Option Pricer", "Volatility Smile", "Volatility Surface", "Monte Carlo", "Scenario & Risk", "Binomial Tree"])


# ----------------------------
# Tab 1: Pricer + Heatmap
# ----------------------------
with tabs[0]:
    st.subheader("Black–Scholes Option Pricer")

    # ---- defaults (so Streamlit has something defined on first run)
    S0 = 100.0
    K0 = 100.0
    T0 = 1.0
    sig0 = 0.20
    rp = 0.01
    qp = 0.00

    c1, c2, c3 = st.columns(3)
    with c1:
        S = st.number_input("Spot Price (S)", value=S0, step=1.0, format="%.2f")
        K = st.number_input("Strike Price (K)", value=K0, step=1.0, format="%.2f")

    with c2:
        T = st.number_input("Maturity (Years, T)", value=T0, step=0.01, format="%.4f")
        sigma = st.number_input("Volatility (σ)", value=sig0, step=0.01, format="%.4f")

    with c3:
        r = st.number_input("Risk-Free Rate (r)", value=rp, step=0.001, format="%.4f")
        q = st.number_input("Dividend Yield (q)", value=qp, step=0.001, format="%.4f")

    # price from the SAME inputs you just collected
    call = bs_call_price(S=S, K=K, r=r, q=q, T=T, sigma=sigma)
    put = bs_put_price(S=S, K=K, r=r, q=q, T=T, sigma=sigma)

    greeks = bs_greeks(S=S, K=K, r=r, q=q, T=T, sigma=sigma)

    st.markdown(
        """
        <style>
        div.block-container { padding-top: 2.5rem; }

        /* ---- Prices: centered row + one subtle red highlight ---- */
        .price-row {
            display: flex;
            justify-content: space-between;   /* centers the two cards */
            gap: 20px;
            margin: 16px 0 12px 0;
        }
        .price-box {
            flex: 1;           /* fixed-ish width so they don't stretch edge-to-edge */
            padding: 22px 20px;
            border-radius: 14px;
            background: rgba(255,255,255,0.04);
            border: 1px solid rgba(255,255,255,0.10);
            text-align: center;        /* centers label + value */
        }
        
        /* subtle red accent to match theme */
        .price-box.call {
            background: linear-gradient(
                135deg,
                rgba(239,68,68,0.18),
                rgba(239,68,68,0.08)
            );
            border: 1px solid rgba(239,68,68,0.35);
        }
        .price-label {
            font-size: 0.9rem;
            opacity: 0.75;
            margin-bottom: 6px;
        }
        .price-value {
            font-size: 2.55rem;
            font-weight: 900;
            letter-spacing: -0.6px;
            line-height: 1.05;
        }

        /* ---- Greeks ---- */
        .greek-grid {
            display: grid;
            grid-template-columns: repeat(5, 1fr);
            gap: 12px;
            margin-bottom: 14px;
        }
        .greek-cell {
            padding: 12px 14px;
            border-radius: 12px;
            background: rgba(255,255,255,0.03);
            border: 1px solid rgba(255,255,255,0.08);
            min-height: 72px;
        }
        .greek-label {
            font-size: 0.8rem;
            opacity: 0.7;
            margin-bottom: 4px;
        }
        .greek-value {
            font-size: 1.35rem;
            font-weight: 750;
        }

        .small-note {
            opacity: 0.7;
            font-size: 0.85rem;
            margin-top: 6px;
            text-align: center;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # ---- Call / Put prices (centered; Call highlighted in subtle red)
    st.markdown(
        f"""
        <div class="price-row">
          <div class="price-box call">
            <div class="price-label">Call Price</div>
            <div class="price-value">${call:,.4f}</div>
          </div>
          <div class="price-box">
            <div class="price-label">Put Price</div>
            <div class="price-value">${put:,.4f}</div>
          </div>
        </div>
        <div class="small-note">
          Put is computed via put–call parity for the same inputs (European).
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    # ---- Option Greeks header (same level as pricer header)
    st.subheader("Option Greeks")

    # ---- Greeks: Call
    st.markdown("**Call**")
    st.markdown(
        f"""
        <div class="greek-grid">
          <div class="greek-cell"><div class="greek-label">Delta (Δ)</div><div class="greek-value">{greeks['delta_call']:.4f}</div></div>
          <div class="greek-cell"><div class="greek-label">Gamma (Γ)</div><div class="greek-value">{greeks['gamma']:.6f}</div></div>
          <div class="greek-cell"><div class="greek-label">Vega (ν)</div><div class="greek-value">{greeks['vega']:.4f}</div></div>
          <div class="greek-cell"><div class="greek-label">Theta (Θ)</div><div class="greek-value">{greeks['theta_call']:.4f}</div></div>
          <div class="greek-cell"><div class="greek-label">Rho (ρ)</div><div class="greek-value">{greeks['rho_call']:.4f}</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ---- Greeks: Put
    st.markdown("**Put**")
    st.markdown(
        f"""
        <div class="greek-grid">
          <div class="greek-cell"><div class="greek-label">Delta (Δ)</div><div class="greek-value">{greeks['delta_put']:.4f}</div></div>
          <div class="greek-cell"><div class="greek-label">Gamma (Γ)</div><div class="greek-value">{greeks['gamma']:.6f}</div></div>
          <div class="greek-cell"><div class="greek-label">Vega (ν)</div><div class="greek-value">{greeks['vega']:.4f}</div></div>
          <div class="greek-cell"><div class="greek-label">Theta (Θ)</div><div class="greek-value">{greeks['theta_put']:.4f}</div></div>
          <div class="greek-cell"><div class="greek-label">Rho (ρ)</div><div class="greek-value">{greeks['rho_put']:.4f}</div></div>
        </div>
        <div class="small-note">
          Theta is per year. Vega is per 1.00 change in volatility.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()
    st.subheader("Options Price — Interactive Heatmap")

    hc1, hc2 = st.columns(2)
    with hc1:
        spot_min = st.number_input("Min Spot Price", value=80.0, step=5.0)
        spot_max = st.number_input("Max Spot Price", value=120.0, step=5.0)
    with hc2:
        vol_min = st.number_input("Min Volatility", value=0.10, step=0.01, format="%.2f")
        vol_max = st.number_input("Max Volatility", value=0.40, step=0.01, format="%.2f")

    grid_n = st.slider("Grid Resolution", 10, 30, 20)

    spots = np.linspace(spot_min, spot_max, grid_n)
    vols = np.linspace(vol_min, vol_max, grid_n)

    call_grid = np.zeros((len(vols), len(spots)))
    put_grid = np.zeros((len(vols), len(spots)))

    for i, vol in enumerate(vols):
        for j, s_ in enumerate(spots):
            call_grid[i, j] = bs_call_price(S=s_, K=K, r=r, q=q, T=T, sigma=vol)
            put_grid[i, j] = bs_put_price(S=s_, K=K, r=r, q=q, T=T, sigma=vol)

    def plot_heatmap(z, title, colorscale):
        fig = go.Figure(
            data=go.Heatmap(
                z=z,
                x=spots,
                y=vols,
                colorscale=colorscale,
                colorbar=dict(title="Price"),
                zsmooth=False,
                text=np.round(z, 2),
                texttemplate="%{text}",
                textfont=dict(size=10),
                hovertemplate="Spot=%{x:.2f}<br>Vol=%{y:.2f}<br>Price=%{z:.4f}<extra></extra>",
            )
        )
        fig.update_layout(
            title=title,
            xaxis_title="Spot Price (S)",
            yaxis_title="Volatility (σ)",
            height=560,
            margin=dict(l=20, r=20, t=50, b=20),
        )
        return fig

    # Call AND Put should both be: low=red, mid=yellow, high=green
    # (so put no longer uses the reversed scale)
    CALL_COLORSCALE = "RdYlGn"
    PUT_COLORSCALE = "RdYlGn"

    h1, h2 = st.columns(2)
    with h1:
        st.plotly_chart(plot_heatmap(call_grid, "Call Price Heatmap", CALL_COLORSCALE), use_container_width=True)
    with h2:
        st.plotly_chart(plot_heatmap(put_grid, "Put Price Heatmap", PUT_COLORSCALE), use_container_width=True)

# ----------------------------
# Tab 2: Smile (Compute button + smoothing)  ✅ NO st.stop()
# ----------------------------
with tabs[1]:
    st.subheader("Implied Volatility Smile")

    st.markdown(
        """
        <style>
          /* overall page top padding (also helps "cut off" feeling) */
          div.block-container { padding-top: 2.2rem; }

          /* tighten spacing between widgets in Smile tab */
          [data-testid="stVerticalBlock"] { gap: 0.6rem; }

          /* selectbox spacing */
          [data-testid="stSelectbox"] { margin-top: 0.25rem; margin-bottom: 0.75rem; }

          /* plotly chart spacing */
          [data-testid="stPlotlyChart"] { margin-top: 0.25rem; }

          /* section headings spacing */
          .sm-section-title { margin: 0.6rem 0 0.25rem 0; font-weight: 700; font-size: 1.05rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # --- Controls
    c0, c1, c2 = st.columns([1.2, 1.2, 2.6])

    with c0:
        m_min_sm = st.number_input(
            "Min Moneyness (K/S)", value=0.60, step=0.05, format="%.2f", key="sm_mmin"
        )
        m_max_sm = st.number_input(
            "Max Moneyness (K/S)", value=1.80, step=0.05, format="%.2f", key="sm_mmax"
        )

    with c1:
        smooth_window = st.slider("Smoothing window", 5, 61, 21, step=2, key="sm_smooth_w")
        if smooth_window % 2 == 0:
            smooth_window += 1

    with c2:
        auto_compute = st.checkbox("Auto compute on change", value=False, key="sm_auto")
        show_table = st.checkbox("Show raw rows (preview)", value=False, key="sm_show_table")

    # --- Compute trigger logic
    sig = (
        str(quote_date),
        bool(use_mid),
        float(r),
        float(q),
        int(max_rows),
        float(min_price),
        float(m_min_sm),
        float(m_max_sm),
    )

    compute_clicked = st.button("Compute IVs for Smile", type="primary", key="btn_smile_compute")
    st.markdown("<div style='height: 0.75rem;'></div>", unsafe_allow_html=True)

    should_compute = compute_clicked or (auto_compute and st.session_state.get("smile_sig") != sig)

    if should_compute:
        st.session_state["smile_sig"] = sig
        t0 = time.time()
        with st.spinner("Computing IVs (cached)…"):
            calls_iv_new = cached_calls_with_iv(
                df=df,
                quote_date=quote_date,
                use_mid=use_mid,
                r=float(r),
                q=float(q),
                max_rows=int(max_rows),
                min_price=float(min_price),
                m_min=float(m_min_sm),
                m_max=float(m_max_sm),
            )
        st.session_state["calls_iv"] = calls_iv_new
        st.session_state["calls_iv_time"] = time.time() - t0

    calls_iv = st.session_state.get("calls_iv")

    # --- If nothing computed yet, show message (NO stop)
    if calls_iv is None:
        st.info("Set your filters, then click **Compute IVs for Smile** (or enable Auto compute).")
    elif len(calls_iv) == 0:
        st.warning("No rows matched your filters. Try widening moneyness or lowering min option price.")
    else:
        # --- Summary
        n = len(calls_iv)
        tsec = st.session_state.get("calls_iv_time", 0.0)

        csum1, csum2, csum3, csum4 = st.columns(4)
        csum1.metric("Rows with IV", f"{n:,}")
        csum2.metric("Median IV", f"{calls_iv['iv'].median():.4f}")
        csum3.metric("IV range", f"{calls_iv['iv'].min():.3f} → {calls_iv['iv'].max():.3f}")
        csum4.metric("Compute time", f"{tsec:.2f}s")

        # --- Optional table preview
        if show_table:
            n_show = st.slider("Rows to preview", 50, 2000, 300, step=50, key="sm_preview_n")
            st.dataframe(
                calls_iv[["QUOTE_DATE", "EXPIRE_DATE", "S", "K", "T", "price", "iv", "moneyness"]]
                .sort_values(["EXPIRE_DATE", "K"])
                .head(int(n_show)),
                use_container_width=True,
                height=320,
            )

        # ----------------------------
        # Expiry selector
        # ----------------------------
        exp_grp = (
            calls_iv.groupby("EXPIRE_DATE")
            .agg(n=("iv", "size"), medT=("T", "median"))
            .sort_values("n", ascending=False)
            .reset_index()
        )

        exp_options = exp_grp["EXPIRE_DATE"].tolist()
        exp_label_map = {
            row["EXPIRE_DATE"]: f"{row['EXPIRE_DATE']} — {int(row['n'])} contracts, ~{row['medT']:.2f}y"
            for _, row in exp_grp.iterrows()
        }

        st.markdown('<div class="sm-section-title">Select Expiry</div>', unsafe_allow_html=True)

        expiry = st.selectbox(
            label="Select Expiry",
            options=exp_options,
            index=0,
            format_func=lambda x: exp_label_map.get(x, str(x)),
            label_visibility="collapsed",
            key="sm_expiry",
        )

        df_exp = calls_iv[calls_iv["EXPIRE_DATE"] == expiry].copy().sort_values("moneyness")

        # Guard: if expiry slice is empty, just warn and skip plotting (NO stop)
        if len(df_exp) == 0:
            st.warning("No rows for that expiry (unexpected). Try selecting another expiry.")
        else:
            # ----------------------------
            # Smile summary metrics (ATM / RR / BF)
            # ----------------------------
            atm_idx = (df_exp["moneyness"] - 1.0).abs().idxmin()
            atm_iv = float(df_exp.loc[atm_idx, "iv"])

            put_wing = df_exp[(df_exp["moneyness"] >= 0.85) & (df_exp["moneyness"] <= 0.95)]
            call_wing = df_exp[(df_exp["moneyness"] >= 1.05) & (df_exp["moneyness"] <= 1.15)]

            put_iv = float(put_wing["iv"].median()) if len(put_wing) else np.nan
            call_iv = float(call_wing["iv"].median()) if len(call_wing) else np.nan

            rr_25 = (call_iv - put_iv) if np.isfinite(call_iv) and np.isfinite(put_iv) else np.nan
            bf_25 = (0.5 * (call_iv + put_iv) - atm_iv) if np.isfinite(call_iv) and np.isfinite(put_iv) else np.nan

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("ATM IV", f"{atm_iv:.4f}")
            m2.metric("25Δ Risk Reversal", f"{rr_25:.4f}" if np.isfinite(rr_25) else "—")
            m3.metric("25Δ Butterfly", f"{bf_25:.4f}" if np.isfinite(bf_25) else "—")
            m4.metric("Wing Definition", "Put: 0.85–0.95 | Call: 1.05–1.15")

            # ----------------------------
            # Smoothing
            # ----------------------------
            w = min(smooth_window, max(5, (len(df_exp) // 2) * 2 + 1))
            if w % 2 == 0:
                w += 1

            if len(df_exp) >= 8:
                df_exp["iv_smooth"] = (
                    df_exp["iv"]
                    .rolling(window=w, center=True, min_periods=max(3, w // 3))
                    .median()
                    .rolling(window=max(3, w // 3), center=True, min_periods=3)
                    .mean()
                )
            else:
                df_exp["iv_smooth"] = np.nan

            # ----------------------------
            # Plot
            # ----------------------------
            fig = go.Figure()

            fig.add_trace(
                go.Scatter(
                    x=df_exp["moneyness"],
                    y=df_exp["iv"],
                    mode="markers",
                    name="IV (raw)",
                    marker=dict(size=6, opacity=0.85),
                    hovertemplate="K/S=%{x:.4f}<br>IV=%{y:.4f}<extra></extra>",
                )
            )

            if df_exp["iv_smooth"].notna().sum() >= 5:
                fig.add_trace(
                    go.Scatter(
                        x=df_exp["moneyness"],
                        y=df_exp["iv_smooth"],
                        mode="lines",
                        name="IV (smoothed)",
                        line=dict(width=3),
                        hovertemplate="K/S=%{x:.4f}<br>IV(smooth)=%{y:.4f}<extra></extra>",
                    )
                )

            fig.add_vline(x=1.0, line_dash="dash", opacity=0.6)

            y_top = float(np.nanmax(df_exp["iv"].values))
            fig.add_annotation(
                x=1.0,
                y=y_top,
                text="ATM (K/S=1.0)",
                showarrow=False,
                yshift=12,
            )

            fig.update_layout(
                title=dict(
                    text=f"SPY IV Smile — {quote_date}, Expiry {expiry}",
                    x=0.0,
                    xanchor="left",
                    y=0.98,
                    yanchor="top",
                    font=dict(size=18),
                ),
                xaxis_title="Moneyness (K/S)",
                yaxis_title="Implied Volatility (σ)",
                legend=dict(
                    orientation="v",
                    x=1.02,
                    xanchor="left",
                    y=1.0,
                    yanchor="top",
                    bgcolor="rgba(0,0,0,0)",
                    font=dict(size=12),
                ),
                margin=dict(l=30, r=30, t=70, b=40),
                height=520,
            )

            st.plotly_chart(fig, use_container_width=True)

            # ----------------------------
            # Term structure (ATM / 25Δ RR / 25Δ BF by expiry)
            # ----------------------------
            st.markdown('<div class="sm-section-title">Term Structure</div>', unsafe_allow_html=True)

            # Build per-expiry metrics table
            rows = []
            for exp, g in calls_iv.groupby("EXPIRE_DATE"):
                g = g.sort_values("moneyness")
                T_med = float(g["T"].median())

                # ATM IV = closest to moneyness 1.0
                atm_idx = (g["moneyness"] - 1.0).abs().idxmin()
                atm_iv = float(g.loc[atm_idx, "iv"])

                # "25Δ-ish" wings via moneyness bands (simple + robust)
                put_wing = g[(g["moneyness"] >= 0.85) & (g["moneyness"] <= 0.95)]
                call_wing = g[(g["moneyness"] >= 1.05) & (g["moneyness"] <= 1.15)]

                put_iv = float(put_wing["iv"].median()) if len(put_wing) else np.nan
                call_iv = float(call_wing["iv"].median()) if len(call_wing) else np.nan

                rr_25 = (call_iv - put_iv) if np.isfinite(call_iv) and np.isfinite(put_iv) else np.nan
                bf_25 = (0.5 * (call_iv + put_iv) - atm_iv) if np.isfinite(call_iv) and np.isfinite(put_iv) else np.nan

                rows.append(
                    {
                        "EXPIRE_DATE": exp,
                        "T": T_med,
                        "n": int(len(g)),
                        "ATM": atm_iv,
                        "RR25": rr_25,
                        "BF25": bf_25,
                    }
                )

            ts = pd.DataFrame(rows).sort_values("T")

            # Optional: show table preview
            with st.expander("Show term structure table"):
                st.dataframe(ts, use_container_width=True, height=260)

            # Plot
            fig_ts = go.Figure()

            fig_ts.add_trace(go.Scatter(
                x=ts["T"], y=ts["ATM"],
                mode="lines+markers",
                name="ATM IV",
                hovertemplate="T=%{x:.3f}y<br>ATM=%{y:.4f}<extra></extra>",
            ))

            fig_ts.add_trace(go.Scatter(
                x=ts["T"], y=ts["RR25"],
                mode="lines+markers",
                name="25Δ Risk Reversal",
                hovertemplate="T=%{x:.3f}y<br>RR25=%{y:.4f}<extra></extra>",
            ))

            fig_ts.add_trace(go.Scatter(
                x=ts["T"], y=ts["BF25"],
                mode="lines+markers",
                name="25Δ Butterfly",
                hovertemplate="T=%{x:.3f}y<br>BF25=%{y:.4f}<extra></extra>",
            ))

            fig_ts.update_layout(
                title=dict(
                    text=f"SPY Vol Term Structure — {quote_date}",
                    x=0.0, xanchor="left",
                    font=dict(size=18),
                ),
                xaxis_title="Time to Maturity (years)",
                yaxis_title="Vol / Skew Metrics",
                margin=dict(l=30, r=30, t=70, b=40),
                height=420,
                legend=dict(x=1.02, xanchor="left", y=1.0, yanchor="top"),
            )

            st.plotly_chart(fig_ts, use_container_width=True)

# ----------------------------
# Tab 3: Surface (Compute button + smoothing controls)  ✅ NO st.stop()
# ----------------------------
with tabs[2]:
    st.subheader("Smoothed Implied Volatility Surface")
    st.caption("Build a smooth IV surface from the computed IV table (calls).")

    # Try to reuse IVs from Smile tab
    calls_iv = st.session_state.get("calls_iv")

    # --- If missing, allow computing here (so Surface tab isn't dependent on Smile tab)
    if calls_iv is None:
        st.warning("No IV table found. Compute IVs here (or go to Smile tab and compute first).")

        cA, cB = st.columns(2)
        with cA:
            m_min_local = st.number_input(
                "IV moneyness min (K/S)",
                value=0.60,
                step=0.05,
                format="%.2f",
                key="mmin_sf_local",
            )
        with cB:
            m_max_local = st.number_input(
                "IV moneyness max (K/S)",
                value=1.80,
                step=0.05,
                format="%.2f",
                key="mmax_sf_local",
            )

        if st.button("Compute IVs for Surface", type="primary", key="btn_compute_ivs_surface"):
            t0 = time.time()
            with st.spinner("Computing IVs (cached)…"):
                calls_iv = cached_calls_with_iv(
                    df=df,
                    quote_date=quote_date,
                    use_mid=use_mid,
                    r=float(r),
                    q=float(q),
                    max_rows=int(max_rows),
                    min_price=float(min_price),
                    m_min=float(m_min_local),
                    m_max=float(m_max_local),
                )
            st.session_state["calls_iv"] = calls_iv
            st.session_state["calls_iv_time"] = time.time() - t0

        calls_iv = st.session_state.get("calls_iv")

    # --- Controls styling (safe to render even if calls_iv missing)
    st.markdown(
        """
        <style>
          .surface-section { margin-top: 1.2rem; margin-bottom: 0.6rem; }
          .surface-subtitle { font-weight: 700; font-size: 1.05rem; margin-bottom: 0.4rem; }
          [data-testid="stSlider"] { padding-top: 0.4rem; }
          [data-testid="stSelectbox"] { margin-top: 0.25rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # If still missing, show hint and DO NOT stop the app
    if calls_iv is None:
        st.info("Click **Compute IVs for Surface** to continue.")
    elif len(calls_iv) == 0:
        st.warning("IV table is empty. Try widening moneyness or lowering min option price.")
    else:
        # ---- Section: Expiries
        st.markdown('<div class="surface-section surface-subtitle">Expiry Selection</div>', unsafe_allow_html=True)
        top_exp = st.slider("Top Liquid Expiries", 3, 20, 8, key="sf_top_exp")

        # ---- Section: Moneyness range
        st.markdown('<div class="surface-section surface-subtitle">Moneyness Range (K / S)</div>', unsafe_allow_html=True)
        mcol1, mcol2 = st.columns(2)
        with mcol1:
            m_min_sf = st.number_input(
                "Minimum Moneyness",
                value=0.75,
                step=0.05,
                format="%.2f",
                key="mmin_sf",
            )
        with mcol2:
            m_max_sf = st.number_input(
                "Maximum Moneyness",
                value=1.25,
                step=0.05,
                format="%.2f",
                key="mmax_sf",
            )

        # ---- Section: Grid resolution
        st.markdown('<div class="surface-section surface-subtitle">Surface Grid Resolution</div>', unsafe_allow_html=True)
        gcol1, gcol2 = st.columns(2)
        with gcol1:
            nm = st.slider("Moneyness Points", 20, 80, 45, key="sf_nm")
        with gcol2:
            nt = st.slider("Maturity Points", 20, 80, 45, key="sf_nt")

        # ---- Section: Smoothing & interpolation
        st.markdown('<div class="surface-section surface-subtitle">Smoothing & Interpolation</div>', unsafe_allow_html=True)
        icol1, icol2 = st.columns(2)
        with icol1:
            interp = st.selectbox(
                "Interpolation Method",
                options=["Linear", "Nearest", "Cubic"],
                index=0,
                key="sf_interp",
            )
        with icol2:
            smooth_sigma = st.slider(
                "Smoothing Strength (σ)",
                0.0,
                3.0,
                1.0,
                step=0.25,
                key="sf_sigma",
            )

        # ---- Optional overlay
        show_scatter = st.toggle("Overlay Raw IV Points", value=False, key="sf_scatter")

        # ---- Compute
        st.markdown("<div style='height: 0.75rem;'></div>", unsafe_allow_html=True)
        compute_surface = st.button("Compute Surface", type="primary", key="btn_compute_surface")

        if compute_surface:
            t0 = time.time()
            with st.spinner("Building smoothed surface grid (cached)…"):
                M_grid, T_grid, IV_grid = cached_surface_grid(
                    calls_iv=calls_iv,
                    top_expiries=int(top_exp),
                    m_min=float(m_min_sf),
                    m_max=float(m_max_sf),
                    nm=int(nm),
                    nt=int(nt),
                    interp_method=str(interp).lower(),
                    smooth_sigma=float(smooth_sigma),
                )
            st.session_state["surface_grid"] = (M_grid, T_grid, IV_grid)
            st.session_state["surface_time"] = time.time() - t0

        grid = st.session_state.get("surface_grid")
        if grid is None:
            st.info("Click **Compute Surface** to generate the smoothed IV surface.")
        else:
            M_grid, T_grid, IV_grid = grid
            st.caption(f"Surface computed in {st.session_state.get('surface_time', 0):.2f}s")

            # Safety check
            if not np.isfinite(IV_grid).any():
                st.error("Surface grid produced no finite IV values. Try Linear interpolation and/or lower σ.")
            else:
                fig = go.Figure()
                fig.add_trace(
                    go.Surface(
                        x=M_grid,
                        y=T_grid,
                        z=IV_grid,
                        colorbar=dict(title="IV (σ)"),
                    )
                )

                if show_scatter:
                    df_sc = calls_iv.copy()
                    df_sc = df_sc[(df_sc["moneyness"] >= m_min_sf) & (df_sc["moneyness"] <= m_max_sf)]

                    topT = (
                        df_sc.groupby("EXPIRE_DATE")["iv"]
                        .count()
                        .sort_values(ascending=False)
                        .head(int(top_exp))
                        .index
                    )
                    df_sc = df_sc[df_sc["EXPIRE_DATE"].isin(topT)]

                    fig.add_trace(
                        go.Scatter3d(
                            x=df_sc["moneyness"],
                            y=df_sc["T"],
                            z=df_sc["iv"],
                            mode="markers",
                            marker=dict(size=2),
                            name="Raw IV",
                        )
                    )

                fig.update_layout(
                    title=f"Implied Volatility Surface (Smoothed) — {quote_date}",
                    scene=dict(
                        xaxis_title="Moneyness (K/S)",
                        yaxis_title="Time to Maturity (Years)",
                        zaxis_title="Implied Volatility (σ)",
                    ),
                    height=720,
                    margin=dict(l=0, r=0, t=60, b=0),
                )

                st.plotly_chart(fig, use_container_width=True)

with tabs[3]:
    st.subheader("Monte Carlo Pricer (GBM)")
    st.caption("Simulates risk-neutral GBM drift (r − q) and prices a European call/put.")

    # --- Pull safe defaults (won't crash if missing)
    S_def = float(st.session_state.get("tab1_S", 100.0))
    K_def = float(st.session_state.get("tab1_K", 100.0))
    T_def = float(st.session_state.get("tab1_T", 1.0))
    sig_def = float(st.session_state.get("tab1_sigma", 0.20))
    r_def = float(st.session_state.get("sidebar_r", 0.01))
    q_def = float(st.session_state.get("sidebar_q", 0.00))

    # --- Styles for cleaner spacing
    st.markdown(
        """
        <style>
          /* tighten spacing inside this tab */
          [data-testid="stVerticalBlock"] { gap: 0.55rem; }
          [data-testid="stSlider"] { padding-top: 0.15rem; }
          [data-testid="stNumberInput"] { margin-top: 0.15rem; }
          [data-testid="stSelectbox"] { margin-top: 0.10rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # ----------------------------
    # Inputs
    # ----------------------------
    st.markdown("### Inputs")

    cA, cB, cC = st.columns(3)
    with cA:
        opt_type = st.selectbox("Option Type", ["Call", "Put"], index=0, key="mc_opt_type")
        S_mc = st.number_input("Spot Price (S)", value=S_def, step=1.0, format="%.2f", key="mc_S")
        K_mc = st.number_input("Strike (K)", value=K_def, step=1.0, format="%.2f", key="mc_K")

    with cB:
        T_mc = st.number_input("Maturity (T, years)", value=T_def, step=0.01, format="%.4f", key="mc_T")
        sigma_mc = st.number_input("Volatility (σ)", value=sig_def, step=0.01, format="%.4f", key="mc_sigma")

    with cC:
        r_mc = st.number_input("Risk-Free Rate (r)", value=r_def, step=0.001, format="%.4f", key="mc_r")
        q_mc = st.number_input("Dividend Yield (q)", value=q_def, step=0.001, format="%.4f", key="mc_q")

    st.markdown("<div style='height: 0.35rem;'></div>", unsafe_allow_html=True)

    # ----------------------------
    # Simulation settings (clean row)
    # ----------------------------
    st.markdown("### Simulation Settings")

    s1, s2, s3 = st.columns([1.25, 1.25, 1.0])
    with s1:
        n_paths = st.slider("Paths", 10_000, 500_000, 200_000, step=10_000, key="mc_paths")
    with s2:
        n_steps = st.slider("Steps", 1, 365, 252, step=1, key="mc_steps")
    with s3:
        seed = st.number_input("Random seed", value=42, step=1, key="mc_seed")

    st.markdown("<div style='height: 0.25rem;'></div>", unsafe_allow_html=True)

    run = st.button("Run Monte Carlo", type="primary", key="run_mc_btn")

    if not run:
        st.info("Set parameters and click **Run Monte Carlo**.")
    else:
        # ----------------------------
        # Run simulation (antithetic always ON)
        # ----------------------------
        with st.spinner("Simulating…"):
            res = mc_european_price(
                S=float(S_mc),
                K=float(K_mc),
                r=float(r_mc),
                q=float(q_mc),
                T=float(T_mc),
                sigma=float(sigma_mc),
                option=opt_type.lower(),
                n_paths=int(n_paths),
                n_steps=int(n_steps),
                antithetic=True,  # ✅ always on
                seed=int(seed),
            )

        # Black–Scholes benchmark
        if opt_type.lower() == "call":
            bs = bs_call_price(
                S=float(S_mc),
                K=float(K_mc),
                r=float(r_mc),
                q=float(q_mc),
                T=float(T_mc),
                sigma=float(sigma_mc),
            )
        else:
            bs = bs_put_price(
                S=float(S_mc),
                K=float(K_mc),
                r=float(r_mc),
                q=float(q_mc),
                T=float(T_mc),
                sigma=float(sigma_mc),
            )

        # ----------------------------
        # Metrics
        # ----------------------------
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Monte Carlo Price", f"${res['price']:.4f}")
        m2.metric("Std. Error", f"{res['stderr']:.6f}")
        m3.metric("95% CI", f"[{res['ci_low']:.4f}, {res['ci_high']:.4f}]")
        m4.metric("Black–Scholes", f"${bs:.4f}")

        st.caption("Tip: increasing Paths tightens the CI (roughly ∝ 1/√N).")

        # ----------------------------
        # Charts (nicer + themed)
        # ----------------------------
        THEME_RED = "rgba(255, 77, 77, 0.85)"
        GRIDLINE = "rgba(255,255,255,0.08)"

        def _hist_fig(x, title, xlab, nbins=45):
            fig = go.Figure(
                data=go.Histogram(
                    x=x,
                    nbinsx=nbins,
                    marker=dict(color=THEME_RED),
                    hovertemplate=f"{xlab}: %{{x:.4f}}<br>Count: %{{y}}<extra></extra>",
                )
            )
            fig.update_layout(
                title=title,
                height=420,
                margin=dict(l=20, r=20, t=45, b=20),
            )
            fig.update_xaxes(title=xlab, showgrid=True, gridcolor=GRIDLINE, zeroline=False)
            fig.update_yaxes(title="Number of Paths", showgrid=True, gridcolor=GRIDLINE, zeroline=False)
            return fig

        ST = res.get("ST", None)
        payoff = res.get("payoff", None)

        if ST is None or payoff is None:
            st.warning("Charts require the Monte Carlo function to return arrays: res['ST'] and res['payoff'].")
        else:
            ch1, ch2 = st.columns(2)
            with ch1:
                st.plotly_chart(
                    _hist_fig(
                        ST,
                        "Terminal Price Distribution",
                        r"Terminal Price",
                    ),
                    use_container_width=True,
                )
            with ch2:
                st.plotly_chart(
                    _hist_fig(payoff, "Payoff Distribution", "Option Payoff at Maturity"),
                    use_container_width=True,
                )

        # ----------------------------
        # Diagnostics (optional; render only if helpers exist)
        # ----------------------------
        st.divider()
        st.subheader("Simulation Diagnostics")

        # Guard if helper functions aren't defined/imported
        missing = []
        for fn_name in ["_simulate_gbm_paths", "_discounted_payoff", "_mc_convergence", "_paths_fig", "_convergence_fig"]:
            if fn_name not in globals():
                missing.append(fn_name)

        if missing:
            st.info(
                "Diagnostics are optional. Missing helper(s): "
                + ", ".join(missing)
                + ". If you want, I can give you the exact helper code to paste in."
            )
        else:
            # --- Extra visuals: small path set (fast)
            tgrid, paths = _simulate_gbm_paths(
                S=float(S_mc),
                r=float(r_mc),
                q=float(q_mc),
                sigma=float(sigma_mc),
                T=float(T_mc),
                n_steps=int(n_steps),
                n_paths=2000,  # small just for paths
                seed=int(seed),
            )

            # --- Convergence: payoff-only run
            rng = np.random.default_rng(int(seed))
            n_conv = int(n_paths)
            dt = float(T_mc) / int(n_steps)
            drift = (float(r_mc) - float(q_mc) - 0.5 * float(sigma_mc) ** 2) * dt
            vol = float(sigma_mc) * np.sqrt(dt)

            Z = rng.standard_normal((n_conv, int(n_steps)))
            logS = np.log(float(S_mc)) + np.sum(drift + vol * Z, axis=1)
            ST_conv = np.exp(logS)

            disc_pay = _discounted_payoff(ST_conv, float(K_mc), float(r_mc), float(T_mc), option=opt_type.lower())
            n_used, mean, lo, hi = _mc_convergence(disc_pay, batch=max(10_000, n_conv // 20))

            p1, p2 = st.columns(2)
            with p1:
                st.plotly_chart(_paths_fig(tgrid, paths, "Sample GBM Paths"), use_container_width=True)
            with p2:
                st.plotly_chart(_convergence_fig(n_used, mean, lo, hi), use_container_width=True)


# ----------------------------
# Tab 5: Scenario & Risk (Strategies + Stress Testing)  — CLEANED UI + CONSISTENT LABELS
# ----------------------------
with tabs[4]:
    st.subheader("Scenario & Risk")
    st.caption("Stress-test option value and Greeks under spot/vol/time shocks (Black–Scholes).")

    # --- UI spacing polish
    st.markdown(
        """
        <style>
          /* general spacing */
          [data-testid="stVerticalBlock"] { gap: 0.70rem; }
          [data-testid="stNumberInput"] { margin-top: 0.10rem; }
          [data-testid="stSelectbox"] { margin-top: 0.10rem; }
          [data-testid="stSlider"] { padding-top: 0.10rem; }
          [data-testid="stExpander"] { border-radius: 12px; }
          .sr-h { font-size: 1.05rem; font-weight: 750; margin: 0.15rem 0 0.2rem 0; opacity: 0.95; }
          .sr-sub { opacity: 0.70; font-size: 0.88rem; margin-top: -0.15rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    THEME_RED = "rgba(255, 77, 77, 0.90)"
    GRIDLINE = "rgba(255,255,255,0.08)"

    # ----------------------------
    # Helpers
    # ----------------------------
    def _bs_price(opt: str, S: float, K: float, r_: float, q_: float, T: float, sig: float) -> float:
        opt = str(opt).lower().strip()
        if opt == "call":
            return float(bs_call_price(S=S, K=K, r=r_, q=q_, T=T, sigma=sig))
        return float(bs_put_price(S=S, K=K, r=r_, q=q_, T=T, sigma=sig))

    def _leg_greeks(opt: str, S: float, K: float, r_: float, q_: float, T: float, sig: float) -> dict:
        opt = str(opt).lower().strip()
        g = bs_greeks(S=S, K=K, r=r_, q=q_, T=T, sigma=sig)
        if opt == "call":
            return {
                "delta": float(g["delta_call"]),
                "gamma": float(g["gamma"]),
                "vega": float(g["vega"]),
                "theta": float(g["theta_call"]),
                "rho": float(g["rho_call"]),
            }
        return {
            "delta": float(g["delta_put"]),
            "gamma": float(g["gamma"]),
            "vega": float(g["vega"]),
            "theta": float(g["theta_put"]),
            "rho": float(g["rho_put"]),
        }

    def _payoff_at_expiry(opt: str, ST: np.ndarray, K: float) -> np.ndarray:
        opt = str(opt).lower().strip()
        if opt == "call":
            return np.maximum(ST - K, 0.0)
        return np.maximum(K - ST, 0.0)

    def _portfolio_from_legs(legs: list[dict], S: float, r_: float, q_: float, T: float, sig: float) -> dict:
        px = 0.0
        greeks = {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0, "rho": 0.0}

        for leg in legs:
            sign = 1.0 if str(leg["pos"]).lower() == "long" else -1.0
            qty = float(leg["qty"])
            K = float(leg["K"])
            opt = str(leg["opt"]).lower()

            p = _bs_price(opt, S, K, r_, q_, T, sig)
            g = _leg_greeks(opt, S, K, r_, q_, T, sig)

            px += sign * qty * p
            for k in greeks:
                greeks[k] += sign * qty * g[k]

        return {"price": float(px), **greeks}

    def _strategy_legs(strategy: str, opt_type: str, K: float, spread_width: float) -> list[dict]:
        opt_type = str(opt_type).lower().strip()

        if strategy == "Single Option":
            return [{"opt": opt_type, "pos": "long", "K": K, "qty": 1}]

        if strategy == "Vertical Spread":
            if opt_type == "call":
                return [
                    {"opt": "call", "pos": "long",  "K": K,                 "qty": 1},
                    {"opt": "call", "pos": "short", "K": K + spread_width,  "qty": 1},
                ]
            return [
                {"opt": "put", "pos": "long",   "K": K,                 "qty": 1},
                {"opt": "put", "pos": "short",  "K": K - spread_width,  "qty": 1},
            ]

        if strategy == "Straddle":
            return [
                {"opt": "call", "pos": "long", "K": K, "qty": 1},
                {"opt": "put",  "pos": "long", "K": K, "qty": 1},
            ]

        if strategy == "Strangle":
            return [
                {"opt": "call", "pos": "long", "K": K + spread_width, "qty": 1},
                {"opt": "put",  "pos": "long", "K": K - spread_width, "qty": 1},
            ]

        if strategy == "Covered Call":
            return [
                {"opt": "call", "pos": "short", "K": K + spread_width, "qty": 1},
            ]

        # fallback
        return [{"opt": opt_type, "pos": "long", "K": K, "qty": 1}]

    # ----------------------------
    # Inputs (cleaner + less crowded)
    # ----------------------------
    st.markdown("### Inputs")

    left, right = st.columns([1.05, 1.0], gap="large")

    with left:
        st.markdown('<div class="sr-h">Strategy Setup</div>', unsafe_allow_html=True)
        st.markdown('<div class="sr-sub">Choose a preset strategy or build your own multi-leg position.</div>', unsafe_allow_html=True)
        st.markdown("<div style='height: 0.45rem;'></div>", unsafe_allow_html=True)

        a1, a2, a3 = st.columns(3)
        with a1:
            strategy = st.selectbox(
                "Strategy",
                ["Single Option", "Vertical Spread", "Straddle", "Strangle", "Covered Call", "Custom (Multi-Leg)"],
                index=0,
                key="sr_strategy",
            )
        with a2:
            contracts = st.number_input("Contracts", value=1, step=1, min_value=1, key="sr_contracts")
        with a3:
            multiplier = st.number_input("Contract Multiplier", value=100, step=1, min_value=1, key="sr_mult")

        b1, b2, b3 = st.columns(3)
        with b1:
            opt_type = st.selectbox("Base Option Type", ["Call", "Put"], index=0, key="sr_opt_type")
        with b2:
            position = st.selectbox("Position (Single Option)", ["Long", "Short"], index=0, key="sr_pos")
        with b3:
            spread_width = st.number_input(
                "Width (Spreads/Strangles/Covered)",
                value=5.0,
                step=1.0,
                min_value=0.5,
                key="sr_width",
            )

    with right:
        st.markdown('<div class="sr-h">Market Inputs</div>', unsafe_allow_html=True)
        st.markdown('<div class="sr-sub">Base parameters used for pricing/Greeks.</div>', unsafe_allow_html=True)
        st.markdown("<div style='height: 0.45rem;'></div>", unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            S0 = st.number_input("Base Spot (S)", value=100.0, step=1.0, format="%.2f", key="sr_S")
            T0 = st.number_input("Base Maturity (T, years)", value=1.0, step=0.01, format="%.4f", key="sr_T")
            r0 = float(st.session_state.get("sidebar_r", 0.01))
            r_in = st.number_input("Risk-Free Rate (r)", value=float(r0), step=0.001, format="%.4f", key="sr_r")
        with c2:
            K0 = st.number_input("Base Strike (K)", value=100.0, step=1.0, format="%.2f", key="sr_K")
            sig0 = st.number_input("Base Vol (σ)", value=0.20, step=0.01, format="%.4f", key="sr_sig")
            q0 = float(st.session_state.get("sidebar_q", 0.00))
            q_in = st.number_input("Dividend Yield (q)", value=float(q0), step=0.001, format="%.4f", key="sr_q")

    st.divider()

    # ----------------------------
    # Strategy legs (presets or custom table) + allow extra option interaction
    # ----------------------------
    base_opt = opt_type.lower()
    has_stock_leg = (strategy == "Covered Call")

    if strategy != "Custom (Multi-Leg)":
        legs = _strategy_legs(strategy, base_opt, float(K0), float(spread_width))

        # Single option position override
        if strategy == "Single Option":
            legs[0]["pos"] = "long" if position.lower() == "long" else "short"

        # Optional “Add another option” (simple two-leg interaction)
        with st.expander("Add Another Option Leg (Optional)", expanded=False):
            st.caption("Quickly add a second option leg to see interaction in payoff/P&L.")
            add_leg = st.toggle("Enable Extra Leg", value=False, key="sr_add_leg")
            if add_leg:
                e1, e2, e3, e4 = st.columns(4)
                with e1:
                    ex_opt = st.selectbox("Type", ["call", "put"], index=0, key="sr_ex_opt")
                with e2:
                    ex_pos = st.selectbox("Side", ["long", "short"], index=0, key="sr_ex_pos")
                with e3:
                    ex_K = st.number_input("Strike", value=float(K0), step=1.0, format="%.2f", key="sr_ex_K")
                with e4:
                    ex_qty = st.number_input("Qty", value=1.0, step=1.0, min_value=0.0, key="sr_ex_qty")

                legs.append({"opt": ex_opt, "pos": ex_pos, "K": float(ex_K), "qty": float(ex_qty)})

    else:
        has_stock_leg = False
        st.markdown("### Custom Legs")
        st.caption("Add legs to model multi-option positions (e.g., condors, ratio spreads).")

        default_legs = pd.DataFrame(
            [
                {"opt": "call", "pos": "long",  "K": float(K0), "qty": 1.0},
                {"opt": "call", "pos": "short", "K": float(K0) + float(spread_width), "qty": 1.0},
            ]
        )

        legs_df = st.data_editor(
            default_legs,
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "opt": st.column_config.SelectboxColumn("Type", options=["call", "put"]),
                "pos": st.column_config.SelectboxColumn("Side", options=["long", "short"]),
                "K": st.column_config.NumberColumn("Strike (K)", format="%.2f", min_value=0.01, step=0.5),
                "qty": st.column_config.NumberColumn("Qty", format="%.2f", min_value=0.0, step=0.5),
            },
            key="sr_custom_legs",
        )

        legs = []
        for _, row in legs_df.iterrows():
            try:
                legs.append(
                    {
                        "opt": str(row["opt"]).strip().lower(),
                        "pos": str(row["pos"]).strip().lower(),
                        "K": float(row["K"]),
                        "qty": float(row["qty"]),
                    }
                )
            except Exception:
                pass

        if len(legs) == 0:
            st.warning("Add at least one leg to compute portfolio value and payoff.")
            # no st.stop() here; just skip calculations safely
            legs = [{"opt": base_opt, "pos": "long", "K": float(K0), "qty": 1.0}]

    # ----------------------------
    # Shocks (consistent capitalization)
    # ----------------------------
    st.markdown("### Shocks")
    sc1, sc2, sc3 = st.columns(3)
    with sc1:
        spot_shock_pct = st.slider("Spot Shock (%)", -30, 30, 0, step=1, key="sr_spot_shock")
    with sc2:
        vol_shock_abs = st.slider("Vol Shock (Δσ)", -0.20, 0.20, 0.00, step=0.01, key="sr_vol_shock")
    with sc3:
        decay_days = st.slider("Time Decay (days)", 0, 90, 0, step=1, key="sr_decay")

    S1 = float(S0) * (1.0 + float(spot_shock_pct) / 100.0)
    sig1 = max(0.0001, float(sig0) + float(vol_shock_abs))
    T1 = max(1e-6, float(T0) - float(decay_days) / 365.0)

    # ----------------------------
    # Portfolio valuation (base vs shocked)
    # ----------------------------
    base = _portfolio_from_legs(legs, S=float(S0), r_=float(r_in), q_=float(q_in), T=float(T0), sig=float(sig0))
    shocked = _portfolio_from_legs(legs, S=float(S1), r_=float(r_in), q_=float(q_in), T=float(T1), sig=float(sig1))

    # IMPORTANT: Only apply Position sign for Single Option.
    scale = float(contracts) * float(multiplier)
    if strategy == "Single Option":
        scale *= (1.0 if position.lower() == "long" else -1.0)

    base_px = scale * base["price"]
    shocked_px = scale * shocked["price"]
    d_px = shocked_px - base_px

    # Stock leg effect (covered call): long shares per contract
    if has_stock_leg:
        stock_pnl = float(S1 - S0) * float(contracts) * float(multiplier)
        d_px += stock_pnl
        shocked_px += stock_pnl

    pnl = d_px

    # ----------------------------
    # Results
    # ----------------------------
    st.markdown("### Results")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Base Value", f"${base_px:,.2f}")
    m2.metric("Shocked Value", f"${shocked_px:,.2f}")
    m3.metric("Δ Value", f"${d_px:,.2f}")
    m4.metric("P&L", f"${pnl:,.2f}")

    st.caption(
        f"Shocked inputs: S={S1:,.2f}, σ={sig1:.4f}, T={T1:.4f} years. "
        + ("(Covered call includes stock P&L.)" if has_stock_leg else "")
    )

    st.divider()

    # ----------------------------
    # Greeks base vs shocked (scaled)
    # ----------------------------
    st.subheader("Greeks: Base vs Shocked")

    def _g(metric: str) -> tuple[float, float]:
        return scale * float(base[metric]), scale * float(shocked[metric])

    g1, g2, g3, g4, g5 = st.columns(5)
    b, s = _g("delta")
    g1.metric("Delta (Δ)", f"{b:.4f} → {s:.4f}")
    b, s = _g("gamma")
    g2.metric("Gamma (Γ)", f"{b:.6f} → {s:.6f}")
    b, s = _g("vega")
    g3.metric("Vega (ν)", f"{b:.4f} → {s:.4f}")
    b, s = _g("theta")
    g4.metric("Theta (Θ)", f"{b:.4f} → {s:.4f}")
    b, s = _g("rho")
    g5.metric("Rho (ρ)", f"{b:.4f} → {s:.4f}")

    st.caption("Theta is per year. Vega is per 1.00 change in volatility (e.g., 0.20 → 0.21 is +0.01).")

    # ----------------------------
    # Payoff chart
    # ----------------------------
    st.divider()
    st.subheader("Payoff at Expiry")

    with st.expander("Payoff Chart Settings", expanded=False):
        p1, p2 = st.columns(2)
        with p1:
            ST_min = st.number_input("Min Spot", value=50.0, step=5.0, key="sr_stmin")
        with p2:
            ST_max = st.number_input("Max Spot", value=150.0, step=5.0, key="sr_stmax")

    ST_grid = np.linspace(float(ST_min), float(ST_max), 350)

    payoff = np.zeros_like(ST_grid)
    for leg in legs:
        sign = 1.0 if str(leg["pos"]).lower() == "long" else -1.0
        payoff += sign * float(leg["qty"]) * _payoff_at_expiry(leg["opt"], ST_grid, float(leg["K"]))

    if has_stock_leg:
        payoff += (ST_grid - float(S0))

    payoff_scaled = payoff * scale

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ST_grid, y=payoff_scaled, mode="lines", line=dict(color=THEME_RED, width=3)))

    fig.update_layout(
        title="Position Payoff at Expiry",
        height=430,
        margin=dict(l=20, r=20, t=50, b=25),
    )
    fig.update_xaxes(
        title="Underlying at expiry (S<sub>T</sub>)",
        showgrid=True,
        gridcolor=GRIDLINE,
        zeroline=False,
    )
    fig.update_yaxes(title="Payoff ($)", showgrid=True, gridcolor=GRIDLINE, zeroline=False)

    st.plotly_chart(fig, use_container_width=True)

    # ----------------------------
    # Legs table (cleaner)
    # ----------------------------
    with st.expander("Show Strategy Legs", expanded=False):
        legs_df_show = pd.DataFrame(legs).rename(
            columns={"opt": "Type", "pos": "Side", "K": "Strike", "qty": "Qty"}
        )
        legs_df_show["Type"] = legs_df_show["Type"].astype(str).str.upper()
        legs_df_show["Side"] = legs_df_show["Side"].astype(str).str.title()

        st.dataframe(
            legs_df_show,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Strike": st.column_config.NumberColumn(format="%.2f"),
                "Qty": st.column_config.NumberColumn(format="%.2f"),
            },
        )

# ----------------------------
# Tab 6: Binomial Tree (CRR) — FINAL
# ----------------------------
with tabs[5]:  # <-- confirm this matches your Binomial tab index
    st.subheader("Binomial Tree Pricer (CRR)")
    st.caption("Prices European/American call/put using a Cox–Ross–Rubinstein tree (supports dividend yield q).")

    # --- UI polish
    st.markdown(
        """
        <style>
          /* spacing + alignment */
          [data-testid="stVerticalBlock"] { gap: 0.70rem; }
          [data-testid="stNumberInput"] { margin-top: 0.08rem; }
          [data-testid="stSelectbox"] { margin-top: 0.08rem; }
          [data-testid="stSlider"] { padding-top: 0.10rem; }

          .bt-h { font-weight: 800; font-size: 1.05rem; margin: 0.25rem 0 0.05rem 0; }
          .bt-sub { opacity: 0.85; font-size: 0.92rem; margin-top: -0.15rem; }

          /* “card” vibe */
          .bt-card {
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 14px;
            padding: 14px 14px 10px 14px;
            background: rgba(255,255,255,0.02);
          }
          .bt-card-title { font-weight: 750; font-size: 1.00rem; margin-bottom: 6px; }
          .bt-muted { opacity: 0.80; font-size: 0.90rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    THEME_RED = "rgba(255, 77, 77, 0.90)"
    GRIDLINE = "rgba(255,255,255,0.08)"

    # ----------------------------
    # Safe defaults
    # ----------------------------
    S_def = float(st.session_state.get("tab1_S", 100.0))
    K_def = float(st.session_state.get("tab1_K", 100.0))
    T_def = float(st.session_state.get("tab1_T", 1.0))
    sig_def = float(st.session_state.get("tab1_sigma", 0.20))
    r_def = float(st.session_state.get("sidebar_r", 0.01))
    q_def = float(st.session_state.get("sidebar_q", 0.00))

    # ----------------------------
    # Mode + quick toggles
    # ----------------------------
    topL, topR, topR2 = st.columns([1.35, 1.0, 1.0])
    with topL:
        st.markdown("<div style='height: 0.55rem;'></div>", unsafe_allow_html=True)
        edu_mode = st.toggle("Education Mode", value=True, key="bt_edu_mode")
        st.markdown(
            '<div class="bt-sub">Education Mode emphasizes intuition (Δ + bond), diagnostics, and keeps the preview readable.</div>',
            unsafe_allow_html=True,
        )
    with topR:
        show_preview = st.toggle("Show Tree Preview", value=True, key="bt_show_preview")
    with topR2:
        show_greeks = st.toggle("Show Greeks", value=True, key="bt_show_greeks")

    st.divider()

    # ----------------------------
    # Inputs (clean layout)
    # ----------------------------
    st.markdown('<div class="bt-h">Inputs</div>', unsafe_allow_html=True)

    a1, a2, a3 = st.columns(3)
    with a1:
        opt_type = st.selectbox("Option Type", ["Call", "Put"], index=0, key="bt_opt_type")
        S_bt = st.number_input("Spot (S)", value=S_def, step=1.0, format="%.2f", key="bt_S")
        K_bt = st.number_input("Strike (K)", value=K_def, step=1.0, format="%.2f", key="bt_K")
    with a2:
        exercise = st.selectbox("Exercise Style", ["European", "American"], index=1, key="bt_ex_style")
        T_bt = st.number_input("Maturity (T, years)", value=T_def, step=0.01, format="%.4f", key="bt_T")
        sigma_bt = st.number_input("Volatility (σ)", value=sig_def, step=0.01, format="%.4f", key="bt_sigma")
    with a3:
        r_bt = st.number_input("Risk-free Rate (r)", value=r_def, step=0.001, format="%.4f", key="bt_r")
        q_bt = st.number_input("Dividend Yield (q)", value=q_def, step=0.001, format="%.4f", key="bt_q")

    st.markdown('<div class="bt-h">Tree Settings</div>', unsafe_allow_html=True)

    default_N = 8 if edu_mode else 100
    default_preview = 5 if edu_mode else 6

    b1, b2, b3, b4 = st.columns([1.55, 1.35, 1.2, 0.9])
    with b1:
        n_steps = st.slider(
            "Steps (N)",
            min_value=2,
            max_value=300,
            value=int(min(max(default_N, 2), 300)),
            step=1,
            key="bt_steps",
        )
        st.caption("Tip: 2–8 for learning; 50–300 for better accuracy.")
    with b2:
        preview_n = st.slider(
            "Preview Steps",
            2,
            8,
            int(min(max(default_preview, 2), 8, n_steps)),
            step=1,
            key="bt_preview_n",
        )
        st.caption("Preview stays small for readability.")
    with b3:
        show_convergence = st.toggle("Show Convergence", value=not edu_mode, key="bt_show_conv")
    with b4:
        run_bt = st.button("Compute", type="primary", key="bt_compute")

    if not run_bt:
        st.info("Set inputs and click **Compute**.")
    else:
        # ----------------------------
        # Core pricing
        # ----------------------------
        try:
            stockN, optN, p, dt, u, d, disc = crr_binomial_tree(
                S=float(S_bt),
                K=float(K_bt),
                r=float(r_bt),
                q=float(q_bt),
                T=float(T_bt),
                sigma=float(sigma_bt),
                n_steps=int(n_steps),
                option=opt_type.lower(),
                exercise=exercise.lower(),
            )
            price_tree = float(optN[0][0])
        except Exception as e:
            st.error("Binomial tree computation failed.")
            st.exception(e)
            price_tree = None

        if price_tree is not None:
            # ----------------------------
            # Black–Scholes (European only)
            # ----------------------------
            bs_bench = None
            if exercise.lower() == "european":
                if opt_type.lower() == "call":
                    bs_bench = float(
                        bs_call_price(
                            S=float(S_bt), K=float(K_bt), r=float(r_bt), q=float(q_bt), T=float(T_bt), sigma=float(sigma_bt)
                        )
                    )
                else:
                    bs_bench = float(
                        bs_put_price(
                            S=float(S_bt), K=float(K_bt), r=float(r_bt), q=float(q_bt), T=float(T_bt), sigma=float(sigma_bt)
                        )
                    )

            # ----------------------------
            # Results “card” (clean, consistent)
            # ----------------------------
            st.markdown('<div class="bt-card">', unsafe_allow_html=True)
            st.markdown('<div class="bt-card-title">Results</div>', unsafe_allow_html=True)

            m1, m2, m3, m4, m5 = st.columns([1.1, 1.0, 1.0, 1.0, 1.0])
            m1.metric("Tree Price", f"${price_tree:.4f}")
            m2.metric("Up Prob (p)", f"{p:.4f}")
            m3.metric("u / d", f"{u:.4f} / {d:.4f}")
            m4.metric("Δt", f"{dt:.6f}")
            m5.metric("Black–Scholes", "—" if bs_bench is None else f"${bs_bench:.4f}")

            if bs_bench is not None:
                diff = float(price_tree - bs_bench)
                st.markdown(
                    f'<div class="bt-muted">Tree − Black–Scholes: <b>{diff:+.5f}</b> '
                    f'(should converge toward 0 as N increases)</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div class="bt-muted">Black–Scholes shown only for European options.</div>',
                    unsafe_allow_html=True,
                )

            st.markdown("</div>", unsafe_allow_html=True)

            # ----------------------------
            # Diagnostics (no-arb checks)
            # ----------------------------
            with st.expander("Diagnostics (no-arbitrage checks)", expanded=edu_mode):
                growth = float(np.exp((float(r_bt) - float(q_bt)) * float(dt)))
                ok_prob = (0.0 <= float(p) <= 1.0)
                ok_bounds = (float(d) < growth < float(u))

                cA, cB, cC = st.columns(3)
                cA.metric("Growth per step", f"{growth:.6f}")
                cB.metric("Prob OK?", "✅" if ok_prob else "⚠️")
                cC.metric("No-arb bounds?", "✅" if ok_bounds else "⚠️")

                if not ok_prob or not ok_bounds:
                    st.warning(
                        "Potential no-arbitrage issues for this discretization. "
                        "Try increasing N (smaller Δt) or adjusting r/q/σ/T."
                    )
                else:
                    st.caption("Looks good: p is in [0,1] and d < e^{(r−q)Δt} < u.")

            # ----------------------------
            # Replicating portfolio (intuition)
            # ----------------------------
            with st.expander("Replicating Portfolio at t=0 (Δ + bond)", expanded=edu_mode):
                # Grab first-step nodes robustly
                Su = float(max(stockN[1][0], stockN[1][1]))
                Sd = float(min(stockN[1][0], stockN[1][1]))
                Vu = float(max(optN[1][0], optN[1][1]))
                Vd = float(min(optN[1][0], optN[1][1]))

                denom = (Su - Sd)
                delta0 = (Vu - Vd) / denom if denom != 0 else np.nan
                B0 = float(np.exp(-float(r_bt) * float(dt)) * (Vu - float(delta0) * Su)) if np.isfinite(delta0) else np.nan

                rr1, rr2, rr3 = st.columns(3)
                rr1.metric("Δ (shares)", f"{delta0:.6f}" if np.isfinite(delta0) else "—")
                rr2.metric("Bond (PV)", f"{B0:.6f}" if np.isfinite(B0) else "—")
                rr3.metric("Check: ΔS + B", f"{(delta0*float(S_bt)+B0):.6f}" if (np.isfinite(delta0) and np.isfinite(B0)) else "—")

                st.caption("Intuition: the option is replicated by holding Δ shares plus a bond position.")

            # ----------------------------
            # Tree preview (small N)
            # ----------------------------
            if show_preview:
                st.divider()
                st.markdown('<div class="bt-h">Tree Preview</div>', unsafe_allow_html=True)
                st.caption("Small tree for readability (educational).")

                try:
                    stockP, optP, *_ = crr_binomial_tree(
                        S=float(S_bt),
                        K=float(K_bt),
                        r=float(r_bt),
                        q=float(q_bt),
                        T=float(T_bt),
                        sigma=float(sigma_bt),
                        n_steps=int(preview_n),
                        option=opt_type.lower(),
                        exercise=exercise.lower(),
                    )
                    st.plotly_chart(binomial_tree_preview_fig(stockP, optP), use_container_width=True)
                except Exception as e:
                    st.warning("Tree preview failed (pricing still may be fine).")
                    st.exception(e)

            # ----------------------------
            # Greeks (finite-difference, tree-based)
            # ----------------------------
            if show_greeks:
                with st.expander("Greeks (tree-based finite differences)", expanded=False):
                    st.caption("Computed by re-pricing with small bumps. Works for European/American.")

                    gA, gB, gC = st.columns(3)
                    with gA:
                        dS = st.number_input("Spot bump (ΔS)", value=1.0, step=0.5, min_value=0.01, key="bt_g_dS")
                    with gB:
                        dSig = st.number_input("Vol bump (Δσ)", value=0.01, step=0.005, min_value=0.0005, key="bt_g_dSig")
                    with gC:
                        dT_days = st.number_input("Theta step (days)", value=1, step=1, min_value=1, max_value=30, key="bt_g_dT")

                    def _price_tree(S_, sig_, T_):
                        _, opt_, *_m = crr_binomial_tree(
                            S=float(S_),
                            K=float(K_bt),
                            r=float(r_bt),
                            q=float(q_bt),
                            T=float(T_),
                            sigma=float(sig_),
                            n_steps=int(n_steps),
                            option=opt_type.lower(),
                            exercise=exercise.lower(),
                        )
                        return float(opt_[0][0])

                    try:
                        S0 = float(S_bt)
                        sig0 = float(sigma_bt)
                        T0 = float(T_bt)

                        V0 = float(price_tree)
                        V_up = _price_tree(S0 + float(dS), sig0, T0)
                        V_dn = _price_tree(max(0.01, S0 - float(dS)), sig0, T0)

                        delta = (V_up - V_dn) / (2.0 * float(dS))
                        gamma = (V_up - 2.0 * V0 + V_dn) / (float(dS) ** 2)

                        V_sig_up = _price_tree(S0, sig0 + float(dSig), T0)
                        V_sig_dn = _price_tree(S0, max(0.0001, sig0 - float(dSig)), T0)
                        vega = (V_sig_up - V_sig_dn) / (2.0 * float(dSig))

                        dT = float(dT_days) / 365.0
                        T_short = max(1e-6, T0 - dT)
                        V_short = _price_tree(S0, sig0, T_short)
                        theta = (V_short - V0) / (-dT)  # per year

                        gg1, gg2, gg3, gg4 = st.columns(4)
                        gg1.metric("Delta (Δ)", f"{delta:.6f}")
                        gg2.metric("Gamma (Γ)", f"{gamma:.6f}")
                        gg3.metric("Vega (ν)", f"{vega:.6f}")
                        gg4.metric("Theta (Θ / year)", f"{theta:.6f}")

                        st.caption("Tree Greeks are approximations — increase N for stability.")
                    except Exception as e:
                        st.warning("Greeks computation failed.")
                        st.exception(e)

            # ----------------------------
            # Convergence (price vs N)
            # ----------------------------
            if show_convergence:
                with st.expander("Convergence (price vs steps N)", expanded=False):
                    st.caption("Runs on-demand to keep the app fast.")

                    c1, c2 = st.columns([1.6, 1.0])
                    with c1:
                        n_list_str = st.text_input(
                            "Steps list (comma-separated)",
                            value="2, 4, 6, 8, 12, 16, 25, 50, 75, 100, 150, 200",
                            key="bt_conv_list",
                        )
                    with c2:
                        conv_btn = st.button("Compute Convergence", key="bt_conv_btn")

                    def _parse_n_list(s: str):
                        out = []
                        for part in s.split(","):
                            part = part.strip()
                            if not part:
                                continue
                            try:
                                n = int(part)
                                if 2 <= n <= 300:
                                    out.append(n)
                            except Exception:
                                pass
                        return sorted(list(dict.fromkeys(out)))

                    @st.cache_data(show_spinner=False)
                    def _conv_prices(S, K, r, q, T, sigma, option, exercise, Ns):
                        ys = []
                        for n in Ns:
                            _, opt_, *_m = crr_binomial_tree(
                                S=float(S),
                                K=float(K),
                                r=float(r),
                                q=float(q),
                                T=float(T),
                                sigma=float(sigma),
                                n_steps=int(n),
                                option=option,
                                exercise=exercise,
                            )
                            ys.append(float(opt_[0][0]))
                        return ys

                    if conv_btn:
                        Ns = _parse_n_list(n_list_str)
                        if len(Ns) < 2:
                            st.info("Enter at least two valid N values between 2 and 300.")
                        else:
                            try:
                                ys = _conv_prices(
                                    S=float(S_bt),
                                    K=float(K_bt),
                                    r=float(r_bt),
                                    q=float(q_bt),
                                    T=float(T_bt),
                                    sigma=float(sigma_bt),
                                    option=opt_type.lower(),
                                    exercise=exercise.lower(),
                                    Ns=tuple(Ns),
                                )

                                fig = go.Figure()
                                fig.add_trace(
                                    go.Scatter(
                                        x=Ns,
                                        y=ys,
                                        mode="lines+markers",
                                        line=dict(color=THEME_RED, width=3),
                                        marker=dict(size=8),
                                        name="Tree Price",
                                    )
                                )
                                if bs_bench is not None:
                                    fig.add_hline(y=float(bs_bench), line_dash="dash", opacity=0.6)

                                fig.update_layout(
                                    title="Convergence of Tree Price vs Steps (N)",
                                    height=420,
                                    margin=dict(l=20, r=20, t=55, b=25),
                                )
                                fig.update_xaxes(title="Steps (N)", showgrid=True, gridcolor=GRIDLINE, zeroline=False)
                                fig.update_yaxes(title="Option Price", showgrid=True, gridcolor=GRIDLINE, zeroline=False)

                                st.plotly_chart(fig, use_container_width=True)
                                st.caption("If the curve flattens as N increases, your price is converging.")
                            except Exception as e:
                                st.warning("Convergence computation failed.")
                                st.exception(e)