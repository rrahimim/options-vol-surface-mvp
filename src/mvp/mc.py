from __future__ import annotations

import numpy as np


def mc_european_price(
    S: float,
    K: float,
    r: float,
    q: float,
    T: float,
    sigma: float,
    option: str = "call",
    n_paths: int = 200_000,
    n_steps: int = 252,
    antithetic: bool = True,
    seed: int | None = 42,
) -> dict:
    """
    Monte Carlo pricer for European options under GBM:
      dS = (r-q)S dt + sigma S dW

    Returns dict:
      price, stderr, ci_low, ci_high
    """
    option = option.lower().strip()
    if option not in {"call", "put"}:
        raise ValueError("option must be 'call' or 'put'")

    if T <= 0 or sigma < 0 or S <= 0 or K <= 0:
        return {"price": float("nan"), "stderr": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}

    rng = np.random.default_rng(seed)
    dt = T / n_steps
    drift = (r - q - 0.5 * sigma * sigma) * dt
    vol = sigma * np.sqrt(dt)

    # If antithetic, generate half and mirror to reduce variance
    if antithetic:
        half = n_paths // 2
        Z = rng.standard_normal((half, n_steps))
        Z = np.vstack([Z, -Z])
        if Z.shape[0] < n_paths:
            extra = rng.standard_normal((n_paths - Z.shape[0], n_steps))
            Z = np.vstack([Z, extra])
    else:
        Z = rng.standard_normal((n_paths, n_steps))

    # simulate log-returns and terminal price
    log_paths = np.cumsum(drift + vol * Z, axis=1)
    ST = S * np.exp(log_paths[:, -1])

    if option == "call":
        payoff = np.maximum(ST - K, 0.0)
    else:
        payoff = np.maximum(K - ST, 0.0)

    disc = np.exp(-r * T)
    pv = disc * payoff

    price = float(pv.mean())
    stderr = float(pv.std(ddof=1) / np.sqrt(len(pv)))
    ci_low = price - 1.96 * stderr
    ci_high = price + 1.96 * stderr

    return {"price": price, "stderr": stderr, "ci_low": ci_low, "ci_high": ci_high}
