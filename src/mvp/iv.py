import numpy as np
from src.mvp.bs import bs_call_price


def implied_vol_call(
    price: float,
    S: float,
    K: float,
    r: float,
    q: float,
    T: float,
    tol: float = 1e-6,
    max_iter: int = 100,
) -> float:
    """
    Compute Black–Scholes implied volatility for a European call option
    using bisection.

    Returns np.nan if no solution is found.
    """

    # No-arbitrage bounds
    intrinsic = max(S * np.exp(-q * T) - K * np.exp(-r * T), 0.0)
    if price < intrinsic or price <= 0 or T <= 0:
        return np.nan

    vol_low = 1e-6
    vol_high = 5.0

    for _ in range(max_iter):
        vol_mid = 0.5 * (vol_low + vol_high)
        price_mid = bs_call_price(S, K, r, q, T, vol_mid)

        if abs(price_mid - price) < tol:
            return vol_mid

        if price_mid > price:
            vol_high = vol_mid
        else:
            vol_low = vol_mid

    return np.nan
