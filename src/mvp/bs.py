import numpy as np
from scipy.stats import norm


def _d1_d2(S: float, K: float, r: float, q: float, T: float, sigma: float) -> tuple[float, float]:
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2


def bs_call_price(S: float, K: float, r: float, q: float, T: float, sigma: float) -> float:
    if T <= 0:
        return max(S - K, 0.0)

    if sigma <= 0:
        return max(S * np.exp(-q * T) - K * np.exp(-r * T), 0.0)

    d1, d2 = _d1_d2(S, K, r, q, T, sigma)
    return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def bs_put_price(S: float, K: float, r: float, q: float, T: float, sigma: float) -> float:
    if T <= 0:
        return max(K - S, 0.0)

    if sigma <= 0:
        return max(K * np.exp(-r * T) - S * np.exp(-q * T), 0.0)

    d1, d2 = _d1_d2(S, K, r, q, T, sigma)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)


def bs_vega(S: float, K: float, r: float, q: float, T: float, sigma: float) -> float:
    """Vega = dPrice/dVol (per 1.0 vol, not 1%)."""
    if T <= 0 or sigma <= 0:
        return 0.0
    d1, _ = _d1_d2(S, K, r, q, T, sigma)
    return S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)

# --- Greeks (Black–Scholes with continuous dividend yield q) ---
from math import log, sqrt, exp, erf, pi

def _N(x: float) -> float:
    """Standard normal CDF."""
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))

def _n(x: float) -> float:
    """Standard normal PDF."""
    return (1.0 / sqrt(2.0 * pi)) * exp(-0.5 * x * x)

def bs_greeks(S: float, K: float, r: float, q: float, T: float, sigma: float) -> dict:
    """
    Returns Black-Scholes greeks for call & put.
    Outputs:
      delta_call, delta_put, gamma, vega, theta_call, theta_put, rho_call, rho_put
    Notes:
      - theta is per YEAR (not per day)
      - vega is per 1.0 vol (so divide by 100 if you want per 1 vol point)
      - Requires T>0 and sigma>0
    """
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return {
            "delta_call": float("nan"),
            "delta_put": float("nan"),
            "gamma": float("nan"),
            "vega": float("nan"),
            "theta_call": float("nan"),
            "theta_put": float("nan"),
            "rho_call": float("nan"),
            "rho_put": float("nan"),
        }

    vsqrt = sigma * sqrt(T)
    d1 = (log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / vsqrt
    d2 = d1 - vsqrt

    Nd1 = _N(d1)
    Nd2 = _N(d2)
    n_d1 = _n(d1)

    disc_q = exp(-q * T)
    disc_r = exp(-r * T)

    # Deltas
    delta_call = disc_q * Nd1
    delta_put = disc_q * (Nd1 - 1.0)

    # Gamma (same for call/put)
    gamma = (disc_q * n_d1) / (S * vsqrt)

    # Vega (same for call/put)
    vega = S * disc_q * n_d1 * sqrt(T)

    # Theta (per year)
    term1 = -(S * disc_q * n_d1 * sigma) / (2.0 * sqrt(T))
    theta_call = term1 + q * S * disc_q * Nd1 - r * K * disc_r * Nd2
    theta_put  = term1 - q * S * disc_q * _N(-d1) + r * K * disc_r * _N(-d2)

    # Rho
    rho_call = K * T * disc_r * Nd2
    rho_put  = -K * T * disc_r * _N(-d2)

    return {
        "delta_call": delta_call,
        "delta_put": delta_put,
        "gamma": gamma,
        "vega": vega,
        "theta_call": theta_call,
        "theta_put": theta_put,
        "rho_call": rho_call,
        "rho_put": rho_put,
    }

