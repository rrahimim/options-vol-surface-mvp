import numpy as np

from src.mvp.bs import bs_call_price, bs_put_price
from src.mvp.iv import implied_vol


def test_implied_vol_roundtrip_call():
    S, K, r, q, T, sigma = 100, 110, 0.03, 0.01, 0.5, 0.25
    price = bs_call_price(S, K, r, q, T, sigma)
    iv = implied_vol(price, S, K, r, q, T, "call")
    assert np.isclose(iv, sigma, atol=1e-8)


def test_implied_vol_roundtrip_put():
    S, K, r, q, T, sigma = 100, 90, 0.03, 0.01, 0.5, 0.25
    price = bs_put_price(S, K, r, q, T, sigma)
    iv = implied_vol(price, S, K, r, q, T, "put")
    assert np.isclose(iv, sigma, atol=1e-8)