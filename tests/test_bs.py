import numpy as np
from src.mvp.bs import bs_call_price


def test_bs_call_price_known_value():
    price = bs_call_price(
        S=100,
        K=100,
        r=0.05,
        q=0.0,
        T=1.0,
        sigma=0.2,
    )
    assert np.isclose(price, 10.450583572185565, atol=1e-10)
