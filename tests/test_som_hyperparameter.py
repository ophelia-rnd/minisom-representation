import numpy as np
from minisom_representation.som_hyperparameter import (
    calc_som_hyparams,
    calc_initial_sigma,
)

def test_calc_som_hyparams():
    np.random.seed(42)
    X = np.random.randn(50, 4)
    params = calc_som_hyparams(X)

    assert "d1" in params
    assert "d2" in params
    assert "sigma" in params

    assert isinstance(params["d1"], int)
    assert isinstance(params["d2"], int)
    assert isinstance(params["sigma"], float)

    assert params["d1"] > 0
    assert params["d2"] > 0
    assert params["sigma"] > 0

def test_calc_initial_sigma_float():
    sigma = calc_initial_sigma(6, 12, factor=4.0)
    assert isinstance(sigma, float)
    assert sigma == 3.0
