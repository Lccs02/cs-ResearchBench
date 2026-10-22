import numpy as np
import pytest
from researchbench.metrics.regression import regression_metrics


def test_hand_calculated_weighted():
    result = regression_metrics([0, 0, 100], [1, 3, -100], weights=[1, 3, 0])
    assert result["MAE"]["value"] == 2.5
    assert result["RMSE"]["value"] == pytest.approx(np.sqrt(7))
    assert result["MAE"]["n_valid"] == 2
    assert result["MAE"]["n_zero_weight"] == 1


def test_omit_counts_and_pairwise_mask():
    r = regression_metrics([0, np.nan, 0, 0], [2, 1, np.inf, 4], weights=[1, 0, 1, 0], missing="omit")["MAE"]
    assert r["value"] == 2
    assert (r["n_total"], r["n_valid"], r["n_missing"], r["n_zero_weight"]) == (4, 1, 2, 1)


@pytest.mark.parametrize("true,pred,kwargs", [
    ([1, 2], [[1, 2]], {}), ([], [], {}),
    ([0], [np.nan], {}), ([np.nan], [0], {"missing": "omit"}),
    ([0], [1], {"weights": [0]}), ([0], [1], {"weights": [-1]}),
    ([0], [1], {"weights": [np.inf]}), ([0], [1], {"weights": 1}),
    ([0], [1], {"missing": "ignore"}),
])
def test_invalid_inputs(true, pred, kwargs):
    with pytest.raises(ValueError):
        regression_metrics(true, pred, **kwargs)


def test_arbitrary_dimensions_and_large_scale():
    rng = np.random.default_rng(42)
    true = rng.normal(size=(2, 3, 4, 5))
    r = regression_metrics(true, true + 2)
    assert r["MAE"]["value"] == pytest.approx(2)
    assert r["RMSE"]["value"] == pytest.approx(2)
    assert regression_metrics([0], [1e200])["RMSE"]["value"] == 1e200
