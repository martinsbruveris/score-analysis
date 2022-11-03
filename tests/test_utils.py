import numpy as np
import pytest

from score_analysis import utils


@pytest.mark.parametrize("method", ["bc", "bca"])
def test_bootstrap_ci_error(method):
    with pytest.raises(ValueError):
        utils.bootstrap_ci(np.array([1, 2, 3]), theta_hat=None, method=method)


@pytest.mark.parametrize(
    "x, y, t, s",
    [
        # Multiple t, scalar output for each
        [[0, 1], [0, 1], [0.2, 0.3, 0.4], [0.2, 0.3, 0.4]],
        # One input (as array), multiple outputs
        [[0, 1, 2], [0, 1, -1], [0.5], [[0.5, 1.25]]],
        # Scalar input, multiple outputs
        [[0, 1, 2], [0, 1, -1], 0.5, [0.5, 1.25]],
        # No solution
        [[0, 1], [0, 1], 2, [1]],
    ],
)
def test_invert_pl_function(x, y, t, s):
    res = utils.invert_pl_function(np.asarray(x), np.asarray(y), np.asarray(t))

    if isinstance(s, list):
        for a, b in zip(s, res):
            np.testing.assert_allclose(a, b)
    else:
        np.testing.assert_allclose(s, res)
