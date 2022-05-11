import numpy as np
import pytest

from score_analysis import utils


@pytest.mark.parametrize("method", ["bc", "bca"])
def test_bootstrap_ci_error(method):
    with pytest.raises(ValueError):
        utils.bootstrap_ci(np.array([1, 2, 3]), theta_hat=None, method=method)
