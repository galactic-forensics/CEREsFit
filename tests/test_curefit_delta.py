"""Test Konrecker delta function."""

import numpy as np
import pytest

from curefit.curefit_linreg import kron_delta


def test_kron_delta():
    """Ensure that 1 is returned if two indexes are the same, zero otherwise."""
    assert kron_delta(1, 1) == 1
    assert kron_delta(1, 2) == 0


def test_kron_delta_array():
    """Kronecker delta return for an array of indexes."""
    in1 = np.array([1, 2, 3, 4])
    in2 = np.array([1, 2, 4, 4])
    in2alt = [1, 2, 4, 4]  # as list
    out_exp = np.array([1, 1, 0, 1])
    np.testing.assert_equal(kron_delta(in1, in2), out_exp)
    np.testing.assert_equal(kron_delta(in1, in2alt), out_exp)


def test_kron_delta_value_error():
    """Raise ValueError if shapes mismatch."""
    arr1 = np.array([1, 2])
    arr2 = np.array([1])
    num = 3
    with pytest.raises(ValueError) as err_info:
        kron_delta(num, arr1)
    err_msg = err_info.value.args[0]
    assert err_msg == "The inputs must have the same shape."

    with pytest.raises(ValueError) as err_info:
        kron_delta(arr1, arr2)
    err_msg = err_info.value.args[0]
    assert err_msg == "The inputs must have the same shape."
