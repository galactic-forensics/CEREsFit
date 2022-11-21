"""Test properties, error messages, etc. on Mahon Linear regression."""

import numpy as np
import pytest

from curefit import LinReg
from .test_curefit_linreg_datasets import read_dataset


def test_linreg_fixpoint_value_error():
    """Raise ValueError if fix point is of bad shape."""
    some_arr = np.array([1, 2])
    with pytest.raises(ValueError):
        _ = LinReg(some_arr, some_arr, some_arr, some_arr, fixpt=42, autocalc=False)


def test_linreg_keywords(ds_path):
    """Ensure keywords are accepted by routine."""
    case = "set1.csv"
    ds = ds_path.joinpath(case)
    xdat, xunc, ydat, yunc, rho, _, params_exp = read_dataset(ds)

    reg_limit = 42
    iter_max = 13
    kwargs = {"regression_limit": reg_limit, "iter_max": iter_max}

    reg = LinReg(xdat, xunc, ydat, yunc, rho=rho, autocalc=False, **kwargs)

    assert reg.reg_limit == reg_limit
    assert reg.iter_max == iter_max


def test_linreg_iteration_warning(ds_path):
    """Test that iteration throws a warning if runs out of max_iter."""
    case = "set1.csv"
    ds = ds_path.joinpath(case)
    xdat, xunc, ydat, yunc, rho, _, params_exp = read_dataset(ds)
    iter_max = 1

    with pytest.warns(UserWarning):
        _ = LinReg(xdat, xunc, ydat, yunc, rho=rho, iter_max=iter_max)


def test_linreg_properties(ds_path):
    """Test that properties return the correct parameters."""
    case = "set1.csv"
    ds = ds_path.joinpath(case)
    xdat, xunc, ydat, yunc, rho, _, params_exp = read_dataset(ds)

    ind_params = 0  # index of which parameters for given case

    reg = LinReg(xdat, xunc, ydat, yunc, rho=rho)

    assert params_exp[ind_params][0:2] == pytest.approx(reg.slope, abs=1e-6)
    assert params_exp[ind_params][2:4] == pytest.approx(reg.intercept, abs=1e-6)
    assert params_exp[ind_params][4] == pytest.approx(reg.mswd, abs=1e-3)


def test_linreg_mswd_ci():
    """Confidence intervals for MSWD."""
    dof = 9
    mswd_ci_exp = 0.300043277775595, 2.11364086651574
    some_arr = np.array([1, 2])

    reg = LinReg(some_arr, some_arr, some_arr, some_arr, auto=False)
    reg._dof = dof

    mswd_ci_rec = reg.mswd_ci()
    assert mswd_ci_exp == pytest.approx(mswd_ci_rec)


def test_linreg_regression_line():
    """Get a regression line for the specified values."""
    some_arr = np.array([1, 2])

    reg = LinReg(some_arr, some_arr, some_arr, some_arr, auto=False)
    reg._slope = 3.14
    reg._intercept = 12.0

    xdat_exp = np.array([1, 2])
    ydat_exp = xdat_exp * reg._slope + reg._intercept

    xdat_rec, ydat_rec = reg.regression_line()

    assert xdat_rec == pytest.approx(xdat_exp)
    assert ydat_rec == pytest.approx(ydat_exp)
