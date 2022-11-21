"""Test the Mahon Linear Regression with the data sets from Stephan and Trappitsch."""

from pathlib import Path
from typing import List

import numpy as np
import pytest

from curefit import LinReg


DATASETS = ["set1.csv", "set2.csv", "set3.csv", "set4.csv", "set5.csv"]
# absolute precision for comparison of non MSWD parameters
PRECISION_ABS = [1e-6, 1e-4, 1e-4, 1e-3, 1e-4]


def read_dataset(ds: Path) -> List[np.ndarray]:
    """Read a specific dataset and return the data.

    :param ds: Path to the dataset csv file.
    :return: xdata, xunc, ydata, yunc, rho, fix_point, parameters
        Parameters contain: Slope, Slope_unc, Intercept, Intercept_unc, MSWD
    """
    load_kwargs = {"delimiter": ",", "skiprows": 1}
    max_rows = [None, None, None, None, None, 1, 1, 3, 3, 3, 3, 3]
    cols_to_read = 12
    read_data = []
    for it in range(cols_to_read):
        read_data.append(
            np.loadtxt(ds, usecols=it, max_rows=max_rows[it], **load_kwargs)
        )

    # prepare return data
    ret_data = read_data[:5]
    ret_data.append(np.array([read_data[5], read_data[6]]))
    ret_data.append(np.array(read_data[7:]).transpose())
    return ret_data


@pytest.mark.parametrize("case", zip(DATASETS, PRECISION_ABS))
def test_linreg_cases_a(case, ds_path):
    """Comparpe cases with correlated uncertainties, no fixed point (sets a)."""
    ds = ds_path.joinpath(case[0])
    xdat, xunc, ydat, yunc, rho, _, params_exp = read_dataset(ds)

    ind_params = 0  # index of which parameters for given case

    reg = LinReg(xdat, xunc, ydat, yunc, rho=rho)
    assert params_exp[ind_params][:4] == pytest.approx(reg.parameters[:4], abs=case[1])
    assert params_exp[ind_params][4] == pytest.approx(reg.parameters[4], rel=1e-3)

    # check mswd and chi squared
    assert reg.chi_squared == pytest.approx(reg.mswd * (len(xdat) - 2))


@pytest.mark.parametrize("case", zip(DATASETS, PRECISION_ABS))
def test_linreg_cases_b(case, ds_path):
    """Comparpe cases with uncorrelated uncertainties, no fixed point (sets b)."""
    ds = ds_path.joinpath(case[0])
    xdat, xunc, ydat, yunc, _, _, params_exp = read_dataset(ds)

    ind_params = 1  # index of which parameters for given case

    reg = LinReg(xdat, xunc, ydat, yunc, rho=None)
    assert params_exp[ind_params][:4] == pytest.approx(reg.parameters[:4], abs=case[1])
    assert params_exp[ind_params][4] == pytest.approx(reg.parameters[4], rel=1e-3)


@pytest.mark.parametrize("case", zip(DATASETS, PRECISION_ABS))
def test_linreg_cases_c(case, ds_path):
    """Comparpe cases with correlated uncertainties, with fixed point (sets c)."""
    ds = ds_path.joinpath(case[0])
    xdat, xunc, ydat, yunc, rho, fixpt, params_exp = read_dataset(ds)

    ind_params = 2  # index of which parameters for given case

    reg = LinReg(xdat, xunc, ydat, yunc, rho=rho, fixpt=fixpt)
    assert params_exp[ind_params][:4] == pytest.approx(reg.parameters[:4], abs=case[1])
    assert params_exp[ind_params][4] == pytest.approx(reg.parameters[4], rel=1e-3)

    # check mswd and chi squared
    assert reg.chi_squared == pytest.approx(reg.mswd * (len(xdat) - 1))
