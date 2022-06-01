"""Provide fixtures for test suite."""

from pathlib import Path
from typing import Tuple

import numpy as np
import pytest


@pytest.fixture(scope="package")
def ds_path(request) -> Path:
    """Provide the path to the data set folder."""
    curr = Path(request.fspath).parents[0]
    return Path(curr).joinpath("datasets").absolute()


@pytest.fixture(scope="package")
def stephan_ci_data(
    ds_path,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Provide confidence interval data from Stephan Macro for testing.

    The data and results are stored in the datasets folder.

    :param ds_path: Pytest fixture to provide the path to the data.

    :return: xdat, sigx, ydat, sigy, rho, x_ci, y_ci_min, y_ci_max
    """
    data_fname = ds_path.joinpath("ci_calc_data.csv")
    results_fname = ds_path.joinpath("ci_calc_results.csv")

    data = np.loadtxt(data_fname, skiprows=1, delimiter=",")
    results = np.loadtxt(results_fname, skiprows=1, delimiter=",")

    return (
        data[:, 0],
        data[:, 1],
        data[:, 2],
        data[:, 3],
        data[:, 4],
        results[:, 0],
        results[:, 1],
        results[:, 2],
    )
