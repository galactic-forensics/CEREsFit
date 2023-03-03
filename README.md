# CEREsFit

[![pypi](https://img.shields.io/pypi/v/ceresfit?color=informational)](https://pypi.org/project/ceresfit/)
[![tests](https://github.com/galactic-forensics/CEREsFit/actions/workflows/package_testing.yml/badge.svg)](https://github.com/galactic-forensics/CEREsFit/actions/workflows/package_testing.yml)
[![codecov](https://codecov.io/gh/galactic-forensics/CEREsFit/branch/main/graph/badge.svg?token=C8KN5UE831)](https://codecov.io/gh/galactic-forensics/CEREsFit)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/galactic-forensics/CEREsFit/main.svg)](https://results.pre-commit.ci/latest/github/galactic-forensics/CEREsFit/main)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

The goal of CEREsFit (Correlated Errors Regression Estimate Fit)
is to provide a python package
that allows to calculate linear regressions on data sets with correlated uncertainties.
The calculations follow the methodology published by
[Mahon (1996)](https://doi.org/10.1080/00206819709465336).
Typos and errors that were made in that work have been corrected.
A method to allow calculating a linear regression through a fixed point,
avoiding previously made errors,
is also provided.

## Installation

The package can be installed from `pypi` via:

```
pip install ceresfit
```

## Usage

Below is an example on how to use the package.

```python
>>> import numpy as np
>>> from ceresfit import LinReg

>>>  # some data
>>> xdata = np.array([1, 2, 3.1, 4.9])
>>> ydata = np.array([1.1, 1.9, 3, 5.5])

>>>  # some uncertainty and correlation
>>> xunc = 0.05 * xdata
>>> yunc = 0.073 * ydata
>>> rho = np.zeros_like(xdata) + 0.5

>>>  # do regression
>>> my_reg = LinReg(xdata, xunc, ydata, yunc, rho)

>>>  # print out the parameters and their uncertainties
>>> my_reg.slope
(0.9983613298400896, 0.06844666435449052)
>>> my_reg.intercept
(0.05545398718611372, 0.11812746374874884)
>>> my_reg.mswd
2.5105964767071143

```

Detailed example on how to use the class for fitting and plotting the results
can be found
[in these Jupyter notebooks](https://github.com/galactic-forensics/CEREsFit/tree/main/examples).


## Development & Contributing

If you would like to contribute,
clone the GitHub repo and then install the package locally from within the folder via:

```
pip install -e .[dev]
```

If you also want to install the full test environment,
run:

```
pip install -e .[dev,test]
```

Code auto formatting is implemented using
[`pre-commit`](https://pre-commit.com/) hooks.
Full testing of the package can be done with
[`nox`](https://nox.thea.codes/en/stable/index.html).

Please feel free to raise issues on GitHub
and open pull requests if you have a feature to be added.
Tests and adequate docstrings should be provided along with your new code.
