"""Test confidence interval calculation."""

import numpy as np
import pytest
import scipy.stats as stats

from curefit import LinReg


# STEPHAN TEST DATASET FROM MACRO
def test_linreg_ci_calc(stephan_ci_data):
    """Compare CI calculation with Stephan Macro."""
    p_conf = 0.95
    (
        xdat,
        sigx,
        ydat,
        sigy,
        rho,
        x_ci_exp,
        y_ci_min_st,
        y_ci_max_st,
    ) = stephan_ci_data

    my_reg = LinReg(xdat, sigx, ydat, sigy, rho, autocalc=True)

    # Stephan uses 1 sigma error bands: mulitply with student t factor
    y_ci = x_ci_exp * my_reg.slope[0] + my_reg.intercept[0]
    zfac = stats.t.ppf(1 - (1 - p_conf) / 2.0, len(xdat) - 2)
    y_ci_min_exp = y_ci - np.abs(y_ci - y_ci_min_st) * zfac
    y_ci_max_exp = y_ci + np.abs(y_ci - y_ci_max_st) * zfac

    xrange = np.array([x_ci_exp.min(), x_ci_exp.max()])
    x_ci_rec, y_ci_min_rec, y_ci_max_rec = my_reg.confidence_intervals(
        p_conf=p_conf, xrange=xrange, bins=len(x_ci_exp)
    )

    assert x_ci_exp == pytest.approx(x_ci_rec)
    assert y_ci_min_exp == pytest.approx(y_ci_min_rec, rel=1e-4)
    assert y_ci_max_exp == pytest.approx(y_ci_max_rec, rel=1e-4)


def test_linreg_uncertainty_band_calc(stephan_ci_data):
    """Compare uncertainty band calculation with Stephan Macro."""
    (
        xdat,
        sigx,
        ydat,
        sigy,
        rho,
        x_ub_exp,
        y_ub_min_exp,
        y_ub_max_exp,
    ) = stephan_ci_data

    my_reg = LinReg(xdat, sigx, ydat, sigy, rho, autocalc=False)

    xrange = np.array([x_ub_exp.min(), x_ub_exp.max()])
    x_ub_rec, y_ub_min_rec, y_ub_max_rec = my_reg.uncertainty_band(
        xrange=xrange, bins=len(x_ub_exp)
    )

    assert x_ub_exp == pytest.approx(x_ub_rec)
    assert y_ub_min_exp == pytest.approx(y_ub_min_rec, rel=1e-4)
    assert y_ub_max_exp == pytest.approx(y_ub_max_rec, rel=1e-4)


def test_linreg_uncertainty_band_sigma(stephan_ci_data):
    """Ensure that 2 sigma band is twice as wide than 1 sigma band."""
    xdat, sigx, ydat, sigy, rho, _, _, _ = stephan_ci_data

    my_reg = LinReg(xdat, sigx, ydat, sigy, rho, autocalc=False)

    x_ub1, y_ub1_min, y_ub1_max = my_reg.uncertainty_band(sigma=1)
    x_ub2, y_ub2_min, y_ub2_max = my_reg.uncertainty_band(sigma=2)

    band_1sig = np.abs(y_ub1_max - y_ub1_min)
    band_2sig = np.abs(y_ub2_max - y_ub2_min)

    assert 2 * band_1sig == pytest.approx(band_2sig)
