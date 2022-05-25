"""Linear Regression according to Mahon (1996) - corrected."""

from typing import Tuple, Union
import warnings

import numpy as np
from numpy.polynomial import Polynomial


class LinReg:
    """Linear regression using (corrected) Mahon (1996) prescription.

    Todo: add DOI of the paper, once published.
    Todo: add Example of usage
    """

    def __init__(
        self,
        xdat: np.ndarray,
        sigx: np.ndarray,
        ydat: np.ndarray,
        sigy: np.ndarray,
        rho: Union[float, np.ndarray] = None,
        fixpt: np.ndarray = None,
        autocalc=True,
        **kwargs,
    ):
        """Initialize the class.

        :param xdat: X data.
        :param ydat: Y data.
        :param sigx: 1 sigma uncertainty of x data.
        :param sigy: 1 sigma uncertainty of y data.
        :param rho: Correlation between x and y data, defaults to no correlation.
        :param fixpt: Fixed point through which regression needs to go.
        :param autocalc: Automatically calculate the regression and print params.
        :param kwargs: Additional keyword arguments:
            "iter_max": Maximum iteration limit for slope (default 1e6)
            "reg_limit": Regression limit for slope (default: 1e-6)

        :raises ValueError: Fix point is of the wrong shape.
        """
        self.xdat = np.array(xdat)
        self.sigx = np.array(sigx)
        self.ydat = np.array(ydat)
        self.sigy = np.array(sigy)
        if rho is None:
            self.rho = np.zeros_like(self.xdat)
        else:
            self.rho = np.array(rho)

        # calculate correlated uncertainty sigxy
        self.sigxy = self.rho * self.sigx * self.sigy

        if fixpt is not None:
            fixpt = np.array(fixpt)
            if fixpt.shape != (2,):
                raise ValueError("Fix point must be of the form [x_fix, yfix].")
        self.fix_pt = fixpt

        # Initialize the slope and intercept
        self._slope = None
        self._slope_unc = None
        self._intercept = None
        self._intercept_unc = None
        self._chi_squared = None
        self._mswd = None

        # keyword arguments
        if "regression_limit" in kwargs:
            self.reg_limit = kwargs["regression_limit"]
        else:
            self.reg_limit = 1e-6
        if "iter_max" in kwargs:
            self.iter_max = kwargs["iter_max"]
        else:
            self.iter_max = 1e6

        # helper variables
        self.xbar = None
        self.ybar = None
        self.weights = None

        if autocalc:
            self.calculate()

    @property
    def chi_squared(self) -> float:
        """Return chi_squared of the regression."""
        return self._chi_squared

    @property
    def intercept(self) -> Tuple[float, float]:
        """Return intercept and its 1 sigma uncertainty."""
        return self._intercept, self._intercept_unc

    @property
    def mswd(self) -> float:
        """Return MSWD of the regression."""
        return self._mswd

    @property
    def parameters(self) -> np.ndarray:
        """Return all parameters of the linear regression.

        :return: slope, slope_uncertainty, intercept, intercept_uncertainty, MSWD
        """
        return np.array(
            [
                self._slope,
                self._slope_unc,
                self._intercept,
                self._intercept_unc,
                self._mswd,
            ]
        )

    @property
    def slope(self) -> Tuple[float, float]:
        """Return slope and its 1 sigma uncertainty."""
        return self._slope, self._slope_unc

    def calculate(self):
        """Do the linear regression and save the parameters in the class variables."""
        self.slope_initial_guess()
        self.slope_calculation()
        self.intercept_calculation()
        self.unc_calculation()
        self.goodness_of_fit()

    def goodness_of_fit(self):
        """Calculate goodness of fit parameters chi-squared and MSWD."""
        chi_sq = np.sum(
            self.weights * (self.ydat - self._slope * self.xdat - self._intercept) ** 2
        )
        self._chi_squared = chi_sq

        dof = len(self.xdat) - 2 if self.fix_pt is None else len(self.xdat) - 1
        self._mswd = chi_sq / dof

    def intercept_calculation(self):
        """Calculate the intercept."""
        self._intercept = self.ybar - self._slope * self.xbar

    def slope_calculation(self):
        """Iterate the slope until it fits."""

        def calc_weights(b: float):
            """Calculate weights and return them.

            :param b: Slope

            :return: Weights
            """
            return 1 / (self.sigy**2 + b**2 * self.sigx**2 - 2 * b * self.sigxy)

        def calc_xbar(weights: np.ndarray):
            """Calculate x bar and return it.

            :param weights: Weights.

            :return: X bar.
            """
            if self.fix_pt is None:
                return np.sum(weights * self.xdat) / np.sum(weights)
            else:
                return self.fix_pt[0]

        def calc_ybar(weights: np.ndarray):
            """Calculate y bar and return it.

            :param weights: Weights.

            :return: Y bar.
            """
            if self.fix_pt is None:
                return np.sum(weights * self.ydat) / np.sum(weights)
            else:
                return self.fix_pt[1]

        def iterate_b(b_old):
            """Do one iteration step with the slope and return the new value."""
            b = b_old
            weights = calc_weights(b)
            u_all = self.xdat - calc_xbar(weights)
            v_all = self.ydat - calc_ybar(weights)
            b_new = np.sum(
                weights**2
                * v_all
                * (
                    u_all * self.sigy**2
                    + b * v_all * self.sigx**2
                    - v_all * self.sigxy
                )
            ) / np.sum(
                weights**2
                * u_all
                * (
                    u_all * self.sigy**2
                    + b * v_all * self.sigx**2
                    - b * u_all * self.sigxy
                )
            )
            return b_new

        # iterate until solution is found
        iter_cnt = 0
        b_old = self._slope
        b_new = iterate_b(b_old)
        while np.abs(b_old - b_new) > self.reg_limit and iter_cnt <= self.iter_max:
            b_old = b_new
            b_new = iterate_b(b_old)
            iter_cnt += 1

        if iter_cnt == self.iter_max:
            warnings.warn(
                f"Iteration count for slope optimization hit the limt at "
                f"{self.iter_max}. The current difference between the old and new "
                f"slope is {np.abs(b_old - b_new)}"
            )

        self._slope = b_new
        self.weights = calc_weights(b_new)
        self.xbar = calc_xbar(self.weights)
        self.ybar = calc_ybar(self.weights)

    def slope_initial_guess(self):
        """Calculate an initial guess of the slope without uncertainties and save it."""
        polyfit = Polynomial.fit(self.xdat, self.ydat, deg=1)
        self._slope = polyfit.convert().coef[1]

    def unc_calculation(self):
        """Calculate uncertainties for slope and intercept with no fixed point."""
        # helper variables
        sigx = self.sigx
        sigy = self.sigy
        sigxy = self.sigxy
        b = self._slope
        weights = self.weights
        xbar = self.xbar
        ybar = self.ybar
        u_all = self.xdat - xbar
        v_all = self.ydat - ybar

        sum_weights = np.sum(weights)

        # d(theta) / db
        dthdb = np.sum(
            weights**2
            * (
                2 * b * (u_all * v_all * sigx**2 - u_all**2 * sigxy)
                + (u_all**2 * sigy**2 - v_all**2 * sigx**2)
            )
        ) + 4 * np.sum(
            weights**3
            * (sigxy - b * sigx**2)
            * (
                b**2 * (u_all * v_all * sigx**2 - u_all**2 * sigxy)
                + b * (u_all**2 * sigy**2 - v_all**2 * sigx**2)
                - (u_all * v_all * sigy**2 - v_all**2 * sigxy)
            )
        )

        def calc_dtheta_dxi(it: int):
            """Calculate partial derivative d(theta)/dxi.

            :param it: Index where the $i$ is at.

            :return: dtheta/dxi
            """
            if self.fix_pt is None:
                sum_all = 0.0
                for jt, wj in enumerate(weights):
                    kron = kron_delta(it, jt)
                    sum_all += (
                        wj**2
                        * (kron - weights[it] / sum_weights)
                        * (
                            b**2 * v_all[jt] * sigx[jt] ** 2
                            - b**2 * 2 * u_all[jt] * sigxy[jt]
                            + 2 * b * u_all[jt] * sigy[jt] ** 2
                            - v_all[jt] * sigy[jt] ** 2
                        )
                    )
                return sum_all
            else:
                return weights[it] ** 2 * (
                    b**2 * v_all[it] * sigx[it] ** 2
                    - b**2 * 2 * u_all[it] * sigxy[it]
                    + 2 * b * u_all[it] * sigy[it] ** 2
                    - v_all[it] * sigy[it] ** 2
                )

        def calc_dtheta_dyi(it: int):
            """Calculate partial derivative d(theta)/dyi.

            :param it: Index where the $i$ is at.

            :return: dtheta/dyi
            """
            if self.fix_pt is None:
                sum_all = 0.0
                for jt, wj in enumerate(weights):
                    kron = kron_delta(it, jt)
                    sum_all += (
                        wj**2
                        * (kron - weights[it] / sum_weights)
                        * (
                            b**2 * u_all[jt] * sigx[jt] ** 2
                            - 2 * b * v_all[jt] * sigx[jt] ** 2
                            - u_all[jt] * sigy[jt] ** 2
                            + 2 * v_all[jt] * sigxy[jt]
                        )
                    )
                return sum_all
            else:
                return weights[it] ** 2 * (
                    b**2 * u_all[it] * sigx[it] ** 2
                    - 2 * b * v_all[it] * sigx[it] ** 2
                    - u_all[it] * sigy[it] ** 2
                    + 2 * v_all[it] * sigxy[it]
                )

        def calc_da_dxi(it: int):
            """Calculate partial derivative da/dxi.

            :param it: Index where the $i$ is at.

            :return: da/dxi
            """
            if self.fix_pt is None:
                return (
                    -b * weights[it] / sum_weights - xbar * calc_dtheta_dxi(it) / dthdb
                )
            else:
                return -xbar * calc_dtheta_dxi(it) / dthdb

        def calc_da_dyi(it: int):
            """Calculate partial derivative da/dyi.

            :param it: Index where the $i$ is at.

            :return: da/dyi
            """
            if self.fix_pt is None:
                return weights[it] / sum_weights - xbar * calc_dtheta_dyi(it) / dthdb
            else:
                return -xbar * calc_dtheta_dyi(it) / dthdb

        # calculate uncertainty for slope
        sigb_sq = 0.0
        for it, sigxi in enumerate(sigx):
            sigyi = sigy[it]
            sigxyi = sigxy[it]
            dtheta_dxi = calc_dtheta_dxi(it)
            dtheta_dyi = calc_dtheta_dyi(it)
            sigb_sq += (
                dtheta_dxi**2 * sigxi**2
                + dtheta_dyi**2 * sigyi**2
                + 2 * sigxyi * dtheta_dxi * dtheta_dyi
            )
        sigb_sq /= dthdb**2
        self._slope_unc = np.sqrt(sigb_sq)

        if self.fix_pt is None:
            siga_sq = 0.0
            for it, sigxi in enumerate(sigx):
                sigyi = sigy[it]
                sigxyi = sigxy[it]
                da_dxi = calc_da_dxi(it)
                da_dyi = calc_da_dyi(it)
                siga_sq += (
                    da_dxi**2 * sigxi**2
                    + da_dyi**2 * sigyi**2
                    + 2 * sigxyi * da_dxi * da_dyi
                )
        else:
            siga_sq = self.fix_pt[0] ** 2 * sigb_sq
        self._intercept_unc = np.sqrt(siga_sq)


def kron_delta(
    ind1: Union[int, np.ndarray], ind2: Union[int, np.ndarray]
) -> Union[int, np.ndarray]:
    """Calculate Kronecker-delta for variables i,j.

    Compare two indexes and return 0 if the same, otherwise 1. If an ndarray is given,
    return an ndarray comparing each index individually.

    :param ind1: Index(es)
    :param ind2: Index(es)

    :return: 1 if ind 1 is identical to ind2, otherwise 0

    :raises ValueError: The input indexes have different shape.
    """
    if np.shape(ind1) != np.shape(ind2):
        raise ValueError("The inputs must have the same shape.")

    if np.shape(ind1) == ():  # don't have arrays
        return 1 if ind1 == ind2 else 0
    else:
        ret_arr = np.zeros_like(ind1)
        ret_arr[np.where(ind1 == ind2)] = 1
        return ret_arr
