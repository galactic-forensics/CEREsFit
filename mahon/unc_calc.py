"""Uncertainty calculations that are outside the linear regression routine."""

from typing import Any, Tuple, Union

import numpy as np


def error_bar_angle_length(
    sigx: Union[float, np.ndarray],
    sigy: Union[float, np.ndarray],
    rho: Union[float, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate the angles and length of a given error bar.

    For a given uncertainty, calculate the angle and length of the rotated error bar
    dependant on the uncertainties and correlation coefficient. Returned are two arrays,
    an angle array and a length array. Both of them will have twice as many entries
    as the original data arrays. For each line in the data arrays, two angles and
    two lengths will be returned for the given error bars.

    :param sigx: X uncertainty
    :param sigy: Y uncertainty
    :param rho: Correlation factor

    :return: angles (radian), lengths
    """
    sigx = _make_np_iterable(sigx)
    sigy = _make_np_iterable(sigy)
    rho = _make_np_iterable(rho)
    if rho.shape != sigx.shape:
        rho = np.zeros_like(sigx) + rho

    sigxy = rho * sigx * sigy

    # matrix calculations
    trace_a = sigx**2 + sigy**2
    det_a = sigx**2 * sigy**2 - sigxy**2

    # eigenvalues
    lambda_1 = trace_a / 2 + np.sqrt((trace_a / 2) ** 2 - det_a)
    lambda_2 = trace_a / 2 - np.sqrt((trace_a / 2) ** 2 - det_a)

    # slopes of error bars
    m1 = (lambda_1 - sigx**2) / sigxy
    m2 = (lambda_2 - sigx**2) / sigxy

    # angles
    alpha1 = np.arctan(m1)
    alpha2 = np.arctan(m2)

    # sort the return values
    alphas_ret = np.zeros(2 * len(sigx))
    lengths_ret = np.zeros(2 * len(sigx))

    for it in range(len(alpha1)):
        alphas_ret[2 * it] = alpha1[it]
        alphas_ret[2 * it + 1] = alpha2[it]
        lengths_ret[2 * it] = np.sqrt(lambda_1[it])
        lengths_ret[2 * it + 1] = np.sqrt(lambda_2[it])

    return alphas_ret, lengths_ret


def error_bar_positions(
    xdat: Union[float, np.ndarray],
    sigx: Union[float, np.ndarray],
    ydat: Union[float, np.ndarray],
    sigy: Union[float, np.ndarray],
    rho: Union[float, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate the positions of rotated error bars.

    This routine returns two arrays, both are twice as long as the initial data with two
    positions per line. Two arrays are returned, one with x positions and one with
    y positions. The first line in each array corresponds to the corresponding x
    and y coordinates between which a line must be drawn to represent an error bar
    for the first datapoint. The second entry represents the second error bar for the
    first entry, etc.

    :param xdat: X data
    :param sigx: X uncertainty
    :param ydat: Y data
    :param sigy: Y uncertainty
    :param rho: Correlation factor

    :return: x positions, y positions
    """
    angles, lengths = error_bar_angle_length(sigx, sigy, rho)

    # calculate deltas for x and y
    dx = np.cos(angles) * lengths
    dy = np.sin(angles) * lengths

    xret = np.zeros((len(dx), 2))
    yret = np.zeros_like(xret)

    # write the data point values into xret and yret
    if np.asarray(xdat).shape == ():
        xret += xdat
        yret += ydat
    else:
        for it in range(len(xdat)):
            xret[2 * it] += xdat[it]
            xret[2 * it + 1] += xdat[it]
            yret[2 * it] += ydat[it]
            yret[2 * it + 1] += ydat[it]

    # add / subtract the dx, dy values
    xret[:, 0] -= dx
    xret[:, 1] += dx
    yret[:, 0] -= dy
    yret[:, 1] += dy

    return xret, yret


def _make_np_iterable(value: Any) -> np.ndarray:
    """Turn a value into an np.ndarray.

    :param value: Value(s) to turn into an array.

    :return: Given value as a numpy array.

    :raises TypeError: Unsupported type for value.
    """
    if isinstance(value, str):
        raise TypeError(f"Unsupported type {type(value)}.")

    if hasattr(value, "__iter__"):
        return np.asarray(value)
    else:  # it's a single value -> turn into array of length 1
        return np.asarray(value).reshape(1)
