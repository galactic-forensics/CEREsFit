"""Test uncertainty calculations."""

import numpy as np
import pytest

from curefit import unc_calc as uc


def test_error_bar_angle_length_single():
    """Calculate error bar angle and length for given correlation (single value)."""
    sigx = 1
    sigy = 2
    rho = 0.75

    angle_exp = np.array([np.deg2rad(67.5), np.deg2rad(-22.5)])
    length_exp = np.array([np.sqrt(2.5 + np.sqrt(4.5)), np.sqrt(2.5 - np.sqrt(4.5))])

    angle_rec, length_rec = uc.error_bar_angle_length(sigx, sigy, rho)

    assert angle_exp == pytest.approx(angle_rec)
    assert length_exp == pytest.approx(length_rec)


@pytest.mark.parametrize("rho", [0.75, np.array([0.75, 0.75])])
def test_error_bar_angle_length_multi(rho):
    """Calculate error bar angle and length for given correlation (multiple values)."""
    sigx = np.array([1, 2])
    sigy = np.array([2, 1])

    length_exp = np.array(
        [
            np.sqrt(2.5 + np.sqrt(4.5)),
            np.sqrt(2.5 - np.sqrt(4.5)),
            np.sqrt(2.5 + np.sqrt(4.5)),
            np.sqrt(2.5 - np.sqrt(4.5)),
        ]
    )

    sigxy = rho * sigx * sigy
    angle_exp = np.array(
        [
            np.arctan((length_exp[:2] ** 2 - sigx[0] ** 2) / sigxy[0]),
            np.arctan((length_exp[2:] ** 2 - sigx[1] ** 2) / sigxy[1]),
        ]
    ).reshape(4)

    angle_rec, length_rec = uc.error_bar_angle_length(sigx, sigy, rho)

    assert angle_exp == pytest.approx(angle_rec)
    assert length_exp == pytest.approx(length_rec)


def test_error_bar_position_single():
    """Test error bar position with Thomas' Macro value."""
    xdat = 49.5263059304119
    sigx = 116.127138038472
    ydat = 94.8320931109692
    sigy = 137.018340778937
    rho = 0.429886405472074

    xpos_exp = np.array(
        [[-37.0849520308128, 136.137563891636], [-27.8300266053324, 126.882638466156]]
    )
    ypos_exp = np.array(
        [[-31.507663087177, 221.171849309115], [147.863137920647, 41.8010483012917]]
    )

    xpos_rec, ypos_rec = uc.error_bar_positions(xdat, sigx, ydat, sigy, rho)

    assert xpos_exp == pytest.approx(xpos_rec)
    assert ypos_exp == pytest.approx(ypos_rec)


def test_error_bar_position_values():
    """Test error bar position with Thomas' Macro values."""
    xdat = np.array([49.5263059304119, -19.8909690562228])
    sigx = np.array([116.127138038472, 108.88204516431])
    ydat = np.array([94.8320931109692, -24.8886147157681])
    sigy = np.array([137.018340778937, 124.423571979121])
    rho = np.array([0.429886405472074, 0.421261303003762])

    xpos_exp = np.array(
        [
            [-37.0849520308128, 136.137563891636],
            [-27.8300266053324, 126.882638466156],
            [-102.671239953765, 62.8893018413192],
            [-90.620923879426, 50.8389857669805],
        ]
    )
    ypos_exp = np.array(
        [
            [-31.507663087177, 221.171849309115],
            [147.863137920647, 41.8010483012917],
            [-138.042347260128, 88.2651178285919],
            [26.8555451027364, -76.6327745342726],
        ]
    )

    xpos_rec, ypos_rec = uc.error_bar_positions(xdat, sigx, ydat, sigy, rho)

    assert xpos_exp == pytest.approx(xpos_rec)
    assert ypos_exp == pytest.approx(ypos_rec)


@pytest.mark.parametrize("val", [1, 13.42])
def test_make_np_iterable(val):
    """Turn an individual value into a np.ndarray of length 1."""
    val_rec = uc._make_np_iterable(val)

    assert isinstance(val_rec, np.ndarray)
    assert len(val_rec) == 1


@pytest.mark.parametrize("val", [[1, 2], (13.42, 15), [23]])
def test_make_np_iterable_list(val):
    """Turn a list / similar into a np.ndarray of length 1."""
    val_rec = uc._make_np_iterable(val)

    assert isinstance(val_rec, np.ndarray)
    assert len(val_rec) == len(val)


def test_make_np_iterable_type_error():
    """Raise a type error if a string is passed."""
    with pytest.raises(TypeError):
        _ = uc._make_np_iterable("s")
