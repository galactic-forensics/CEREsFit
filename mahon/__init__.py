"""Linear regression of data sets with correlated and uncorrelated uncertainties."""

from .mahon_linreg import LinReg

# Package information
__version__ = "0.1.0"
__all__ = ["LinReg"]

__title__ = "mahon"
__description__ = (
    "Linear regression of data sets with correlated and uncorrelated uncertainties."
    "Methodology follows the work of Mahon (1996) with correction for errors. For "
    "details, see the readme file on https://github.com/galactic-forensics/mahon."
)

__uri__ = "https://github.com/galactic-forensics/mahon"
__author__ = "Reto Trappitsch"
__email__ = "reto@galactic-forensics.space"

__license__ = "MIT"
__copyright__ = "Copyright (c) 2022, Reto Trappitsch"
