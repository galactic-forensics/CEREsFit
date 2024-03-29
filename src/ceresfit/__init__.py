"""Linear regression of data sets with correlated and uncorrelated uncertainties."""

from .ceresfit_linreg import LinReg

# Package information
__all__ = ["LinReg", "__version__"]

__title__ = "ceresfit"
__description__ = (
    "Linear regression of data sets with correlated and uncorrelated uncertainties."
    "Methodology follows the work of Mahon (1996) with correction for errors. For "
    "details, see the readme file on https://github.com/galactic-forensics/CEREsFit."
)

__uri__ = "https://github.com/galactic-forensics/CEREsFit"
__author__ = "Reto Trappitsch"
__email__ = "reto@galactic-forensics.space"

__license__ = "MIT"
__copyright__ = "Copyright (c) 2023-2024, Reto Trappitsch"
