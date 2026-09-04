"""Library for machine learning on physical tensors"""

import reax

from . import config, data, electric, keys, nmr_spectra, structure_search

__version__ = "0.1.3"

__all__ = (
    "data",
    "config",
    "electric",
    "keys",
    "nmr_spectra",
    "structure_search",
)
