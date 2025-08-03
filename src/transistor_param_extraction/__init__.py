"""
Deep Learning for Transistor Parameter Extraction

This package provides neural network models and utilities for automated
parameter extraction and model fitting of two-dimensional transistors.

Author: R.K.A. Bennett, J.L. Uslu, et al.
Paper: "Deep Learning to Automate Parameter Extraction and Model Fitting 
       of Two-Dimensional Transistors"
arXiv: 2507.05134
"""

__version__ = "0.1.0"
__author__ = "R.K.A. Bennett, J.L. Uslu"
__email__ = "rkabenne@stanford.edu"

# Import legacy modules for backward compatibility
from . import NN_fns
from . import NN_variables

# Import new modular structure
from . import training
from . import losses
from . import data_processing
from . import data_augmentation
from . import evaluation
from . import scaling
from . import utils

__all__ = [
    "NN_fns",
    "NN_variables",  # Legacy modules
    "training",
    "losses",
    "data_processing",
    "data_augmentation",
    "evaluation",
    "scaling",
    "utils",  # New modular structure
]
