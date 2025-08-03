import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from transistor_param_extraction import NN_fns


class TestBasicFunctionality:
    """Basic tests for core functionality"""

    def test_imports(self):
        """Test that core modules can be imported"""
        from transistor_param_extraction import NN_fns, NN_variables

        assert NN_fns is not None
        assert NN_variables is not None

    def test_numpy_arrays(self):
        """Test basic numpy operations work"""
        arr = np.array([1, 2, 3, 4, 5])
        assert len(arr) == 5
        assert arr.mean() == 3.0

    @pytest.mark.skipif(
        not os.path.exists("data/processed"),
        reason="Processed data directory not found",
    )
    def test_data_directory_exists(self):
        """Test that data directories exist"""
        assert os.path.exists("data")
        assert os.path.exists("data/processed")

    @pytest.mark.skipif(
        not os.path.exists("models/trained"), reason="Models directory not found"
    )
    def test_models_directory_exists(self):
        """Test that models directory exists"""
        assert os.path.exists("models")
        assert os.path.exists("models/trained")
