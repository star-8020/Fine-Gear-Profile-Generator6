import unittest
import numpy as np
import sys
import os

# Add the project root to the Python path to allow for absolute imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from fine_gear_profile_generator.core import geometry_generator

class TestGeometryGeneration(unittest.TestCase):

    def setUp(self):
        """Set up common parameters for a standard gear."""
        self.test_params = {
            'M': 1.0,
            'Z': 20,
            'ALPHA': 20.0,
            'X': 0.0,
            'B': 0.0,
            'A': 1.0,
            'D': 1.25,
            'C': 0.25,
            'E': 0.1,
            'SEG_INVOLUTE': 20,
            'SEG_EDGE_R': 10,
            'SEG_ROOT_R': 10,
            'SEG_OUTER': 5,
            'SEG_ROOT': 5
        }

    def test_generate_single_tooth_profile_runs_successfully(self):
        """
        Tests that the main geometry generation function runs without raising exceptions
        and returns valid data structures.
        """
        try:
            X_tooth, Y_tooth, Z_calc, P_ANGLE, ALIGN_ANGLE = geometry_generator.generate_tooth_profile(**self.test_params)

            # Check if the outputs are of the correct type
            self.assertIsInstance(X_tooth, np.ndarray)
            self.assertIsInstance(Y_tooth, np.ndarray)
            self.assertIsInstance(Z_calc, (int, float))
            self.assertIsInstance(P_ANGLE, float)
            self.assertIsInstance(ALIGN_ANGLE, float)

        except Exception as e:
            self.fail(f"generate_tooth_profile() raised an exception unexpectedly: {e}")

    def test_generated_coordinates_are_valid(self):
        """
        Tests that the generated coordinates are finite and non-empty, which is a
        proxy for checking spline smoothness and avoiding mathematical errors.
        """
        X_tooth, Y_tooth, _, _, _ = geometry_generator.generate_tooth_profile(**self.test_params)

        # Check that the arrays are not empty
        self.assertTrue(X_tooth.size > 0)
        self.assertTrue(Y_tooth.size > 0)

        # Check that the arrays have the same length
        self.assertEqual(X_tooth.shape, Y_tooth.shape)

        # Check for any NaN (Not a Number) or Inf (Infinity) values
        self.assertTrue(np.all(np.isfinite(X_tooth)), "X coordinates contain NaN or Inf values.")
        self.assertTrue(np.all(np.isfinite(Y_tooth)), "Y coordinates contain NaN or Inf values.")

if __name__ == '__main__':
    unittest.main()