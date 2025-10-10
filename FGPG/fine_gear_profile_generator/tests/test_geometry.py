import unittest
import numpy as np

from fine_gear_profile_generator.core import geometry_generator
from fine_gear_profile_generator.core import models

class TestGeometryGeneration(unittest.TestCase):
    def setUp(self):
        """Set up common parameters for geometry tests."""
        self.test_params = {
            'M': 1.5,
            'Z': 20,
            'ALPHA': 20.0,
            'X': 0.5,
            'B': 0.1,
            'A': 1.0,
            'D': 1.25,
            'C': 0.2,
            'E': 0.1,
            'SEG_INVOLUTE': 20,
            'SEG_EDGE_R': 10,
            'SEG_ROOT_R': 10,
            'SEG_OUTER': 10,
            'SEG_ROOT': 10,
        }

    def test_generate_single_tooth_profile_runs_successfully(self):
        """
        Tests that the main geometry generation function runs without raising exceptions
        and returns valid data structures.
        """
        try:
            profile, Z_calc, P_ANGLE, ALIGN_ANGLE = geometry_generator.generate_tooth_profile(**self.test_params)

            # Check if the outputs are of the correct type
            self.assertIsInstance(profile, models.StructuredToothProfile)
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
        profile, _, _, _ = geometry_generator.generate_tooth_profile(**self.test_params)

        all_vertices = []
        for element in profile.elements:
            if isinstance(element, models.Spline):
                all_vertices.extend(element.vertices)
            elif isinstance(element, models.Arc):
                self.assertTrue(np.all(np.isfinite(element.center)))
                self.assertTrue(np.isfinite(element.radius))
                self.assertTrue(np.isfinite(element.start_angle))
                self.assertTrue(np.isfinite(element.end_angle))

        # Check that we have some vertices from the splines
        self.assertTrue(len(all_vertices) > 0)

        # Check for finite values in spline vertices
        all_vertices_arr = np.array(all_vertices)
        self.assertTrue(np.all(np.isfinite(all_vertices_arr)))

if __name__ == '__main__':
    unittest.main()