import unittest
import os
import shutil
import sys

# Add the project root to the Python path to allow for absolute imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from fine_gear_profile_generator.core import geometry_generator
from fine_gear_profile_generator.io import dxf_exporter

class TestDxfOutput(unittest.TestCase):

    def setUp(self):
        """Set up parameters and a temporary directory for test files."""
        self.temp_dir = "temp_test_output"
        os.makedirs(self.temp_dir, exist_ok=True)

        self.gear1_params = {
            'M': 1.0, 'Z': 18, 'ALPHA': 20.0, 'X': 0.2, 'B': 0.05,
            'A': 1.0, 'D': 1.25, 'C': 0.2, 'E': 0.1,
            'SEG_INVOLUTE': 15, 'SEG_EDGE_R': 15, 'SEG_ROOT_R': 15,
            'SEG_OUTER': 5, 'SEG_ROOT': 5
        }
        self.gear2_params = self.gear1_params.copy()
        self.gear2_params['Z'] = 36
        self.gear2_params['X'] = 0.0

    def tearDown(self):
        """Remove the temporary directory and its contents after the test."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_dxf_file_is_created(self):
        """
        Tests that the DXF exporter function runs and creates a non-empty file
        at the expected location.
        """
        # Generate the geometry for both gears
        gear1_data = geometry_generator.generate_tooth_profile(**self.gear1_params)
        gear2_data = geometry_generator.generate_tooth_profile(**self.gear2_params)

        # Define parameters for the exporter
        center_dist = 27.0  # A typical value for this gear pair
        x_offset = 0.0
        y_offset = 0.0

        # Call the DXF exporter
        dxf_exporter.export_gear_pair_to_dxf(
            self.temp_dir,
            gear1_data,
            gear2_data,
            center_dist,
            x_offset,
            y_offset
        )

        # Check if the file was created
        expected_filepath = os.path.join(self.temp_dir, 'Result_Gear_Pair.dxf')
        self.assertTrue(os.path.exists(expected_filepath), "DXF file was not created.")

        # Check if the file is not empty
        self.assertTrue(os.path.getsize(expected_filepath) > 0, "DXF file is empty.")

if __name__ == '__main__':
    unittest.main()