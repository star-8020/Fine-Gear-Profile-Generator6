"""DXF export utilities for the Fine Gear Profile Generator."""

from __future__ import annotations

import os
from typing import Tuple

import ezdxf
import numpy as np

from ..core import transformations

GearPlotTuple = Tuple[np.ndarray, np.ndarray, int, float, float]


def export_gear_pair_to_dxf(
    working_dir: str,
    gear1_data: GearPlotTuple,
    gear2_data: GearPlotTuple,
    center_dist: float,
    x_offset: float,
    y_offset: float,
) -> None:
    """Export the supplied gear pair geometry to a DXF file."""

    doc = ezdxf.new('R2000')
    msp = doc.modelspace()

    x_tooth1, y_tooth1, z1, pitch_angle1, alignment_angle1 = gear1_data
    x_rot1, y_rot1 = transformations.rotate(x_tooth1, y_tooth1, alignment_angle1)
    for i in range(int(z1)):
        x_temp, y_temp = transformations.rotate(x_rot1, y_rot1, pitch_angle1 * i)
        x_final, y_final = transformations.translate(x_temp, y_temp, x_offset, y_offset)
        msp.add_lwpolyline(list(zip(x_final, y_final)), close=True, dxfattribs={'color': 5})

    x_tooth2, y_tooth2, z2, pitch_angle2, alignment_angle2 = gear2_data
    initial_rotation2 = np.pi + (np.pi / z2)
    x_rot2, y_rot2 = transformations.rotate(x_tooth2, y_tooth2, alignment_angle2 + initial_rotation2)
    for i in range(int(z2)):
        x_temp, y_temp = transformations.rotate(x_rot2, y_rot2, pitch_angle2 * i)
        x_final, y_final = transformations.translate(x_temp, y_temp, x_offset + center_dist, y_offset)
        msp.add_lwpolyline(list(zip(x_final, y_final)), close=True, dxfattribs={'color': 1})

    output_path = os.path.join(working_dir, 'Result_Gear_Pair.dxf')
    try:
        doc.saveas(output_path)
    except IOError:  # pragma: no cover - filesystem errors
        print(f"Error: Could not save DXF file to {output_path}.")