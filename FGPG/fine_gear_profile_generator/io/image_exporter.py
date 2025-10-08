"""Utilities for exporting gear pair previews as images."""

from __future__ import annotations

import os
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np

from ..core import transformations

GearPlotTuple = Tuple[np.ndarray, np.ndarray, int, float, float]


def export_gear_pair_to_image(
    working_dir: str,
    gear1_data: GearPlotTuple,
    gear2_data: GearPlotTuple,
    center_dist: float,
    module_value: float,
    gear1_teeth: int,
    gear2_teeth: int,
    x_offset: float = 0.0,
    y_offset: float = 0.0,
) -> None:
    """Generate and save a PNG preview for the supplied gear pair."""

    if 'DISPLAY' not in os.environ and 'XDG_SESSION_TYPE' not in os.environ:
        plt.switch_backend('Agg')

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    ax.set_aspect('equal')
    ax.set_title('Fine Gear Profile Generator - Gear Pair Preview')
    ax.grid(True)

    x_tooth1, y_tooth1, z1, pitch_angle1, alignment_angle1 = gear1_data
    x_rot1, y_rot1 = transformations.rotate(x_tooth1, y_tooth1, alignment_angle1)
    for i in range(int(z1)):
        x_temp, y_temp = transformations.rotate(x_rot1, y_rot1, pitch_angle1 * i)
        x_final, y_final = transformations.translate(x_temp, y_temp, x_offset, y_offset)
        ax.plot(x_final, y_final, '-', linewidth=1.5, color='blue')

    x_tooth2, y_tooth2, z2, pitch_angle2, alignment_angle2 = gear2_data
    initial_rotation2 = np.pi + (np.pi / z2)
    x_rot2, y_rot2 = transformations.rotate(x_tooth2, y_tooth2, alignment_angle2 + initial_rotation2)
    for i in range(int(z2)):
        x_temp, y_temp = transformations.rotate(x_rot2, y_rot2, pitch_angle2 * i)
        x_final, y_final = transformations.translate(x_temp, y_temp, x_offset + center_dist, y_offset)
        ax.plot(x_final, y_final, '-', linewidth=1.5, color='red')

    ax.set_xlim(-module_value * gear1_teeth / 1.5, center_dist + module_value * gear2_teeth / 1.5)
    max_teeth = max(gear1_teeth, gear2_teeth)
    ax.set_ylim(-module_value * max_teeth * 1.2, module_value * max_teeth * 1.2)

    output_path = os.path.join(working_dir, 'Result1.png')
    try:
        fig.savefig(output_path, dpi=100)
    except OSError as error:  # pragma: no cover - filesystem errors
        print(f"Error saving image: {error}")
    finally:
        plt.close(fig)