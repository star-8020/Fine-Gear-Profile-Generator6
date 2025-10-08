"""DXF export utilities for the Fine Gear Profile Generator."""

from __future__ import annotations

import math
import os

import ezdxf
import numpy as np

from ..core.models import ArcSegment, GearProfileGeometry, SplineSegment, ToothProfileData


def _rotation_matrix(angle: float) -> np.ndarray:
    cos_angle = math.cos(angle)
    sin_angle = math.sin(angle)
    return np.array([[cos_angle, -sin_angle], [sin_angle, cos_angle]])


def _rotate_points(points: np.ndarray, angle: float) -> np.ndarray:
    return points @ _rotation_matrix(angle).T


def _transform_spline(
    segment: SplineSegment,
    rotation: float,
    translation: np.ndarray,
) -> np.ndarray:
    rotated = _rotate_points(segment.points, rotation)
    return rotated + translation


def _normalize_arc_angles(start: float, end: float) -> tuple[float, float]:
    sweep = end - start
    if sweep <= 0:
        sweep += 2 * math.pi
    start_deg = math.degrees(start) % 360.0
    sweep_deg = math.degrees(sweep)
    return start_deg, start_deg + sweep_deg


def _transform_arc(
    segment: ArcSegment,
    rotation: float,
    translation: np.ndarray,
) -> tuple[tuple[float, float], float, float]:
    center = _rotate_points(np.array([[segment.center[0], segment.center[1]]]), rotation)[0]
    center += translation
    start_angle = segment.start_angle + rotation
    end_angle = segment.end_angle + rotation
    start_deg, end_deg = _normalize_arc_angles(start_angle, end_angle)
    return (float(center[0]), float(center[1])), start_deg, end_deg


def _draw_spline(msp, points: np.ndarray, color: int) -> None:
    msp.add_spline(fit_points=[(float(x), float(y), 0.0) for x, y in points], dxfattribs={'color': color})


def _draw_arc(msp, center: tuple[float, float], radius: float, start_deg: float, end_deg: float, color: int) -> None:
    msp.add_arc(center=center, radius=radius, start_angle=start_deg, end_angle=end_deg, dxfattribs={'color': color})


def _draw_tooth(
    msp,
    profile: ToothProfileData,
    rotation: float,
    translation: np.ndarray,
    color: int,
) -> None:
    for element in profile.elements_in_order():
        if isinstance(element, ArcSegment):
            center, start_deg, end_deg = _transform_arc(element, rotation, translation)
            _draw_arc(msp, center, element.radius, start_deg, end_deg, color)
        elif isinstance(element, SplineSegment):
            points = _transform_spline(element, rotation, translation)
            _draw_spline(msp, points, color)


def _draw_gear(
    msp,
    gear: GearProfileGeometry,
    base_rotation: float,
    translation: np.ndarray,
    color: int,
) -> None:
    for i in range(int(gear.teeth)):
        tooth_rotation = base_rotation + gear.pitch_angle * i
        _draw_tooth(msp, gear.profile, tooth_rotation, translation, color)


def _initial_rotation_for_meshing(gear1: GearProfileGeometry, gear2: GearProfileGeometry) -> float:
    return math.pi + 0.5 * (gear1.pitch_angle + gear2.pitch_angle)


def export_gear_pair_to_dxf(
    working_dir: str,
    gear1: GearProfileGeometry,
    gear2: GearProfileGeometry,
    center_dist: float,
    x_offset: float,
    y_offset: float,
) -> None:
    """Export the supplied gear pair geometry to a DXF file."""

    doc = ezdxf.new('R2000')
    msp = doc.modelspace()

    offset1 = np.array([x_offset, y_offset])
    offset2 = np.array([x_offset + center_dist, y_offset])

    _draw_gear(msp, gear1, gear1.alignment_angle, offset1, color=5)

    rotation2 = gear2.alignment_angle + _initial_rotation_for_meshing(gear1, gear2)
    _draw_gear(msp, gear2, rotation2, offset2, color=1)

    output_path = os.path.join(working_dir, 'Result_Gear_Pair.dxf')
    try:
        doc.saveas(output_path)
    except IOError:  # pragma: no cover - filesystem errors
        print(f"Error: Could not save DXF file to {output_path}.")