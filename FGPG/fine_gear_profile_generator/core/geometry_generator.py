"""Generate tooth geometry for spur gears."""

from __future__ import annotations

from typing import List, Tuple, Union

import numpy as np

from . import gear_math, transformations
from .models import Arc, Spline, StructuredToothProfile


def involute_curve(
    module: float,
    teeth: int,
    segment_count: int,
    theta_start: float,
    theta_end: float,
    base_angle: float,
    start_angle: float,
) -> List[Tuple[float, float]]:
    """Generate the involute flank curve."""

    theta_values = np.linspace(theta_start, theta_end, segment_count)
    radius = 0.5 * module * teeth * np.cos(base_angle) * np.sqrt(1 + theta_values**2)
    x_coords = radius * np.cos(start_angle + theta_values - np.arctan(theta_values))
    y_coords = radius * np.sin(start_angle + theta_values - np.arctan(theta_values))
    return list(zip(x_coords, y_coords))


def edge_round_curve(
    module: float,
    tooth_edge_radius: float,
    flank_vertices: List[Tuple[float, float]],
    edge_x: float,
    edge_y: float,
    edge_center_x: float,
    edge_center_y: float,
    segment_count: int,
) -> List[Tuple[float, float]]:
    """Generate the rounded edge curve at the tooth tip."""

    last_flank_vertex = flank_vertices[-1]
    theta_min = np.arctan2(last_flank_vertex[1] - edge_center_y, last_flank_vertex[0] - edge_center_x)
    theta_max = np.arctan2(edge_y - edge_center_y, edge_x - edge_center_x)
    theta_values = np.linspace(theta_min, theta_max, segment_count)
    x_coords = module * tooth_edge_radius * np.cos(theta_values) + edge_center_x
    y_coords = module * tooth_edge_radius * np.sin(theta_values) + edge_center_y
    return list(zip(x_coords, y_coords))


def root_round_curve(
    module: float,
    teeth: int,
    profile_shift: float,
    dedendum_factor: float,
    hob_edge_radius: float,
    backlash_factor: float,
    theta_end: float,
    alpha_transition: float,
    segment_count: int,
) -> List[Tuple[float, float]]:
    """Generate the trochoidal root fillet curve."""

    theta_values = np.linspace(0, theta_end, segment_count)
    denominator = module * (dedendum_factor - profile_shift - hob_edge_radius)
    if hob_edge_radius != 0 and denominator == 0:
        theta_s = (np.pi / 2) * np.ones(len(theta_values))
    elif denominator != 0:
        theta_s = np.arctan((module * teeth * theta_values / 2) / denominator)
    else:
        theta_s = np.zeros(len(theta_values))
    x_coords = module * (
        (teeth / 2 + profile_shift - dedendum_factor + hob_edge_radius)
        * np.cos(theta_values + alpha_transition)
        + (teeth / 2)
        * theta_values
        * np.sin(theta_values + alpha_transition)
        - hob_edge_radius * np.cos(theta_s + theta_values + alpha_transition)
    )
    y_coords = module * (
        (teeth / 2 + profile_shift - dedendum_factor + hob_edge_radius)
        * np.sin(theta_values + alpha_transition)
        - (teeth / 2)
        * theta_values
        * np.cos(theta_values + alpha_transition)
        - hob_edge_radius * np.sin(theta_s + theta_values + alpha_transition)
    )
    return list(zip(x_coords, y_coords))


def outer_arc(
    module: float,
    teeth: int,
    profile_shift: float,
    addendum_factor: float,
    edge_angle: float,
    mid_angle: float,
) -> Arc:
    """Generate the outer arc at the tooth tip (addendum circle)."""

    radius = module * (teeth / 2 + addendum_factor + profile_shift)
    return Arc(center=(0, 0), radius=radius, start_angle=edge_angle, end_angle=mid_angle)


def root_arc(
    module: float,
    teeth: int,
    profile_shift: float,
    dedendum_factor: float,
    transition_angle: float,
) -> Arc:
    """Generate the root arc at the bottom of the tooth space (dedendum circle)."""

    radius = module * (teeth / 2 - dedendum_factor + profile_shift)
    return Arc(center=(0, 0), radius=radius, start_angle=0, end_angle=transition_angle)


def _generate_tooth_profile_impl(
    module: float,
    teeth: int,
    pressure_angle_deg: float,
    profile_shift: float,
    backlash_factor: float,
    addendum_factor: float,
    dedendum_factor: float,
    hob_edge_radius: float,
    tooth_edge_radius: float,
    segments_involute: int,
    segments_edge: int,
    segments_root_round: int,
    use_structured_output: bool = True,
) -> Union[
    Tuple[np.ndarray, np.ndarray, float, float, float],
    Tuple[StructuredToothProfile, float, float, float],
]:
    """Generate a single gear tooth profile with associated metadata."""

    (teeth_calc, shift_calc, backlash_calc, addendum_calc, dedendum_calc,
     hob_edge_calc, tooth_edge_calc) = gear_math.handle_internal_gear_parameters(
        teeth, profile_shift, backlash_factor, addendum_factor, dedendum_factor, hob_edge_radius, tooth_edge_radius
    )

    (
        base_angle,
        mid_angle,
        involute_start_angle,
        involute_theta_start,
        involute_theta_end,
        edge_angle,
        tooth_edge_calc,
        pitch_angle,
        alignment_angle,
    ) = gear_math.calculate_gear_parameters(
        module,
        teeth_calc,
        pressure_angle_deg,
        shift_calc,
        backlash_calc,
        addendum_calc,
        dedendum_calc,
        hob_edge_calc,
        tooth_edge_calc,
    )

    flank1_vertices = involute_curve(
        module,
        teeth_calc,
        segments_involute,
        involute_theta_start,
        involute_theta_end,
        base_angle,
        involute_start_angle,
    )
    flank2_vertices = transformations.reflect_y_vertices(flank1_vertices)

    edge_x = module * ((teeth_calc / 2) + shift_calc + addendum_calc) * np.cos(edge_angle)
    edge_y = module * ((teeth_calc / 2) + shift_calc + addendum_calc) * np.sin(edge_angle)
    edge_center_x = module * (teeth_calc / 2 + shift_calc + addendum_calc - tooth_edge_calc) * np.cos(edge_angle)
    edge_center_y = module * (teeth_calc / 2 + shift_calc + addendum_calc - tooth_edge_calc) * np.sin(edge_angle)

    edge1_vertices = edge_round_curve(
        module,
        tooth_edge_calc,
        flank1_vertices,
        edge_x,
        edge_y,
        edge_center_x,
        edge_center_y,
        segments_edge,
    )
    edge2_vertices = transformations.reflect_y_vertices(edge1_vertices)

    alpha_transition = (
        (2 * (hob_edge_calc * (1 - np.sin(base_angle)) - dedendum_calc) * np.sin(base_angle) + backlash_calc)
        / (teeth_calc * np.cos(base_angle))
        - 2 * hob_edge_calc * np.cos(base_angle) / teeth_calc
        + np.pi / (2 * teeth_calc)
    )
    theta_end = (
        2 * hob_edge_calc * np.cos(base_angle) / teeth_calc
        - 2 * (dedendum_calc - shift_calc - hob_edge_calc * (1 - np.sin(base_angle)))
        * np.cos(base_angle)
        / (teeth_calc * np.sin(base_angle))
    )

    root1_vertices = root_round_curve(
        module,
        teeth_calc,
        shift_calc,
        dedendum_calc,
        hob_edge_calc,
        backlash_calc,
        theta_end,
        alpha_transition,
        segments_root_round,
    )
    root2_vertices = transformations.reflect_y_vertices(root1_vertices)

    outer1_arc = outer_arc(
        module,
        teeth_calc,
        shift_calc,
        addendum_calc,
        edge_angle,
        mid_angle,
    )
    outer2_arc = transformations.reflect_y_arc(outer1_arc)

    root1_arc = root_arc(
        module,
        teeth_calc,
        shift_calc,
        dedendum_calc,
        alpha_transition,
    )
    root2_arc = transformations.reflect_y_arc(root1_arc)

    structured_profile = StructuredToothProfile(elements=[
        outer2_arc,
        Spline(vertices=edge2_vertices),
        Spline(vertices=flank2_vertices),
        Spline(vertices=root2_vertices),
        root2_arc,
        root1_arc,
        Spline(vertices=root1_vertices),
        Spline(vertices=flank1_vertices),
        Spline(vertices=edge1_vertices),
        outer1_arc,
    ])

    return structured_profile, float(teeth_calc), float(pitch_angle), float(alignment_angle)


def generate_tooth_profile(*args, **kwargs):
    """Public wrapper supporting both legacy kwargs and new positional arguments."""

    if kwargs:
        key_map = {
            'M': 'module',
            'Z': 'teeth',
            'ALPHA': 'pressure_angle_deg',
            'X': 'profile_shift',
            'B': 'backlash_factor',
            'A': 'addendum_factor',
            'D': 'dedendum_factor',
            'C': 'hob_edge_radius',
            'E': 'tooth_edge_radius',
            'SEG_INVOLUTE': 'segments_involute',
            'SEG_EDGE_R': 'segments_edge',
            'SEG_ROOT_R': 'segments_root_round',
        }
        normalized = {}
        for legacy_key, new_key in key_map.items():
            if legacy_key not in kwargs:
                raise TypeError(f"Missing required parameter '{legacy_key}' for tooth profile generation")
            normalized[new_key] = kwargs[legacy_key]
        return _generate_tooth_profile_impl(**normalized)

    expected_args = 12
    if len(args) != expected_args:
        raise TypeError(
            f"generate_tooth_profile() expects {expected_args} positional arguments"
        )

    return _generate_tooth_profile_impl(*args)
