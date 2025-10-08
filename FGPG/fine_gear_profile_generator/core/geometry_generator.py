"""Generate tooth geometry for spur gears."""

from __future__ import annotations

from typing import Tuple

import numpy as np

from . import gear_math
from . import transformations


def involute_curve(
    module: float,
    teeth: int,
    segment_count: int,
    theta_start: float,
    theta_end: float,
    base_angle: float,
    start_angle: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate the involute flank curve."""

    theta_values = np.linspace(theta_start, theta_end, segment_count)
    radius = 0.5 * module * teeth * np.cos(base_angle) * np.sqrt(1 + theta_values**2)
    x_coords = radius * np.cos(start_angle + theta_values - np.arctan(theta_values))
    y_coords = radius * np.sin(start_angle + theta_values - np.arctan(theta_values))
    return x_coords, y_coords


def edge_round_curve(
    module: float,
    tooth_edge_radius: float,
    flank_x: np.ndarray,
    flank_y: np.ndarray,
    edge_x: float,
    edge_y: float,
    edge_center_x: float,
    edge_center_y: float,
    segment_count: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate the rounded edge curve at the tooth tip."""

    theta_min = np.arctan2((flank_y[-1] - edge_center_y), (flank_x[-1] - edge_center_x))
    theta_max = np.arctan2((edge_y - edge_center_y), (edge_x - edge_center_x))
    theta_values = np.linspace(theta_min, theta_max, segment_count)
    x_coords = module * tooth_edge_radius * np.cos(theta_values) + edge_center_x
    y_coords = module * tooth_edge_radius * np.sin(theta_values) + edge_center_y
    return x_coords, y_coords


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
) -> Tuple[np.ndarray, np.ndarray]:
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
    return x_coords, y_coords


def outer_arc(
    module: float,
    teeth: int,
    profile_shift: float,
    addendum_factor: float,
    edge_angle: float,
    mid_angle: float,
    segment_count: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate the outer arc at the tooth tip (addendum circle)."""

    theta_values = np.linspace(edge_angle, mid_angle, segment_count)
    radius = module * (teeth / 2 + addendum_factor + profile_shift)
    x_coords = radius * np.cos(theta_values)
    y_coords = radius * np.sin(theta_values)
    return x_coords, y_coords


def root_arc(
    module: float,
    teeth: int,
    profile_shift: float,
    dedendum_factor: float,
    transition_angle: float,
    segment_count: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate the root arc at the bottom of the tooth space (dedendum circle)."""

    theta_values = np.linspace(0, transition_angle, segment_count)
    radius = module * (teeth / 2 - dedendum_factor + profile_shift)
    x_coords = radius * np.cos(theta_values)
    y_coords = radius * np.sin(theta_values)
    return x_coords, y_coords


def combine_tooth_profile(
    flank1_x: np.ndarray,
    flank1_y: np.ndarray,
    edge1_x: np.ndarray,
    edge1_y: np.ndarray,
    root1_x: np.ndarray,
    root1_y: np.ndarray,
    outer1_x: np.ndarray,
    outer1_y: np.ndarray,
    root_arc1_x: np.ndarray,
    root_arc1_y: np.ndarray,
    flank2_x: np.ndarray,
    flank2_y: np.ndarray,
    edge2_x: np.ndarray,
    edge2_y: np.ndarray,
    root2_x: np.ndarray,
    root2_y: np.ndarray,
    outer2_x: np.ndarray,
    outer2_y: np.ndarray,
    root_arc2_x: np.ndarray,
    root_arc2_y: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Combine all curve segments into a single continuous tooth profile."""

    x_coords = np.concatenate(
        (
            outer2_x[1:],
            edge2_x[1:],
            flank2_x[1:],
            root2_x[1:],
            root_arc2_x[1:],
            root_arc1_x,
            root1_x[1:],
            flank1_x[1:],
            edge1_x[1:],
            outer1_x[1:],
        )
    )
    y_coords = np.concatenate(
        (
            outer2_y[1:],
            edge2_y[1:],
            flank2_y[1:],
            root2_y[1:],
            root_arc2_y[1:],
            root_arc1_y,
            root1_y[1:],
            flank1_y[1:],
            edge1_y[1:],
            outer1_y[1:],
        )
    )
    return x_coords, y_coords


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
    segments_outer: int,
    segments_root: int,
) -> Tuple[np.ndarray, np.ndarray, float, float, float]:
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

    flank1_x, flank1_y = involute_curve(
        module,
        teeth_calc,
        segments_involute,
        involute_theta_start,
        involute_theta_end,
        base_angle,
        involute_start_angle,
    )
    flank2_x, flank2_y = transformations.reflect_y(flank1_x, flank1_y)

    edge_x = module * ((teeth_calc / 2) + shift_calc + addendum_calc) * np.cos(edge_angle)
    edge_y = module * ((teeth_calc / 2) + shift_calc + addendum_calc) * np.sin(edge_angle)
    edge_center_x = module * (teeth_calc / 2 + shift_calc + addendum_calc - tooth_edge_calc) * np.cos(edge_angle)
    edge_center_y = module * (teeth_calc / 2 + shift_calc + addendum_calc - tooth_edge_calc) * np.sin(edge_angle)

    edge1_x, edge1_y = edge_round_curve(
        module,
        tooth_edge_calc,
        flank1_x,
        flank1_y,
        edge_x,
        edge_y,
        edge_center_x,
        edge_center_y,
        segments_edge,
    )
    edge2_x, edge2_y = transformations.reflect_y(edge1_x, edge1_y)

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

    root1_x, root1_y = root_round_curve(
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
    root2_x, root2_y = transformations.reflect_y(root1_x, root1_y)

    outer1_x, outer1_y = outer_arc(
        module,
        teeth_calc,
        shift_calc,
        addendum_calc,
        edge_angle,
        mid_angle,
        segments_outer,
    )
    outer2_x, outer2_y = transformations.reflect_y(outer1_x, outer1_y)

    root_arc1_x, root_arc1_y = root_arc(
        module,
        teeth_calc,
        shift_calc,
        dedendum_calc,
        alpha_transition,
        segments_root,
    )
    root_arc2_x, root_arc2_y = transformations.reflect_y(root_arc1_x, root_arc1_y)

    tooth_x, tooth_y = combine_tooth_profile(
        flank1_x,
        flank1_y,
        edge1_x,
        edge1_y,
        root1_x,
        root1_y,
        outer1_x,
        outer1_y,
        root_arc1_x,
        root_arc1_y,
        flank2_x,
        flank2_y,
        edge2_x,
        edge2_y,
        root2_x,
        root2_y,
        outer2_x,
        outer2_y,
        root_arc2_x,
        root_arc2_y,
    )

    return tooth_x, tooth_y, float(teeth_calc), float(pitch_angle), float(alignment_angle)


def generate_tooth_profile(*args, **kwargs) -> Tuple[np.ndarray, np.ndarray, float, float, float]:
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
            'SEG_OUTER': 'segments_outer',
            'SEG_ROOT': 'segments_root',
        }
        normalized = {}
        for legacy_key, new_key in key_map.items():
            if legacy_key not in kwargs:
                raise TypeError(f"Missing required parameter '{legacy_key}' for tooth profile generation")
            normalized[new_key] = kwargs[legacy_key]
        return _generate_tooth_profile_impl(**normalized)

    expected_args = 14
    if len(args) != expected_args:
        raise TypeError(
            "generate_tooth_profile() expects 14 positional arguments or legacy keyword arguments"
        )

    return _generate_tooth_profile_impl(*args)
