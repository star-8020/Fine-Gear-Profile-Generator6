"""Mathematical utilities used by the gear generation engine."""

from __future__ import annotations

from typing import Tuple

import numpy as np

FULL_TURN = 2 * np.pi
RIGHT_ANGLE = np.pi / 2


def inv(alpha_rad: float) -> float:
    """Calculate the involute function for the provided angle in radians."""

    return float(np.tan(alpha_rad) - alpha_rad)


def calculate_operating_pressure_angle(
    z1: int,
    z2: int,
    x1: float,
    x2: float,
    alpha_deg: float,
) -> float:
    """Determine the operating pressure angle for a meshing gear pair."""

    alpha_rad = np.deg2rad(alpha_deg)
    inv_alpha_w = inv(alpha_rad) + 2 * (x1 + x2) * np.tan(alpha_rad) / (z1 + z2)
    alpha_w = alpha_rad
    for _ in range(10):
        function_val = inv(alpha_w) - inv_alpha_w
        derivative = np.tan(alpha_w) ** 2
        if abs(derivative) < 1e-9:
            break
        alpha_w = alpha_w - function_val / derivative
    return float(alpha_w)


def calculate_contact_ratio(
    module: float,
    z1: int,
    z2: int,
    x1: float,
    x2: float,
    alpha_deg: float,
    a1: float = 1.0,
) -> Tuple[float, float]:
    """Calculate the contact ratio and center distance for a gear pair."""

    alpha_rad = np.deg2rad(alpha_deg)
    alpha_w_rad = calculate_operating_pressure_angle(z1, z2, x1, x2, alpha_deg)

    center_distance = module * (z1 + z2) / 2 * (np.cos(alpha_rad) / np.cos(alpha_w_rad))

    base_radius_gear1 = module * z1 * np.cos(alpha_rad) / 2
    base_radius_gear2 = module * z2 * np.cos(alpha_rad) / 2

    addendum_radius_gear1 = module * (z1 / 2 + a1 + x1)
    addendum_radius_gear2 = module * (z2 / 2 + a1 + x2)

    val1 = addendum_radius_gear1**2 - base_radius_gear1**2
    val2 = addendum_radius_gear2**2 - base_radius_gear2**2
    if val1 < 0 or val2 < 0:
        return 0.0, float(center_distance)

    path_of_contact = (np.sqrt(val1) + np.sqrt(val2)) - center_distance * np.sin(alpha_w_rad)
    base_pitch = module * np.pi * np.cos(alpha_rad)
    epsilon_alpha = path_of_contact / base_pitch
    return float(epsilon_alpha), float(center_distance)


def check_undercut(teeth: int, pressure_angle_deg: float, profile_shift: float, addendum: float) -> str:
    """Assess whether a gear is at risk of undercutting."""

    if teeth <= 0:
        return "Not applicable for internal gears in this context"
    alpha_rad = np.deg2rad(pressure_angle_deg)
    x_min = addendum - (teeth / 2.0) * (np.sin(alpha_rad) ** 2)
    if profile_shift < x_min:
        return f"Warning: Risk of undercut (x < {x_min:.3f})"
    return "OK"


def handle_internal_gear_parameters(
    teeth: int,
    profile_shift: float,
    backlash_factor: float,
    addendum_factor: float,
    dedendum_factor: float,
    hob_edge_radius: float,
    tooth_edge_radius: float,
) -> Tuple[int, float, float, float, float, float, float]:
    """Normalize the sign conventions for internal gears."""

    if teeth < 0:
        teeth, profile_shift, backlash_factor = -teeth, -profile_shift, -backlash_factor
        addendum_factor, dedendum_factor = dedendum_factor, addendum_factor
        hob_edge_radius, tooth_edge_radius = tooth_edge_radius, hob_edge_radius
    return (
        int(teeth),
        float(profile_shift),
        float(backlash_factor),
        float(addendum_factor),
        float(dedendum_factor),
        float(hob_edge_radius),
        float(tooth_edge_radius),
    )


def calculate_gear_parameters(
    module: float,
    teeth: int,
    pressure_angle_deg: float,
    profile_shift: float,
    backlash_factor: float,
    addendum_factor: float,
    dedendum_factor: float,
    hob_edge_radius: float,
    tooth_edge_radius: float,
) -> Tuple[float, float, float, float, float, float, float, float, float]:
    """Derive the angular parameters used to build the tooth profile."""

    base_pressure_angle = np.deg2rad(pressure_angle_deg)
    tooth_center_angle = np.pi / teeth
    half_tooth_center_angle = tooth_center_angle / 2
    pitch_angle = FULL_TURN / teeth
    mid_tooth_angle = tooth_center_angle
    involute_start_angle = (
        base_pressure_angle
        + half_tooth_center_angle
        + backlash_factor / (teeth * np.cos(base_pressure_angle))
        - (1 + 2 * profile_shift / teeth) * np.sin(base_pressure_angle) / np.cos(base_pressure_angle)
    )
    involute_theta_start = np.tan(base_pressure_angle) + 2 * (
        hob_edge_radius * (1 - np.sin(base_pressure_angle)) + profile_shift - dedendum_factor
    ) / (teeth * np.cos(base_pressure_angle) * np.sin(base_pressure_angle))

    sqrt_val_ie = ((teeth + 2 * (profile_shift + addendum_factor - tooth_edge_radius)) / (teeth * np.cos(base_pressure_angle)))**2 - 1
    sqrt_val_ie = max(sqrt_val_ie, 0)
    involute_theta_end = 2 * tooth_edge_radius / (teeth * np.cos(base_pressure_angle)) + np.sqrt(sqrt_val_ie)

    sqrt_val_ae = ((teeth + 2 * (profile_shift + addendum_factor - tooth_edge_radius)) / (teeth * np.cos(base_pressure_angle)))**2 - 1
    sqrt_val_ae = max(sqrt_val_ae, 0)
    edge_angle = involute_start_angle + involute_theta_end - np.arctan(np.sqrt(sqrt_val_ae))

    if (edge_angle > mid_tooth_angle) and (mid_tooth_angle > involute_start_angle + involute_theta_end - np.arctan(involute_theta_end)):
        sqrt_val_e = (1 / np.cos(involute_start_angle + involute_theta_end - mid_tooth_angle))**2 - 1
        sqrt_val_e = max(sqrt_val_e, 0)
        tooth_edge_radius = (tooth_edge_radius / 2) * np.cos(base_pressure_angle) * (involute_theta_end - np.sqrt(sqrt_val_e))

    alignment_angle = RIGHT_ANGLE - tooth_center_angle

    return (
        float(base_pressure_angle),
        float(mid_tooth_angle),
        float(involute_start_angle),
        float(involute_theta_start),
        float(involute_theta_end),
        float(edge_angle),
        float(tooth_edge_radius),
        float(pitch_angle),
        float(alignment_angle),
    )