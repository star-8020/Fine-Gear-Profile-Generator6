"""Core calculation orchestrator for the Fine Gear Profile Generator."""

from __future__ import annotations

from typing import Dict, Union

from . import gear_math, geometry_generator
from .models import (
    GearPairAnalysis,
    GearPairParameters,
    GearPairResult,
    GearProfileGeometry,
)


def _resolve_parameters(
    params: Union[GearPairParameters, Dict[str, float]],
) -> GearPairParameters:
    """Normalize legacy dictionaries into ``GearPairParameters`` instances."""

    if isinstance(params, GearPairParameters):
        return params
    return GearPairParameters.from_dict(params)


def generate_gear_pair(
    params: Union[GearPairParameters, Dict[str, float]],
) -> GearPairResult:
    """Compute all derived geometry and analysis data for the supplied gear pair."""

    normalized_params = _resolve_parameters(params)
    calc_params = normalized_params.to_calculation_dict()

    contact_ratio, center_dist = gear_math.calculate_contact_ratio(
        calc_params['M'],
        calc_params['Z'],
        calc_params['z2'],
        calc_params['X'],
        calc_params['x2'],
        calc_params['ALPHA'],
        calc_params['A']
    )

    undercut_status1 = gear_math.check_undercut(
        calc_params['Z'], calc_params['ALPHA'], calc_params['X'], calc_params['A']
    )
    undercut_status2 = gear_math.check_undercut(
        calc_params['z2'], calc_params['ALPHA'], calc_params['x2'], calc_params['A']
    )

    gear1_profile, gear1_teeth, pitch_angle1, alignment_angle1 = geometry_generator.generate_tooth_profile(
        calc_params['M'], calc_params['Z'], calc_params['ALPHA'], calc_params['X'], calc_params['B'],
        calc_params['A'], calc_params['D'], calc_params['C'], calc_params['E'],
        calc_params['SEG_INVOLUTE'], calc_params['SEG_EDGE_R'], calc_params['SEG_ROOT_R'],
        calc_params['SEG_OUTER'], calc_params['SEG_ROOT']
    )

    gear2_profile, gear2_teeth, pitch_angle2, alignment_angle2 = geometry_generator.generate_tooth_profile(
        calc_params['M'], calc_params['z2'], calc_params['ALPHA'], calc_params['x2'], calc_params['B'],
        calc_params['A'], calc_params['D'], calc_params['C'], calc_params['E'],
        calc_params['SEG_INVOLUTE'], calc_params['SEG_EDGE_R'], calc_params['SEG_ROOT_R'],
        calc_params['SEG_OUTER'], calc_params['SEG_ROOT']
    )

    return GearPairResult(
        analysis=GearPairAnalysis(
            contact_ratio=contact_ratio,
            center_distance=center_dist,
        ),
        gear1=GearProfileGeometry(
            profile=gear1_profile,
            teeth=int(gear1_teeth),
            pitch_angle=float(pitch_angle1),
            alignment_angle=float(alignment_angle1),
            undercut_status=undercut_status1,
        ),
        gear2=GearProfileGeometry(
            profile=gear2_profile,
            teeth=int(gear2_teeth),
            pitch_angle=float(pitch_angle2),
            alignment_angle=float(alignment_angle2),
            undercut_status=undercut_status2,
        ),
    )
