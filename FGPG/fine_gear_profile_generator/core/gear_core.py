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

    gear1_profile = geometry_generator.generate_tooth_profile(
        calc_params['M'], calc_params['Z'], calc_params['ALPHA'], calc_params['X'], calc_params['B'],
        calc_params['A'], calc_params['D'], calc_params['C'], calc_params['E'],
        calc_params['SEG_INVOLUTE'], calc_params['SEG_EDGE_R'], calc_params['SEG_ROOT_R'],
        calc_params['SEG_OUTER'], calc_params['SEG_ROOT']
    )

    gear2_profile = geometry_generator.generate_tooth_profile(
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
            coordinates=(gear1_profile[0], gear1_profile[1]),
            teeth=int(gear1_profile[2]),
            pitch_angle=float(gear1_profile[3]),
            alignment_angle=float(gear1_profile[4]),
            undercut_status=undercut_status1,
        ),
        gear2=GearProfileGeometry(
            coordinates=(gear2_profile[0], gear2_profile[1]),
            teeth=int(gear2_profile[2]),
            pitch_angle=float(gear2_profile[3]),
            alignment_angle=float(gear2_profile[4]),
            undercut_status=undercut_status2,
        ),
    )
