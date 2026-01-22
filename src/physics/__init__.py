"""
Physics Module
==============
Core physics engines for the Archery Tether Propulsion System.

Submodules:
- tether_dynamics: Cable tension, constraints, snap mechanics
- aerodynamics: Lift, drag, side-force calculations
- momentum: Slingshot release and kinetic intercept physics
"""

from .tether_dynamics import (
    TetherConstraint,
    TetherArray,
    CableState,
    CableProperties
)

from .aerodynamics import (
    AerodynamicsEngine,
    TABAerodynamics,
    AeroSurface,
    FlightRegime
)

from .momentum import (
    MomentumEngine,
    MomentumState,
    SlingshotManeuver,
    SlingshotParameters,
    ReleaseMode
)

from .cable_geometry import (
    CableGeometry,
    CableIntersectionDetector,
    SectorConstrainedActionSpace,
    OperationalSector,
    TangleState,
    OPERATIONAL_SECTORS
)

__all__ = [
    # Tether
    'TetherConstraint',
    'TetherArray', 
    'CableState',
    'CableProperties',
    # Aero
    'AerodynamicsEngine',
    'TABAerodynamics',
    'AeroSurface',
    'FlightRegime',
    # Momentum
    'MomentumEngine',
    'MomentumState',
    'SlingshotManeuver',
    'SlingshotParameters',
    'ReleaseMode',
    # Cable Geometry
    'CableGeometry',
    'CableIntersectionDetector',
    'SectorConstrainedActionSpace',
    'OperationalSector',
    'TangleState',
    'OPERATIONAL_SECTORS',
]
