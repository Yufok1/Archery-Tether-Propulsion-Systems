"""
Entities Module
===============
Physical entities in the KAPS simulation.

- MotherDrone: Central high-value asset
- TowedAerodynamicBody: The "skiers" in the cross formation
"""

from .mother_drone import MotherDrone, MotherDroneConfig, DroneFlightMode
from .towed_body import (
    TowedAerodynamicBody, 
    TABArray, 
    TABConfig, 
    TABState,
    FormationPosition
)

__all__ = [
    'MotherDrone',
    'MotherDroneConfig', 
    'DroneFlightMode',
    'TowedAerodynamicBody',
    'TABArray',
    'TABConfig',
    'TABState',
    'FormationPosition'
]
