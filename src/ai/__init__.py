"""
AI Module
=========
Artificial intelligence controllers for the KAPS system.

- DefensiveMatrixAI: 360° threat detection and intercept
- FormationController: Cross-formation maintenance
"""

from .defensive_matrix import (
    DefensiveMatrixAI,
    DefenseConfig,
    TrackedThreat,
    ThreatType,
    ThreatPriority
)

from .formation_ctrl import (
    FormationController,
    FormationConfig,
    FormationMode
)

__all__ = [
    'DefensiveMatrixAI',
    'DefenseConfig',
    'TrackedThreat',
    'ThreatType',
    'ThreatPriority',
    'FormationController',
    'FormationConfig',
    'FormationMode'
]
