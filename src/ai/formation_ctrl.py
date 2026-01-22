"""
Formation Controller
====================
AI controller for maintaining the cross-formation of TABs.

Handles:
- Position control within cable constraints
- Aerodynamic force balancing
- Tangle prevention
- Coordinated formation maneuvers
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from enum import Enum


class FormationMode(Enum):
    """Formation operation modes"""
    CRUISE = "cruise"              # Standard cross-formation
    DEFENSIVE = "defensive"        # Expanded bubble for threat response
    TIGHT = "tight"                # Contracted for high-speed dash
    SPIRAL = "spiral"              # Orbiting for slingshot prep
    DISPERSED = "dispersed"        # Maximum separation


@dataclass
class FormationConfig:
    """Formation controller configuration"""
    nominal_separation: float = 30.0    # m - standard cable length
    defensive_expansion: float = 1.2    # Multiplier for defensive mode
    tight_contraction: float = 0.7      # Multiplier for tight mode
    correction_gain_p: float = 0.8      # Proportional gain
    correction_gain_d: float = 0.2      # Derivative gain
    max_correction_rate: float = 5.0    # m/s max position adjustment


class FormationController:
    """
    AI controller for TAB formation maintenance.
    
    Uses a PD control loop to maintain each TAB at its designated
    position relative to the mother drone, using cable tension
    and aerodynamic control surfaces.
    """
    
    def __init__(self, config: Optional[FormationConfig] = None):
        self.config = config or FormationConfig()
        self.mode = FormationMode.CRUISE
        
        # Error tracking for derivative control
        self._prev_errors: Dict[str, np.ndarray] = {}
        
        # Formation geometry (unit vectors from mother to TAB positions)
        self._formation_vectors = {
            "UP": np.array([0, 0, 1]),       # Positive Z (up)
            "DOWN": np.array([0, 0, -1]),    # Negative Z (down)
            "LEFT": np.array([0, -1, 0]),    # Negative Y (left)
            "RIGHT": np.array([0, 1, 0]),    # Positive Y (right)
        }
        
    def get_target_positions(self, 
                             mother_position: np.ndarray,
                             mother_velocity: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Calculate target positions for all TABs based on current mode.
        
        Target positions trail behind the mother drone and offset
        according to the cross-formation.
        """
        # Base separation (affected by mode)
        if self.mode == FormationMode.DEFENSIVE:
            separation = self.config.nominal_separation * self.config.defensive_expansion
        elif self.mode == FormationMode.TIGHT:
            separation = self.config.nominal_separation * self.config.tight_contraction
        else:
            separation = self.config.nominal_separation
        
        # Calculate trailing direction (opposite of velocity)
        speed = np.linalg.norm(mother_velocity)
        if speed > 1:
            trail_direction = -mother_velocity / speed
        else:
            trail_direction = np.array([-1, 0, 0])  # Default: behind in X
        
        # Base trailing position
        trail_distance = separation * 0.8  # TABs trail 80% of cable length
        base_trail = mother_position + trail_direction * trail_distance
        
        targets = {}
        for tab_id, offset_vector in self._formation_vectors.items():
            # Cross offset perpendicular to flight direction
            lateral_offset = offset_vector * (separation * 0.3)  # 30% lateral spread
            targets[tab_id] = base_trail + lateral_offset
        
        return targets
    
    def compute_control_commands(self,
                                  mother_position: np.ndarray,
                                  mother_velocity: np.ndarray,
                                  tab_positions: Dict[str, np.ndarray],
                                  tab_velocities: Dict[str, np.ndarray],
                                  dt: float) -> Dict[str, Dict]:
        """
        Calculate control commands for all TABs to maintain formation.
        
        Returns elevator and rudder commands for each TAB.
        """
        targets = self.get_target_positions(mother_position, mother_velocity)
        commands = {}
        
        for tab_id in tab_positions:
            if tab_id not in targets:
                continue
            
            target = targets[tab_id]
            current = tab_positions[tab_id]
            velocity = tab_velocities.get(tab_id, np.zeros(3))
            
            # Position error
            error = target - current
            
            # Derivative of error
            prev_error = self._prev_errors.get(tab_id, error)
            error_dot = (error - prev_error) / dt if dt > 0 else np.zeros(3)
            self._prev_errors[tab_id] = error.copy()
            
            # PD control
            correction = (self.config.correction_gain_p * error + 
                         self.config.correction_gain_d * error_dot)
            
            # Limit correction rate
            correction_mag = np.linalg.norm(correction)
            if correction_mag > self.config.max_correction_rate:
                correction = correction * (self.config.max_correction_rate / correction_mag)
            
            # Convert to control surface commands
            # Vertical correction -> elevator
            # Lateral correction -> rudder
            
            elevator_cmd = np.clip(correction[2] * 0.1, -0.4, 0.4)  # radians
            rudder_cmd = np.clip(correction[1] * 0.1, -0.35, 0.35)
            
            commands[tab_id] = {
                'elevator': elevator_cmd,
                'rudder': rudder_cmd,
                'aileron': 0.0,
                'error_magnitude': np.linalg.norm(error),
                'target_position': target,
                'correction_vector': correction
            }
        
        return commands
    
    def set_mode(self, mode: FormationMode):
        """Change formation mode"""
        self.mode = mode
    
    def check_formation_integrity(self,
                                   tab_positions: Dict[str, np.ndarray],
                                   cable_length: float) -> Dict:
        """
        Check if formation is within acceptable bounds.
        
        Returns warnings for any TABs approaching limits.
        """
        warnings = []
        status = "OK"
        
        for tab_id, pos in tab_positions.items():
            # Would need mother position for full check
            # Simplified: check relative positions
            pass
        
        # Check for tangle risk (TABs too close to each other)
        tab_ids = list(tab_positions.keys())
        for i, id1 in enumerate(tab_ids):
            for id2 in tab_ids[i+1:]:
                distance = np.linalg.norm(
                    tab_positions[id1] - tab_positions[id2]
                )
                
                if distance < cable_length * 0.2:
                    warnings.append({
                        'type': 'tangle_risk',
                        'tabs': [id1, id2],
                        'distance': distance
                    })
                    status = "WARNING"
        
        return {
            'status': status,
            'warnings': warnings,
            'mode': self.mode.value
        }
    
    def prepare_spiral(self, 
                       spiral_direction: str = "clockwise",
                       rate_dps: float = 45.0) -> Dict:
        """
        Prepare formation for spiral maneuver (slingshot wind-up).
        
        TABs will naturally orbit due to centripetal force.
        """
        self.mode = FormationMode.SPIRAL
        
        return {
            'mode': 'spiral',
            'direction': spiral_direction,
            'rate': rate_dps,
            'expected_orbital_velocity': rate_dps * self.config.nominal_separation * np.pi / 180
        }
    
    def get_formation_status(self,
                             mother_position: np.ndarray,
                             tab_positions: Dict[str, np.ndarray]) -> Dict:
        """Get detailed formation status"""
        targets = self.get_target_positions(mother_position, np.array([50, 0, 0]))
        
        tab_status = {}
        for tab_id, pos in tab_positions.items():
            if tab_id in targets:
                error = np.linalg.norm(pos - targets[tab_id])
                tab_status[tab_id] = {
                    'position': pos.tolist(),
                    'target': targets[tab_id].tolist(),
                    'error': error,
                    'on_station': error < 5.0  # Within 5m is "on station"
                }
        
        return {
            'mode': self.mode.value,
            'tabs': tab_status,
            'formation_intact': all(t['on_station'] for t in tab_status.values())
        }
