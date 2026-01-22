"""
Towed Aerodynamic Body (TAB) Entity
===================================
AIRFOIL DEFENDERS - The 4 flying guardians of the Buzzard.

CRITICAL UNDERSTANDING:
- The CABLES connect the Buzzard to each TAB (simple tethers)
- The TABs themselves ARE AIRFOILS - they FLY, they don't just hang
- Each TAB has wings, control surfaces, and active aerodynamic control
- Their ONLY PURPOSE is to PROTECT THE BUZZARD

The TABs are NOT passive - they actively:
- Generate lift with their wings
- Maneuver using elevator, aileron, rudder
- Position themselves to intercept threats
- Sacrifice themselves when necessary

Architecture:
    
         [TAB-UP]           <- Airfoil with wings
            |
            | (cable)
            |
    [TAB-L]-+-[TAB-R]       <- Cross formation, each is an airfoil
            |
            | 
         [BUZZARD]          <- Mother drone (what we're protecting)
            |
            |
         [TAB-DOWN]         <- Airfoil with wings

Each TAB has:
- Wingspan: 1.5m
- Control surfaces: elevator, aileron, rudder  
- Mass: 8kg
- Purpose: DEFEND THE BUZZARD AT ALL COSTS
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
from enum import Enum

from ..physics import (
    MomentumState, 
    AeroSurface, 
    TABAerodynamics,
    CableState
)


class TABState(Enum):
    """Operational states for a TAB"""
    FORMATION = "formation"        # Normal towed operation - watching for threats
    MANEUVERING = "maneuvering"    # Active position adjustment for better coverage
    PRE_RELEASE = "pre_release"    # Preparing for intercept run
    RELEASED = "released"          # Detached, on intercept trajectory
    INTERCEPT = "intercept"        # Active intercept in progress
    SACRIFICED = "sacrificed"      # Gave its life to protect the Buzzard
    DESTROYED = "destroyed"        # Lost without protecting anything


class FormationPosition(Enum):
    """Cross-formation positions around the Buzzard"""
    UP = "UP"       # Above - covers high threats
    DOWN = "DOWN"   # Below - covers low threats  
    LEFT = "LEFT"   # Port side coverage
    RIGHT = "RIGHT" # Starboard coverage


@dataclass
class TABConfig:
    """
    Configuration for Towed Aerodynamic Body (AIRFOIL).
    
    These are small flying wings, not passive weights!
    """
    mass: float = 8.0                      # kg - light but substantial
    wingspan: float = 1.5                  # m - actual wings
    chord: float = 0.3                     # m - wing chord
    drag_coefficient: float = 0.08         # Aerodynamic, not a brick
    frontal_area: float = 0.15             # m² - small profile
    
    # Control surface limits (radians) - ACTIVE FLIGHT CONTROL
    elevator_max: float = np.radians(25)   # Pitch control
    aileron_max: float = np.radians(30)    # Roll control  
    rudder_max: float = np.radians(20)     # Yaw control
    
    # Servo speed (rad/s) - fast response for intercept
    actuation_rate: float = np.radians(200)
    
    # Lift characteristics
    cl_max: float = 1.4                    # Maximum lift coefficient
    stall_angle: float = np.radians(15)    # Stall angle of attack


class TowedAerodynamicBody:
    """
    A single Towed Aerodynamic Body (TAB) - AN AIRFOIL DEFENDER.
    
    This is a small flying wing connected to the Buzzard by cable.
    It actively flies, generating lift and maneuvering to protect.
    
    PRIMARY MISSION: Protect the Buzzard at all costs.
    
    Capabilities:
    - Aerodynamic flight within cable constraints
    - Active threat tracking and intercept
    - Kinetic kill via collision with threat
    - Sacrificial defense (intentional self-destruction to stop threat)
    """
    
    def __init__(self, 
                 tab_id: str,
                 formation_pos: FormationPosition,
                 cable_length: float = 30.0,
                 config: Optional[TABConfig] = None):
        
        self.tab_id = tab_id
        self.formation_pos = formation_pos
        self.cable_length = cable_length
        self.config = config or TABConfig()
        
        # State
        self.state = TABState.FORMATION
        self.position = np.zeros(3)
        self.velocity = np.zeros(3)
        self.orientation = np.zeros(3)  # roll, pitch, yaw
        self.angular_velocity = np.zeros(3)
        
        # Control surfaces (current deflection in radians)
        self.elevator = 0.0
        self.aileron = 0.0
        self.rudder = 0.0
        
        # Control targets
        self._target_elevator = 0.0
        self._target_rudder = 0.0
        
        # Aerodynamics
        self.aero = TABAerodynamics()
        self.wing = AeroSurface(
            span=self.config.wingspan,
            chord=self.config.chord,
            aspect_ratio=self.config.wingspan**2 / (self.config.wingspan * self.config.chord),
            cl_alpha=5.7,
            cl_max=1.4,
            cd_0=self.config.drag_coefficient,
            cm_alpha=-0.1
        )
        
        # Initialize position based on formation
        self._initialize_formation_position()
        
    def _initialize_formation_position(self):
        """Set initial position based on formation assignment"""
        # Offset from mother drone (assume mother at origin initially)
        # TABs start behind and offset according to their position
        offsets = {
            FormationPosition.UP: np.array([-self.cable_length, 0, 5]),
            FormationPosition.DOWN: np.array([-self.cable_length, 0, -5]),
            FormationPosition.LEFT: np.array([-self.cable_length, -10, 0]),
            FormationPosition.RIGHT: np.array([-self.cable_length, 10, 0]),
        }
        self.position = offsets[self.formation_pos]
        
        # Initial velocity matches formation flight
        self.velocity = np.array([50.0, 0.0, 0.0])
    
    @property
    def momentum_state(self) -> MomentumState:
        """Get current momentum state for physics calculations"""
        return MomentumState(
            mass=self.config.mass,
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            angular_velocity=self.angular_velocity.copy()
        )
    
    @property
    def speed(self) -> float:
        """Current airspeed in m/s"""
        return np.linalg.norm(self.velocity)
    
    @property
    def is_attached(self) -> bool:
        """Check if still attached to mother drone"""
        return self.state in (TABState.FORMATION, TABState.MANEUVERING, TABState.PRE_RELEASE)
    
    def set_control_targets(self, elevator: float, rudder: float, aileron: float = 0.0):
        """
        Set target control surface deflections.
        
        Servos will actuate toward these targets at limited rate.
        
        Args:
            elevator: Target elevator deflection (radians)
            rudder: Target rudder deflection (radians)  
            aileron: Target aileron deflection (radians)
        """
        self._target_elevator = np.clip(elevator, -self.config.elevator_max, self.config.elevator_max)
        self._target_rudder = np.clip(rudder, -self.config.rudder_max, self.config.rudder_max)
    
    def update_control_surfaces(self, dt: float):
        """
        Actuate control surfaces toward targets at limited rate.
        
        Models realistic servo response.
        """
        max_delta = self.config.actuation_rate * dt
        
        # Elevator
        delta_elev = self._target_elevator - self.elevator
        self.elevator += np.clip(delta_elev, -max_delta, max_delta)
        
        # Rudder
        delta_rud = self._target_rudder - self.rudder
        self.rudder += np.clip(delta_rud, -max_delta, max_delta)
    
    def compute_formation_control(self, 
                                   mother_pos: np.ndarray,
                                   mother_vel: np.ndarray,
                                   cable_tension: float) -> Dict:
        """
        Calculate control inputs to maintain formation position.
        
        Uses the TABAerodynamics formation controller.
        """
        # Desired offset from mother based on formation position
        desired_offsets = {
            FormationPosition.UP: np.array([-self.cable_length * 0.9, 0, 10]),
            FormationPosition.DOWN: np.array([-self.cable_length * 0.9, 0, -10]),
            FormationPosition.LEFT: np.array([-self.cable_length * 0.9, -15, 0]),
            FormationPosition.RIGHT: np.array([-self.cable_length * 0.9, 15, 0]),
        }
        
        desired_pos = mother_pos + desired_offsets[self.formation_pos]
        current_offset = self.position - desired_pos
        
        # Get required control inputs
        control = self.aero.compute_formation_forces(
            self.formation_pos.value,
            current_offset,
            self.velocity
        )
        
        if control['achievable']:
            self.set_control_targets(
                elevator=control['elevator'],
                rudder=control['rudder']
            )
        
        return {
            'formation_error': np.linalg.norm(current_offset),
            'control_achievable': control['achievable'],
            'elevator_cmd': control['elevator'],
            'rudder_cmd': control['rudder']
        }
    
    def update(self,
               dt: float,
               cable_force: np.ndarray,
               cable_state: CableState) -> Dict:
        """
        Update TAB state for one timestep.
        
        Args:
            dt: Timestep in seconds
            cable_force: Force from cable tension (N)
            cable_state: Current state of the tether
            
        Returns:
            Dictionary with updated state info
        """
        # Check for cable release
        if cable_state == CableState.SEVERED and self.state in (TABState.FORMATION, TABState.MANEUVERING):
            self.state = TABState.RELEASED
        
        # Update control surfaces
        self.update_control_surfaces(dt)
        
        # Collect forces
        forces = np.zeros(3)
        
        # Aerodynamic forces
        control_inputs = {
            'elevator': self.elevator,
            'aileron': self.aileron,
            'rudder': self.rudder
        }
        aero_result = self.aero.compute_forces(self.velocity, self.wing, control_inputs)
        forces += aero_result['total_force']
        
        # Cable force (if attached)
        if self.is_attached:
            forces += cable_force
        
        # Gravity
        forces += np.array([0, 0, -9.81]) * self.config.mass
        
        # Newton's second law
        acceleration = forces / self.config.mass
        
        # Integrate
        self.velocity += acceleration * dt
        self.position += self.velocity * dt
        
        # Ground collision check
        if self.position[2] < 0:
            self.position[2] = 0
            self.velocity[2] = 0
            if self.state == TABState.RELEASED:
                self.state = TABState.DESTROYED
        
        return {
            'position': self.position.copy(),
            'velocity': self.velocity.copy(),
            'speed': self.speed,
            'state': self.state.value,
            'elevator': np.degrees(self.elevator),
            'rudder': np.degrees(self.rudder),
            'lift': aero_result['lift'],
            'drag': aero_result['drag'],
            'regime': aero_result['regime'].value
        }
    
    def prepare_for_release(self):
        """Prepare TAB for release (pre-release checks)"""
        self.state = TABState.PRE_RELEASE
        # Could trigger any pre-release sequences here
    
    def execute_release(self, release_velocity: np.ndarray) -> Dict:
        """
        Execute release with given velocity boost.
        
        Args:
            release_velocity: New velocity vector post-release
            
        Returns:
            Release telemetry
        """
        old_speed = self.speed
        self.velocity = release_velocity
        self.state = TABState.RELEASED
        
        return {
            'success': True,
            'old_speed': old_speed,
            'new_speed': self.speed,
            'speed_gain': self.speed - old_speed,
            'trajectory': release_velocity / (np.linalg.norm(release_velocity) + 1e-9)
        }
    
    def set_intercept_mode(self, target_position: np.ndarray):
        """
        Switch to intercept mode toward a target.
        
        Only valid if already released.
        """
        if self.state == TABState.RELEASED:
            self.state = TABState.INTERCEPT
            # Calculate intercept trajectory adjustments
            # (would use control surfaces for terminal guidance)
    
    def get_status_report(self) -> Dict:
        """Get comprehensive status for UI/telemetry"""
        return {
            'id': self.tab_id,
            'formation_position': self.formation_pos.value,
            'state': self.state.value,
            'position': {
                'x': self.position[0],
                'y': self.position[1],
                'z': self.position[2]
            },
            'velocity': {
                'x': self.velocity[0],
                'y': self.velocity[1],
                'z': self.velocity[2],
                'speed': self.speed
            },
            'control_surfaces': {
                'elevator': np.degrees(self.elevator),
                'aileron': np.degrees(self.aileron),
                'rudder': np.degrees(self.rudder)
            },
            'attached': self.is_attached
        }


class TABArray:
    """
    Manages the complete array of TABs in cross-formation.
    
    Coordinates:
    - Formation maintenance
    - Synchronized maneuvers
    - Selective release sequences
    - Threat-based TAB selection
    """
    
    def __init__(self, cable_length: float = 30.0, mother_position: np.ndarray = None):
        self.tabs: Dict[str, TowedAerodynamicBody] = {}
        self.cable_length = cable_length
        
        if mother_position is None:
            mother_position = np.array([0.0, 0.0, 1000.0])
        
        # Formation offsets: cross pattern behind mother drone
        # Mother flies forward (+X), TABs trail behind (-X direction offset)
        # with lateral/vertical spread for the cross
        formation_offsets = {
            FormationPosition.UP: np.array([-cable_length * 0.8, 0.0, cable_length * 0.6]),
            FormationPosition.DOWN: np.array([-cable_length * 0.8, 0.0, -cable_length * 0.6]),
            FormationPosition.LEFT: np.array([-cable_length * 0.8, -cable_length * 0.6, 0.0]),
            FormationPosition.RIGHT: np.array([-cable_length * 0.8, cable_length * 0.6, 0.0]),
        }
        
        # Create cross-formation
        for pos in FormationPosition:
            tab = TowedAerodynamicBody(
                tab_id=f"TAB-{pos.value}",
                formation_pos=pos,
                cable_length=cable_length
            )
            # Set initial position relative to mother drone
            tab.position = mother_position + formation_offsets[pos]
            tab.velocity = np.array([50.0, 0.0, 0.0])  # Match mother's cruise velocity
            self.tabs[pos.value] = tab
    
    def get_positions(self) -> Dict[str, np.ndarray]:
        """Get all TAB positions"""
        return {
            tab_id: tab.position.copy()
            for tab_id, tab in self.tabs.items()
        }
    
    def get_velocities(self) -> Dict[str, np.ndarray]:
        """Get all TAB velocities"""
        return {
            tab_id: tab.velocity.copy()
            for tab_id, tab in self.tabs.items()
        }
    
    def get_momentum_states(self) -> Dict[str, MomentumState]:
        """Get momentum states for all TABs"""
        return {
            tab_id: tab.momentum_state
            for tab_id, tab in self.tabs.items()
        }
    
    def update_all(self,
                   dt: float,
                   cable_forces: Dict[str, np.ndarray],
                   cable_states: Dict[str, CableState]) -> Dict:
        """
        Update all TABs for one timestep.
        
        Returns aggregated status.
        """
        results = {}
        
        for tab_id, tab in self.tabs.items():
            cable_force = cable_forces.get(tab_id, np.zeros(3))
            cable_state = cable_states.get(tab_id, CableState.ATTACHED)
            
            results[tab_id] = tab.update(dt, cable_force, cable_state)
        
        return results
    
    def select_best_interceptor(self, threat_direction: np.ndarray) -> Optional[str]:
        """
        Select the TAB best positioned to intercept a threat.
        
        Args:
            threat_direction: Unit vector toward threat from mother drone
            
        Returns:
            TAB ID of best interceptor, or None if none suitable
        """
        best_score = -1
        best_tab = None
        
        for tab_id, tab in self.tabs.items():
            if not tab.is_attached:
                continue
            
            # Score based on alignment with threat direction
            tab_direction = tab.position - np.zeros(3)  # From mother
            tab_direction = tab_direction / (np.linalg.norm(tab_direction) + 1e-9)
            
            alignment = np.dot(tab_direction, threat_direction)
            
            if alignment > best_score:
                best_score = alignment
                best_tab = tab_id
        
        return best_tab if best_score > 0.5 else None
    
    def count_attached(self) -> int:
        """Count TABs still attached"""
        return sum(1 for tab in self.tabs.values() if tab.is_attached)
