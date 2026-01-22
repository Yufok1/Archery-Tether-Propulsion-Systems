"""
Mother Drone Entity
===================
The central high-value asset in the KAPS system.

The "Buzzard" - large surveillance drone that provides:
- Primary thrust for the tethered system
- Command and control for TAB array
- Sensor platform and data fusion
- Emergency escape capability via speed burst
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum

from ..physics import MomentumState, AeroSurface, AerodynamicsEngine


class DroneFlightMode(Enum):
    """Operational modes for mother drone"""
    CRUISE = "cruise"              # Normal towed flight
    EVASIVE = "evasive"            # Active threat avoidance
    INTERCEPT_SUPPORT = "intercept" # Supporting TAB intercept maneuver
    SPEED_BURST = "speed_burst"    # Post-release acceleration
    RECOVERY = "recovery"          # Stabilizing after maneuver


@dataclass
class MotherDroneConfig:
    """Configuration parameters for mother drone"""
    mass: float = 150.0                    # kg
    wingspan: float = 8.0                  # m
    max_thrust: float = 2500.0             # N
    cruise_thrust_ratio: float = 0.9
    drag_coefficient: float = 0.025
    frontal_area: float = 1.2              # m²
    
    # Inertia tensor (simplified as diagonal)
    inertia_xx: float = 50.0               # kg*m²
    inertia_yy: float = 80.0               # kg*m²
    inertia_zz: float = 100.0              # kg*m²
    
    # Control limits
    max_bank_angle: float = 60.0           # degrees
    max_pitch_rate: float = 30.0           # deg/s
    max_yaw_rate: float = 45.0             # deg/s


class MotherDrone:
    """
    The central high-value asset of the KAPS system.
    
    Responsibilities:
    - Provide thrust for entire tethered formation
    - Manage cable deployment and release
    - Execute evasive maneuvers
    - Coordinate TAB positioning via formation commands
    """
    
    def __init__(self, config: Optional[MotherDroneConfig] = None):
        self.config = config or MotherDroneConfig()
        
        # State
        self.position = np.array([0.0, 0.0, 1000.0])  # Start at 1km altitude
        self.velocity = np.array([50.0, 0.0, 0.0])    # 50 m/s forward
        self.orientation = np.array([0.0, 0.0, 0.0])  # Euler angles (roll, pitch, yaw)
        self.angular_velocity = np.zeros(3)
        
        # Control state
        self.throttle = self.config.cruise_thrust_ratio
        self.flight_mode = DroneFlightMode.CRUISE
        
        # Aerodynamics
        self.aero = AerodynamicsEngine()
        self.wing = AeroSurface(
            span=self.config.wingspan,
            chord=1.0,
            aspect_ratio=self.config.wingspan,  # Assuming chord = 1m
            cl_alpha=5.5,
            cl_max=1.6,
            cd_0=self.config.drag_coefficient,
            cm_alpha=-0.08
        )
        
        # TAB management
        self.attached_tabs: Dict[str, bool] = {
            "UP": True,
            "DOWN": True, 
            "LEFT": True,
            "RIGHT": True
        }
        
        # Telemetry history
        self._speed_history: List[float] = []
        
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
    def current_thrust(self) -> float:
        """Get current thrust output in Newtons"""
        return self.throttle * self.config.max_thrust
    
    @property
    def speed(self) -> float:
        """Current airspeed in m/s"""
        return np.linalg.norm(self.velocity)
    
    @property
    def altitude(self) -> float:
        """Current altitude in meters"""
        return self.position[2]
    
    @property
    def heading(self) -> float:
        """Current heading in degrees (0 = North)"""
        return np.degrees(np.arctan2(self.velocity[1], self.velocity[0]))
    
    def get_anchor_points_world(self) -> Dict[str, np.ndarray]:
        """
        Get cable anchor points in world coordinates.
        
        Accounts for drone orientation.
        """
        # Local anchor points (body frame)
        anchors_local = {
            "UP": np.array([-0.5, 0.0, 0.3]),      # Rear top
            "DOWN": np.array([-0.5, 0.0, -0.3]),   # Rear bottom
            "LEFT": np.array([-0.5, -0.5, 0.0]),   # Rear left
            "RIGHT": np.array([-0.5, 0.5, 0.0]),   # Rear right
        }
        
        # Simplified rotation (just yaw for now)
        yaw = self.orientation[2]
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        
        R_yaw = np.array([
            [cos_yaw, -sin_yaw, 0],
            [sin_yaw, cos_yaw, 0],
            [0, 0, 1]
        ])
        
        anchors_world = {}
        for name, local_pos in anchors_local.items():
            world_offset = R_yaw @ local_pos
            anchors_world[name] = self.position + world_offset
            
        return anchors_world
    
    def compute_parasitic_drag(self) -> float:
        """
        Calculate drag on mother drone body (excluding TAB drag).
        
        D = 0.5 * rho * V² * Cd * A
        """
        rho = 1.225  # Sea level air density
        V_squared = np.dot(self.velocity, self.velocity)
        
        drag = 0.5 * rho * V_squared * self.config.drag_coefficient * self.config.frontal_area
        
        return drag
    
    def update(self, 
               dt: float,
               tether_forces: np.ndarray,
               external_forces: np.ndarray = None) -> Dict:
        """
        Update drone state for one timestep.
        
        Args:
            dt: Timestep in seconds
            tether_forces: Sum of all cable tension forces (N)
            external_forces: Any additional forces (wind, etc.)
            
        Returns:
            Dictionary with updated state info
        """
        # Collect all forces
        forces = np.zeros(3)
        
        # Thrust (forward in body frame)
        thrust_direction = np.array([
            np.cos(self.orientation[2]) * np.cos(self.orientation[1]),
            np.sin(self.orientation[2]) * np.cos(self.orientation[1]),
            -np.sin(self.orientation[1])
        ])
        forces += thrust_direction * self.current_thrust
        
        # Aerodynamic forces
        aero_result = self.aero.compute_forces(self.velocity, self.wing)
        forces += aero_result['total_force']
        
        # Tether forces (already in world frame)
        forces += tether_forces
        
        # External forces
        if external_forces is not None:
            forces += external_forces
        
        # Gravity
        forces += np.array([0, 0, -9.81]) * self.config.mass
        
        # Newton's second law
        acceleration = forces / self.config.mass
        
        # Integrate (simple Euler)
        self.velocity += acceleration * dt
        self.position += self.velocity * dt
        
        # Store speed history
        self._speed_history.append(self.speed)
        if len(self._speed_history) > 1000:
            self._speed_history.pop(0)
        
        return {
            'position': self.position.copy(),
            'velocity': self.velocity.copy(),
            'speed': self.speed,
            'altitude': self.altitude,
            'heading': self.heading,
            'thrust': self.current_thrust,
            'drag': aero_result['drag'],
            'acceleration': np.linalg.norm(acceleration),
            'flight_mode': self.flight_mode.value
        }
    
    def release_tab(self, tab_id: str) -> bool:
        """
        Release a specific TAB cable.
        
        Returns True if successful, False if already released.
        """
        if tab_id in self.attached_tabs and self.attached_tabs[tab_id]:
            self.attached_tabs[tab_id] = False
            return True
        return False
    
    def release_all_tabs(self) -> int:
        """
        Emergency release of all TABs (speed burst maneuver).
        
        Returns number of TABs released.
        """
        count = 0
        for tab_id in self.attached_tabs:
            if self.attached_tabs[tab_id]:
                self.attached_tabs[tab_id] = False
                count += 1
        
        self.flight_mode = DroneFlightMode.SPEED_BURST
        return count
    
    def set_throttle(self, throttle: float):
        """Set throttle (0.0 to 1.0)"""
        self.throttle = np.clip(throttle, 0.0, 1.0)
    
    def execute_spiral(self, rate_dps: float, dt: float):
        """
        Execute spiral maneuver for slingshot wind-up.
        
        Args:
            rate_dps: Yaw rate in degrees per second
            dt: Timestep
        """
        self.flight_mode = DroneFlightMode.INTERCEPT_SUPPORT
        self.angular_velocity[2] = np.radians(rate_dps)
        self.orientation[2] += self.angular_velocity[2] * dt
        
        # Wrap yaw to [-pi, pi]
        self.orientation[2] = np.arctan2(
            np.sin(self.orientation[2]),
            np.cos(self.orientation[2])
        )
    
    def get_status_report(self) -> Dict:
        """Get comprehensive status for UI/telemetry"""
        return {
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
            'orientation': {
                'roll': np.degrees(self.orientation[0]),
                'pitch': np.degrees(self.orientation[1]),
                'yaw': np.degrees(self.orientation[2])
            },
            'systems': {
                'throttle': self.throttle,
                'thrust': self.current_thrust,
                'flight_mode': self.flight_mode.value,
                'tabs_attached': sum(self.attached_tabs.values())
            },
            'tabs': self.attached_tabs.copy()
        }
