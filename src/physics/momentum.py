"""
Momentum Transfer Module
========================
Physics for slingshot release, speed burst, and kinetic intercept mechanics.

The "Archer" system - where the bow leaves the arrow.

Key concepts:
- Conservation of momentum during cable release
- Centripetal acceleration for orbital wind-up
- Kinetic energy transfer optimization
- Intercept trajectory prediction
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, List, Optional
from enum import Enum


class ReleaseMode(Enum):
    """Types of cable release maneuvers"""
    INSTANT = "instant"              # Immediate release (speed burst)
    SLINGSHOT = "slingshot"          # Orbital wind-up + release
    TARGETED = "targeted"            # AI-directed intercept trajectory
    SACRIFICIAL = "sacrificial"      # Decoy/intercept sacrifice


@dataclass
class MomentumState:
    """Complete momentum state of a body"""
    mass: float                      # kg
    position: np.ndarray             # m (world frame)
    velocity: np.ndarray             # m/s (world frame)
    angular_velocity: np.ndarray     # rad/s (body frame)
    
    @property
    def linear_momentum(self) -> np.ndarray:
        """p = m * v"""
        return self.mass * self.velocity
    
    @property
    def kinetic_energy(self) -> float:
        """KE = 0.5 * m * v²"""
        return 0.5 * self.mass * np.dot(self.velocity, self.velocity)
    
    @property
    def speed(self) -> float:
        return np.linalg.norm(self.velocity)


@dataclass
class SlingshotParameters:
    """Configuration for slingshot maneuver"""
    spiral_rate: float               # rad/s - rotation rate of mother drone
    wind_up_time: float              # seconds - duration of spin-up
    release_angle: float             # radians - optimal release point
    cable_length: float              # meters
    conservation_efficiency: float   # 0-1 momentum transfer factor


class MomentumEngine:
    """
    Core physics engine for momentum-based maneuvers.
    
    Handles the "Archery" mechanics where stored kinetic energy
    is transferred from the tethered system to released TABs.
    """
    
    def __init__(self, gravity: np.ndarray = None):
        self.gravity = gravity if gravity is not None else np.array([0, 0, -9.81])
        
    def compute_release_velocity(self,
                                  mother_state: MomentumState,
                                  tab_state: MomentumState,
                                  cable_tension: float,
                                  release_mode: ReleaseMode,
                                  slingshot_params: Optional[SlingshotParameters] = None) -> np.ndarray:
        """
        Calculate the velocity vector of a TAB after cable release.
        
        The TAB inherits:
        1. Mother drone's velocity (forward component)
        2. Its own velocity from formation position
        3. Additional velocity from slingshot (if orbital release)
        4. Impulse from cable tension release
        
        Args:
            mother_state: Current state of mother drone
            tab_state: Current state of TAB before release
            cable_tension: Tension in cable at release moment (N)
            release_mode: Type of release maneuver
            slingshot_params: Optional slingshot configuration
            
        Returns:
            Final velocity vector of released TAB (m/s)
        """
        # Start with TAB's current velocity (ensure float64 for arithmetic)
        v_final = tab_state.velocity.astype(np.float64).copy()
        
        if release_mode == ReleaseMode.INSTANT:
            # Simple release - TAB keeps its current velocity
            # No additional momentum transfer
            pass
            
        elif release_mode == ReleaseMode.SLINGSHOT:
            if slingshot_params is None:
                raise ValueError("Slingshot params required for SLINGSHOT mode")
            
            # Calculate orbital velocity component
            v_orbital = self._compute_orbital_velocity(
                mother_state, tab_state, slingshot_params
            )
            
            # Add orbital velocity with efficiency factor
            v_final += v_orbital * slingshot_params.conservation_efficiency
            
        elif release_mode == ReleaseMode.TARGETED:
            # AI would pre-calculate optimal release point
            # For now, use current velocity plus tension impulse
            
            # Tension release impulse (spring energy)
            # Approximate impulse duration
            impulse_duration = 0.01  # 10ms release
            impulse = (cable_tension * impulse_duration) / tab_state.mass
            
            # Direction: along cable toward mother (reaction force)
            cable_dir = (mother_state.position - tab_state.position)
            cable_dir = cable_dir / (np.linalg.norm(cable_dir) + 1e-9)
            
            # TAB gets pushed away from mother
            v_final -= cable_dir * impulse
            
        elif release_mode == ReleaseMode.SACRIFICIAL:
            # Maximum energy transfer for intercept
            # Combine all velocity sources
            v_orbital = self._compute_orbital_velocity(
                mother_state, tab_state, 
                slingshot_params or SlingshotParameters(
                    spiral_rate=np.radians(45),
                    wind_up_time=2.0,
                    release_angle=np.pi/2,
                    cable_length=30.0,
                    conservation_efficiency=0.95
                )
            )
            v_final += v_orbital * 0.95
        
        return v_final
    
    def _compute_orbital_velocity(self,
                                   mother_state: MomentumState,
                                   tab_state: MomentumState,
                                   params: SlingshotParameters) -> np.ndarray:
        """
        Calculate the tangential velocity from orbital motion.
        
        When the mother drone spirals, the TAB on the cable end
        acts like a weight on a string being swung.
        
        v_tangential = ω × r
        
        Where:
            ω = angular velocity of rotation
            r = cable length (radius)
        """
        # Orbital velocity magnitude: v = ω * r
        v_orbital_mag = params.spiral_rate * params.cable_length
        
        # Direction: tangent to the circle at release point
        # Perpendicular to the cable, in the plane of rotation
        cable_vector = tab_state.position - mother_state.position
        cable_vector = cable_vector / (np.linalg.norm(cable_vector) + 1e-9)
        
        # Assume rotation is primarily in horizontal plane (yaw)
        # Tangent direction
        rotation_axis = np.array([0, 0, 1])  # Vertical axis
        tangent = np.cross(rotation_axis, cable_vector)
        tangent = tangent / (np.linalg.norm(tangent) + 1e-9)
        
        return tangent * v_orbital_mag
    
    def compute_mother_acceleration_burst(self,
                                           mother_state: MomentumState,
                                           tab_states: List[MomentumState],
                                           mother_thrust: float,
                                           parasitic_drag: float) -> dict:
        """
        Calculate the "speed burst" when all cables are released.
        
        This is the "mechanical capacitor" effect:
        - Before release: High thrust fighting high drag
        - After release: Same thrust, drastically reduced drag
        
        a = (F_thrust - F_drag) / m
        
        Returns:
            Dictionary with acceleration before/after and velocity predictions
        """
        # Calculate total drag contribution from TABs
        tab_drag = sum(
            0.5 * 1.225 * state.speed**2 * 0.15 * 0.08  # Approx: q * S * Cd
            for state in tab_states
        )
        
        # Cable drag contribution
        cable_drag_estimate = parasitic_drag * 0.2  # ~20% of total parasitic
        
        total_tow_drag = tab_drag + cable_drag_estimate
        
        # Acceleration BEFORE release
        total_drag_before = parasitic_drag + total_tow_drag
        a_before = (mother_thrust - total_drag_before) / mother_state.mass
        
        # Acceleration AFTER release
        total_drag_after = parasitic_drag * 0.6  # Cleaner airframe without cables
        a_after = (mother_thrust - total_drag_after) / mother_state.mass
        
        # Acceleration ratio (the "lurch")
        acceleration_multiplier = a_after / (a_before + 1e-9)
        
        # Predict velocity after 1 second of burst
        v_predicted = mother_state.speed + a_after * 1.0
        
        return {
            'acceleration_before': a_before,
            'acceleration_after': a_after,
            'acceleration_multiplier': acceleration_multiplier,
            'drag_reduction_percent': (1 - total_drag_after / total_drag_before) * 100,
            'predicted_speed_1s': v_predicted,
            'current_speed': mother_state.speed,
            'total_tow_drag': total_tow_drag
        }
    
    def predict_intercept_trajectory(self,
                                      tab_release_pos: np.ndarray,
                                      tab_release_vel: np.ndarray,
                                      threat_pos: np.ndarray,
                                      threat_vel: np.ndarray,
                                      max_time: float = 5.0) -> dict:
        """
        Predict if a released TAB will intercept a threat.
        
        Uses simple ballistic trajectory (gravity + initial velocity)
        with linear threat motion.
        
        Returns intercept time, position, and miss distance.
        """
        dt = 0.01  # 10ms timestep for prediction
        
        tab_pos = tab_release_pos.copy()
        tab_vel = tab_release_vel.copy()
        
        min_distance = float('inf')
        intercept_time = None
        intercept_pos = None
        
        for t in np.arange(0, max_time, dt):
            # Update threat position (constant velocity)
            threat_current = threat_pos + threat_vel * t
            
            # Update TAB position (ballistic + drag approximation)
            # Simplified: just gravity and initial velocity
            tab_pos_t = tab_release_pos + tab_release_vel * t + 0.5 * self.gravity * t**2
            
            # Check distance
            distance = np.linalg.norm(tab_pos_t - threat_current)
            
            if distance < min_distance:
                min_distance = distance
                intercept_time = t
                intercept_pos = tab_pos_t.copy()
            
            # Check for hit (within 2 meter radius)
            if distance < 2.0:
                return {
                    'intercept': True,
                    'time': t,
                    'position': tab_pos_t,
                    'miss_distance': distance
                }
        
        return {
            'intercept': False,
            'closest_approach_time': intercept_time,
            'closest_approach_pos': intercept_pos,
            'miss_distance': min_distance
        }


class SlingshotManeuver:
    """
    High-level controller for the slingshot release sequence.
    
    Orchestrates:
    1. Wind-up phase (spiral acceleration)
    2. Optimal release point calculation
    3. Momentum transfer execution
    4. Post-release mother drone stabilization
    """
    
    def __init__(self, momentum_engine: MomentumEngine, params: SlingshotParameters):
        self.engine = momentum_engine
        self.params = params
        
        # State machine
        self.phase = "idle"
        self.wind_up_progress = 0.0
        self.current_angle = 0.0
        
    def start_wind_up(self) -> dict:
        """Begin the orbital wind-up phase"""
        self.phase = "winding"
        self.wind_up_progress = 0.0
        self.current_angle = 0.0
        
        return {
            'status': 'wind_up_started',
            'spiral_rate_dps': np.degrees(self.params.spiral_rate),
            'duration': self.params.wind_up_time
        }
    
    def update(self, dt: float) -> dict:
        """Update slingshot state machine"""
        if self.phase != "winding":
            return {'status': self.phase}
        
        self.wind_up_progress += dt / self.params.wind_up_time
        self.current_angle += self.params.spiral_rate * dt
        
        # Calculate current orbital velocity of TABs
        v_orbital = self.params.spiral_rate * self.params.cable_length
        
        if self.wind_up_progress >= 1.0:
            self.phase = "ready"
            return {
                'status': 'ready_to_release',
                'orbital_velocity': v_orbital,
                'current_angle': np.degrees(self.current_angle)
            }
        
        return {
            'status': 'winding',
            'progress': self.wind_up_progress,
            'orbital_velocity': v_orbital * self.wind_up_progress,
            'current_angle': np.degrees(self.current_angle)
        }
    
    def calculate_optimal_release_angle(self, 
                                         target_direction: np.ndarray) -> float:
        """
        Find the release angle that sends TAB toward target.
        
        The TAB will travel tangent to the circle at release,
        so we need to release when tangent points at target.
        """
        # Normalize target direction
        target_norm = target_direction / (np.linalg.norm(target_direction) + 1e-9)
        
        # For a circular orbit, the tangent is 90° ahead of the radius
        # So we want radius to be 90° behind target direction
        release_angle = np.arctan2(target_norm[1], target_norm[0]) - np.pi/2
        
        return release_angle
    
    def execute_release(self, 
                        mother_state: MomentumState,
                        tab_state: MomentumState,
                        cable_tension: float) -> dict:
        """Execute the slingshot release"""
        if self.phase != "ready":
            return {'success': False, 'error': 'Not in ready state'}
        
        v_final = self.engine.compute_release_velocity(
            mother_state,
            tab_state,
            cable_tension,
            ReleaseMode.SLINGSHOT,
            self.params
        )
        
        self.phase = "released"
        
        return {
            'success': True,
            'release_velocity': v_final,
            'release_speed': np.linalg.norm(v_final),
            'speed_gain': np.linalg.norm(v_final) - tab_state.speed,
            'kinetic_energy': 0.5 * tab_state.mass * np.dot(v_final, v_final)
        }
