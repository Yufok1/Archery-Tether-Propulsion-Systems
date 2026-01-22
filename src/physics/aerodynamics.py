"""
Aerodynamics Module
===================
Lift, drag, and side-force calculations for all bodies in the KAPS system.

Implements:
- Thin airfoil theory for TAB wings
- Parasitic and induced drag models
- Side-force generation for lateral positioning
- Stall modeling with Viterna extrapolation
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple
from enum import Enum


class FlightRegime(Enum):
    """Aerodynamic flight regime classification"""
    ATTACHED = "attached_flow"
    STALL_ONSET = "stall_onset"
    DEEP_STALL = "deep_stall"
    POST_STALL = "post_stall"


@dataclass
class AeroSurface:
    """Definition of an aerodynamic surface (wing/tail/control)"""
    span: float              # Wingspan (m)
    chord: float             # Mean chord (m)
    aspect_ratio: float      # AR = span² / area
    cl_alpha: float          # Lift curve slope (per radian)
    cl_max: float            # Maximum lift coefficient (stall limit)
    cd_0: float              # Zero-lift drag coefficient
    cm_alpha: float          # Pitching moment slope (per radian)
    
    @property
    def area(self) -> float:
        return self.span * self.chord
    
    @property
    def oswald_efficiency(self) -> float:
        """Oswald efficiency factor for induced drag"""
        # Empirical correlation for rectangular planform
        return 1.78 * (1 - 0.045 * self.aspect_ratio**0.68) - 0.64


class AerodynamicsEngine:
    """
    Core aerodynamics calculator for the ATPS simulation.
    
    Provides force/moment calculations for:
    - Mother drone (large lifting body)
    - Towed Aerodynamic Bodies (TABs)
    - Control surface deflections
    """
    
    def __init__(self, air_density: float = 1.225):
        self.rho = air_density  # kg/m³
        
        # Stall behavior parameters
        self.stall_alpha_deg = 15.0
        self.stall_alpha_rad = np.radians(self.stall_alpha_deg)
        
    def compute_dynamic_pressure(self, velocity: np.ndarray) -> float:
        """
        Calculate dynamic pressure q = 0.5 * rho * V²
        
        Args:
            velocity: Velocity vector in body frame (m/s)
            
        Returns:
            Dynamic pressure (Pa)
        """
        V = np.linalg.norm(velocity)
        return 0.5 * self.rho * V**2
    
    def compute_aero_angles(self, velocity: np.ndarray) -> Tuple[float, float]:
        """
        Extract angle of attack (alpha) and sideslip (beta) from velocity.
        
        Assumes body frame: X-forward, Y-right, Z-down
        
        Returns:
            alpha: Angle of attack (radians)
            beta: Sideslip angle (radians)
        """
        V = np.linalg.norm(velocity)
        if V < 1e-6:
            return 0.0, 0.0
        
        # Angle of attack: pitch of velocity vector
        alpha = np.arctan2(velocity[2], velocity[0])
        
        # Sideslip: yaw of velocity vector
        beta = np.arcsin(np.clip(velocity[1] / V, -1, 1))
        
        return alpha, beta
    
    def compute_lift_coefficient(self, 
                                  alpha: float, 
                                  surface: AeroSurface,
                                  control_deflection: float = 0.0) -> Tuple[float, FlightRegime]:
        """
        Calculate lift coefficient with stall modeling.
        
        Uses linear region + Viterna extrapolation for post-stall.
        
        Args:
            alpha: Angle of attack (radians)
            surface: Aerodynamic surface properties
            control_deflection: Elevator/flap deflection (radians)
            
        Returns:
            cl: Lift coefficient
            regime: Current flight regime
        """
        # Effective angle of attack with control surface
        alpha_eff = alpha + 0.5 * control_deflection  # Simplified control effectiveness
        
        # Linear region
        cl_linear = surface.cl_alpha * alpha_eff
        
        if abs(alpha_eff) < self.stall_alpha_rad:
            # Attached flow - linear aerodynamics
            cl = np.clip(cl_linear, -surface.cl_max, surface.cl_max)
            regime = FlightRegime.ATTACHED
            
        elif abs(alpha_eff) < self.stall_alpha_rad * 1.2:
            # Stall onset - nonlinear transition
            stall_factor = (abs(alpha_eff) - self.stall_alpha_rad) / (0.2 * self.stall_alpha_rad)
            cl_stall = surface.cl_max * (1 - 0.3 * stall_factor)
            cl = np.sign(alpha_eff) * cl_stall
            regime = FlightRegime.STALL_ONSET
            
        else:
            # Deep/post stall - Viterna extrapolation
            # Cl approaches flat plate behavior: Cl ≈ 2 * sin(α) * cos(α)
            cl = 2 * np.sin(alpha_eff) * np.cos(alpha_eff)
            regime = FlightRegime.DEEP_STALL if abs(alpha_eff) < np.pi/4 else FlightRegime.POST_STALL
        
        return cl, regime
    
    def compute_drag_coefficient(self,
                                  cl: float,
                                  surface: AeroSurface,
                                  regime: FlightRegime) -> float:
        """
        Calculate total drag coefficient (parasitic + induced).
        
        Cd = Cd0 + Cl² / (π * e * AR)
        
        Plus stall drag penalty in separated flow.
        """
        # Parasitic drag (skin friction + pressure)
        cd_parasitic = surface.cd_0
        
        # Induced drag (drag due to lift)
        cd_induced = cl**2 / (np.pi * surface.oswald_efficiency * surface.aspect_ratio)
        
        # Stall drag penalty
        if regime in (FlightRegime.STALL_ONSET, FlightRegime.DEEP_STALL):
            cd_stall = 0.02 * (1 + abs(cl))  # Separation drag
        elif regime == FlightRegime.POST_STALL:
            cd_stall = 0.1 * (1 + abs(cl))   # Massive separation
        else:
            cd_stall = 0.0
        
        return cd_parasitic + cd_induced + cd_stall
    
    def compute_side_force_coefficient(self,
                                        beta: float,
                                        rudder_deflection: float = 0.0) -> float:
        """
        Calculate side force coefficient for lateral positioning.
        
        This is the "sailboat keel" effect that allows TABs to
        pull left/right relative to the tow line.
        
        Args:
            beta: Sideslip angle (radians)
            rudder_deflection: Rudder/fin deflection (radians)
        """
        # Side force from sideslip (like a kite in crosswind)
        cy_beta = -0.5  # Side force derivative (typical value)
        
        # Side force from rudder
        cy_rudder = 0.15  # Rudder effectiveness
        
        return cy_beta * beta + cy_rudder * rudder_deflection
    
    def compute_forces(self,
                       velocity: np.ndarray,
                       surface: AeroSurface,
                       control_inputs: dict = None) -> dict:
        """
        Main aerodynamic force calculation.
        
        Args:
            velocity: Velocity vector in body frame (m/s)
            surface: Aerodynamic surface definition
            control_inputs: Dict with 'elevator', 'aileron', 'rudder' (radians)
            
        Returns:
            Dictionary with all aerodynamic outputs
        """
        if control_inputs is None:
            control_inputs = {'elevator': 0.0, 'aileron': 0.0, 'rudder': 0.0}
        
        # Dynamic pressure
        q = self.compute_dynamic_pressure(velocity)
        V = np.linalg.norm(velocity)
        
        if V < 1e-6:
            return {
                'lift': 0.0,
                'drag': 0.0,
                'side_force': 0.0,
                'lift_vector': np.zeros(3),
                'drag_vector': np.zeros(3),
                'side_vector': np.zeros(3),
                'total_force': np.zeros(3),
                'regime': FlightRegime.ATTACHED,
                'cl': 0.0,
                'cd': 0.0,
                'cy': 0.0
            }
        
        # Aero angles
        alpha, beta = self.compute_aero_angles(velocity)
        
        # Coefficients
        cl, regime = self.compute_lift_coefficient(
            alpha, surface, control_inputs.get('elevator', 0.0)
        )
        cd = self.compute_drag_coefficient(cl, surface, regime)
        cy = self.compute_side_force_coefficient(
            beta, control_inputs.get('rudder', 0.0)
        )
        
        # Force magnitudes
        S = surface.area
        lift = q * S * cl
        drag = q * S * cd
        side_force = q * S * cy
        
        # Convert to body-axis force vectors
        # Lift is perpendicular to velocity, in the vertical plane
        # Drag is opposite to velocity
        # Side force is perpendicular to velocity, in the horizontal plane
        
        vel_norm = velocity / V
        
        # Drag vector (opposite to velocity)
        drag_vector = -vel_norm * drag
        
        # Lift vector (perpendicular to velocity, in xz plane)
        # Simplified: assume wings are horizontal
        lift_direction = np.array([
            -np.sin(alpha) * np.cos(beta),
            0,
            -np.cos(alpha)
        ])
        lift_direction /= (np.linalg.norm(lift_direction) + 1e-9)
        lift_vector = lift_direction * lift
        
        # Side force vector (perpendicular to velocity, in xy plane)
        side_direction = np.array([0, 1, 0])  # Simplified
        side_vector = side_direction * side_force
        
        # Total aerodynamic force
        total_force = lift_vector + drag_vector + side_vector
        
        return {
            'lift': lift,
            'drag': drag,
            'side_force': side_force,
            'lift_vector': lift_vector,
            'drag_vector': drag_vector,
            'side_vector': side_vector,
            'total_force': total_force,
            'regime': regime,
            'alpha': alpha,
            'beta': beta,
            'cl': cl,
            'cd': cd,
            'cy': cy,
            'dynamic_pressure': q
        }


class TABAerodynamics(AerodynamicsEngine):
    """
    Specialized aerodynamics for Towed Aerodynamic Bodies.
    
    Includes:
    - Formation holding force calculations
    - Optimal control surface deflections for positioning
    - Cross-formation side-force requirements
    """
    
    def __init__(self, air_density: float = 1.225):
        super().__init__(air_density)
        
        # TAB wing definition (from config)
        self.tab_wing = AeroSurface(
            span=1.5,
            chord=0.3,
            aspect_ratio=5.0,
            cl_alpha=5.7,
            cl_max=1.4,
            cd_0=0.04,
            cm_alpha=-0.1
        )
    
    def compute_formation_forces(self,
                                  target_position: str,
                                  current_offset: np.ndarray,
                                  velocity: np.ndarray) -> dict:
        """
        Calculate required aerodynamic forces to maintain formation position.
        
        Args:
            target_position: "UP", "DOWN", "LEFT", or "RIGHT"
            current_offset: Current position relative to desired (m)
            velocity: Current velocity vector (m/s)
            
        Returns:
            Required control inputs and expected forces
        """
        V = np.linalg.norm(velocity)
        if V < 5.0:  # Minimum airspeed for control authority
            return {'elevator': 0.0, 'rudder': 0.0, 'achievable': False}
        
        # Target force directions for each formation position
        force_requirements = {
            "UP": np.array([0, 0, 1]),      # Positive lift (upward)
            "DOWN": np.array([0, 0, -1]),   # Negative lift (downward)
            "LEFT": np.array([0, -1, 0]),   # Side force left
            "RIGHT": np.array([0, 1, 0]),   # Side force right
        }
        
        target_direction = force_requirements.get(target_position, np.zeros(3))
        
        # Calculate required deflections (simplified inverse)
        if target_position in ("UP", "DOWN"):
            # Use elevator for vertical positioning
            required_cl = 0.8 * np.sign(target_direction[2])
            elevator = required_cl / (self.tab_wing.cl_alpha * 0.5)  # Inverse of cl calc
            rudder = 0.0
        else:
            # Use rudder for lateral positioning
            elevator = 0.3  # Slight positive lift to counter cable weight
            required_cy = 0.3 * target_direction[1]
            rudder = required_cy / 0.15  # Inverse of cy_rudder
        
        # Clamp to authority limits
        elevator = np.clip(elevator, -np.radians(25), np.radians(25))
        rudder = np.clip(rudder, -np.radians(20), np.radians(20))
        
        return {
            'elevator': elevator,
            'rudder': rudder,
            'aileron': 0.0,
            'achievable': True,
            'target_direction': target_direction
        }
