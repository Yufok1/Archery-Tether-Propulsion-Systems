"""
Slingshot / Nunchaku Dynamics for KAPS
======================================

BOLA MODE: All TABs consolidate into a single articulated mass.
The Buzzard swings this combined mass like Bruce Lee with nunchucks.

Physics Principles:
-------------------
1. MOMENTUM WHIP: Angular velocity transfers through cable to terminal mass
2. CENTRIPETAL STORAGE: Circular motion stores kinetic energy in the "bola head"
3. RELEASE TIMING: DreamerV3 learns optimal release angles for max velocity
4. DEPLOYABLE AIRFOILS: Grid-fin style surfaces for mid-flight steering

The key insight: Instead of 4 independent TABs, we can CONSOLIDATE them into
a single heavy striking mass connected by braided cables. The Buzzard becomes
the "handle" and the TAB cluster becomes the "nunchaku head".

                    STANDARD MODE              SLINGSHOT/BOLA MODE
                    
                         ↑ TAB                     ╔═══════╗
                         │                         ║ BOLA  ║ ← All TABs merged
                    ───●─┼─●───                    ║ HEAD  ║   into single mass
                         │                         ╚═══╤═══╝
                         ↓                             │
                       [BUZ]                      [BUZZARD] ← Swings like nunchaku
                                                       ↻

DEPLOYABLE AIRFOIL GEOMETRY:
----------------------------
Like SpaceX Falcon 9 grid fins - titanium lattice that deploys from the
TAB body to provide massive drag/lift modulation:

    RETRACTED:          DEPLOYED:
    
      ┌──┐               ╔══════╗
      │  │               ║ ╔══╗ ║  ← Grid fin array
      │  │               ║ ╠══╣ ║
      └──┘               ╚══════╝

The grid fins "sail" through the air, allowing precise trajectory control
during the swing and after release.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum, auto


class SlingshotState(Enum):
    """Slingshot system operational states."""
    DISPERSED = auto()      # TABs operating independently (standard mode)
    CONSOLIDATING = auto()  # TABs converging to form bola
    BOLA_READY = auto()     # Consolidated, ready to swing
    SWINGING = auto()       # Active nunchaku motion
    RELEASING = auto()      # Cable release initiated
    BALLISTIC = auto()      # Post-release, bola in free flight
    SAILING = auto()        # Grid fins deployed, steering to target


class AirfoilDeployState(Enum):
    """Deployable airfoil surface states."""
    RETRACTED = auto()      # Minimum drag, stored
    DEPLOYING = auto()      # Transitioning
    DEPLOYED = auto()       # Full extension, max control
    FEATHERED = auto()      # Edge-on, minimum resistance


@dataclass
class GridFinConfig:
    """Configuration for deployable grid fin arrays."""
    # Geometry
    span: float = 1.5           # Extended span (meters)
    chord: float = 0.8          # Fin chord
    grid_density: int = 4       # Grid cells per fin
    
    # Deployment
    deploy_time: float = 0.3    # Seconds to full deployment
    retract_time: float = 0.2   # Faster retraction
    
    # Aerodynamics
    cd_retracted: float = 0.1   # Drag coefficient retracted
    cd_deployed: float = 1.8    # Drag coefficient deployed (HIGH!)
    cl_max: float = 1.2         # Max lift coefficient
    
    # Actuation limits
    deflection_rate: float = 90.0   # Degrees per second
    max_deflection: float = 45.0    # Max fin deflection


@dataclass
class BolaHead:
    """
    Consolidated TAB mass - the "striking head" of the nunchaku.
    
    When TABs merge into bola mode, their properties combine:
    - Mass: Sum of all TAB masses
    - Inertia: Combined moment of inertia
    - Airfoils: All grid fins available for control
    """
    # Physical properties (combined from TABs)
    mass: float = 0.0               # kg
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    angular_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    
    # Moment of inertia tensor (simplified as scalar for now)
    inertia: float = 0.0            # kg*m^2
    
    # Cable connection
    cable_attachment: np.ndarray = field(default_factory=lambda: np.zeros(3))
    cable_tension: float = 0.0      # Newtons
    
    # Deployable surfaces
    grid_fins: Dict[str, 'DeployableGridFin'] = field(default_factory=dict)
    
    # State
    is_consolidated: bool = False
    contributing_tabs: List[str] = field(default_factory=list)


@dataclass
class DeployableGridFin:
    """
    A single deployable grid fin surface.
    
    Like SpaceX Falcon 9 fins - titanium lattice that provides
    massive drag/lift for precise trajectory control.
    """
    # Identity
    fin_id: str = ""
    position_local: np.ndarray = field(default_factory=lambda: np.zeros(3))
    
    # Deployment state
    deploy_state: AirfoilDeployState = AirfoilDeployState.RETRACTED
    deploy_fraction: float = 0.0    # 0 = retracted, 1 = deployed
    
    # Control
    deflection: float = 0.0         # Current deflection angle (degrees)
    deflection_cmd: float = 0.0     # Commanded deflection
    
    # Configuration
    config: GridFinConfig = field(default_factory=GridFinConfig)
    
    def update(self, dt: float):
        """Update deployment and deflection."""
        # Deployment transition
        if self.deploy_state == AirfoilDeployState.DEPLOYING:
            self.deploy_fraction += dt / self.config.deploy_time
            if self.deploy_fraction >= 1.0:
                self.deploy_fraction = 1.0
                self.deploy_state = AirfoilDeployState.DEPLOYED
        elif self.deploy_state == AirfoilDeployState.RETRACTED and self.deploy_fraction > 0:
            self.deploy_fraction -= dt / self.config.retract_time
            self.deploy_fraction = max(0, self.deploy_fraction)
        
        # Deflection servo
        deflection_error = self.deflection_cmd - self.deflection
        max_delta = self.config.deflection_rate * dt
        if abs(deflection_error) > max_delta:
            self.deflection += np.sign(deflection_error) * max_delta
        else:
            self.deflection = self.deflection_cmd
        
        # Clamp
        self.deflection = np.clip(self.deflection, 
                                   -self.config.max_deflection,
                                   self.config.max_deflection)
    
    def get_drag_coefficient(self) -> float:
        """Get current drag coefficient based on deployment."""
        cd_range = self.config.cd_deployed - self.config.cd_retracted
        return self.config.cd_retracted + cd_range * self.deploy_fraction
    
    def get_lift_coefficient(self, alpha: float) -> float:
        """Get lift coefficient based on deployment and angle of attack."""
        # Simplified: linear up to stall
        cl = self.config.cl_max * self.deploy_fraction * np.sin(2 * np.radians(alpha))
        return cl
    
    def deploy(self):
        """Command deployment."""
        self.deploy_state = AirfoilDeployState.DEPLOYING
    
    def retract(self):
        """Command retraction."""
        self.deploy_state = AirfoilDeployState.RETRACTED


class SlingshotController:
    """
    Controls slingshot/nunchaku dynamics.
    
    DreamerV3 learns to:
    1. Time the consolidation of TABs into bola
    2. Execute the swing motion (circular acceleration)
    3. Release at optimal angle for max terminal velocity
    4. Deploy grid fins for post-release steering
    
    The key is MOMENTUM WHIP physics:
    - Buzzard rotation creates centripetal acceleration
    - Angular momentum transfers to bola head
    - Release converts angular to linear velocity
    - Grid fins steer to target
    """
    
    def __init__(self, cable_length: float = 50.0):
        self.cable_length = cable_length
        self.state = SlingshotState.DISPERSED
        
        # Bola configuration
        self.bola = BolaHead()
        self.consolidated_cable_length = cable_length * 1.5  # Longer when consolidated
        
        # Swing dynamics
        self.swing_angular_velocity = 0.0   # rad/s
        self.swing_angle = 0.0              # radians
        self.swing_plane_normal = np.array([0, 0, 1])  # Swing in XY plane
        
        # Release targeting
        self.release_target: Optional[np.ndarray] = None
        self.optimal_release_angle: float = 0.0
        
        # Physics constants
        self.max_swing_rate = 4 * np.pi  # Max 2 rotations per second
        self.swing_damping = 0.05        # Energy loss per revolution
        
        # Airfoiled Buzzard properties
        self.buzzard_airfoil_deployed = False
        self.buzzard_lift_coefficient = 0.0
    
    def consolidate_tabs(self, tab_states: Dict[str, dict]) -> BolaHead:
        """
        Merge all TABs into a single bola head.
        
        Combines mass, creates grid fin array from combined airfoils.
        """
        self.state = SlingshotState.CONSOLIDATING
        
        total_mass = 0.0
        combined_position = np.zeros(3)
        combined_velocity = np.zeros(3)
        tab_count = 0
        
        for tab_id, tab in tab_states.items():
            if tab.get('is_attached', True):
                total_mass += tab.get('mass', 5.0)
                combined_position += tab.get('position', np.zeros(3))
                combined_velocity += tab.get('velocity', np.zeros(3))
                self.bola.contributing_tabs.append(tab_id)
                tab_count += 1
        
        if tab_count > 0:
            combined_position /= tab_count
            combined_velocity /= tab_count
        
        self.bola.mass = total_mass
        self.bola.position = combined_position
        self.bola.velocity = combined_velocity
        self.bola.inertia = total_mass * 0.5  # Approximate
        self.bola.is_consolidated = True
        
        # Create grid fins from contributing TABs
        fin_positions = [
            ("FIN_UP", np.array([0, 0, 0.5])),
            ("FIN_DOWN", np.array([0, 0, -0.5])),
            ("FIN_LEFT", np.array([-0.5, 0, 0])),
            ("FIN_RIGHT", np.array([0.5, 0, 0])),
        ]
        
        for fin_id, pos in fin_positions:
            self.bola.grid_fins[fin_id] = DeployableGridFin(
                fin_id=fin_id,
                position_local=pos
            )
        
        self.state = SlingshotState.BOLA_READY
        return self.bola
    
    def begin_swing(self, 
                    buzzard_position: np.ndarray,
                    initial_angular_rate: float = 1.0,
                    swing_plane: np.ndarray = None):
        """
        Begin nunchaku swing motion.
        
        The Buzzard rotates, pulling the bola in a circular arc.
        Angular momentum builds up in the bola head.
        """
        if self.state != SlingshotState.BOLA_READY:
            return False
        
        self.state = SlingshotState.SWINGING
        self.swing_angular_velocity = initial_angular_rate
        
        if swing_plane is not None:
            self.swing_plane_normal = swing_plane / np.linalg.norm(swing_plane)
        
        # Initial angle from current position
        delta = self.bola.position - buzzard_position
        self.swing_angle = np.arctan2(delta[1], delta[0])
        
        return True
    
    def accelerate_swing(self, torque: float, dt: float):
        """
        Apply torque to accelerate the swing.
        
        DreamerV3 controls the torque to build angular momentum.
        """
        if self.state != SlingshotState.SWINGING:
            return
        
        # Angular acceleration = torque / moment_of_inertia
        # For the bola on a cable: I = m * r^2
        moment = self.bola.mass * self.consolidated_cable_length ** 2
        angular_accel = torque / moment
        
        # Update angular velocity with damping
        self.swing_angular_velocity += angular_accel * dt
        self.swing_angular_velocity *= (1 - self.swing_damping * dt)
        
        # Clamp to max
        self.swing_angular_velocity = np.clip(
            self.swing_angular_velocity,
            -self.max_swing_rate,
            self.max_swing_rate
        )
        
        # Update angle
        self.swing_angle += self.swing_angular_velocity * dt
    
    def update_swing_physics(self, 
                             buzzard_pos: np.ndarray,
                             buzzard_vel: np.ndarray,
                             dt: float) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Update bola position and velocity during swing.
        
        Returns: (bola_position, bola_velocity, cable_tension)
        """
        if self.state not in [SlingshotState.SWINGING, SlingshotState.BOLA_READY]:
            return self.bola.position, self.bola.velocity, 0.0
        
        # Bola position relative to Buzzard
        # Circular motion in the swing plane
        r = self.consolidated_cable_length
        
        # Create rotation in swing plane
        # Default: XY plane (swing_plane_normal = [0,0,1])
        x = r * np.cos(self.swing_angle)
        y = r * np.sin(self.swing_angle)
        z = 0
        
        # Rotate to swing plane if not XY
        if not np.allclose(self.swing_plane_normal, [0, 0, 1]):
            # Rodrigues rotation - rotate from XY to swing plane
            pass  # TODO: implement full 3D swing planes
        
        relative_pos = np.array([x, y, z])
        self.bola.position = buzzard_pos + relative_pos
        
        # Bola velocity = buzzard velocity + tangential swing velocity
        tangent = np.array([-np.sin(self.swing_angle), 
                            np.cos(self.swing_angle), 
                            0])
        swing_speed = self.swing_angular_velocity * r
        self.bola.velocity = buzzard_vel + tangent * swing_speed
        
        # Cable tension = centripetal force = m * v^2 / r
        centripetal_accel = (self.swing_angular_velocity ** 2) * r
        self.bola.cable_tension = self.bola.mass * centripetal_accel
        
        return self.bola.position, self.bola.velocity, self.bola.cable_tension
    
    def compute_optimal_release_angle(self, target: np.ndarray) -> float:
        """
        Compute optimal release angle to hit target.
        
        DreamerV3 learns this, but we provide the analytical solution
        for comparison.
        """
        # Direction to target from Buzzard
        # Release velocity will be tangent to the swing circle
        # Need to find angle where tangent points at target
        
        # Simplified: release when tangent vector points toward target
        to_target = target - self.bola.position
        to_target_2d = to_target[:2]  # XY plane
        target_angle = np.arctan2(to_target_2d[1], to_target_2d[0])
        
        # Tangent at angle θ points at θ + π/2 (counter-clockwise)
        # So release at θ where θ + π/2 = target_angle
        # θ = target_angle - π/2
        self.optimal_release_angle = target_angle - np.pi / 2
        self.release_target = target
        
        return self.optimal_release_angle
    
    def release(self) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Release the bola - cut the cable!
        
        Returns: (position, velocity, speed) at release
        """
        if self.state != SlingshotState.SWINGING:
            return self.bola.position, self.bola.velocity, 0.0
        
        self.state = SlingshotState.RELEASING
        
        # Store release parameters
        release_pos = self.bola.position.copy()
        release_vel = self.bola.velocity.copy()
        release_speed = np.linalg.norm(release_vel)
        
        # Bola is now ballistic
        self.bola.cable_tension = 0.0
        self.state = SlingshotState.BALLISTIC
        
        print(f"[SLINGSHOT] RELEASE! Speed: {release_speed:.1f} m/s")
        print(f"[SLINGSHOT] Angle: {np.degrees(self.swing_angle):.1f}°")
        
        return release_pos, release_vel, release_speed
    
    def deploy_grid_fins(self):
        """Deploy all grid fins for post-release steering."""
        if self.state not in [SlingshotState.BALLISTIC, SlingshotState.SAILING]:
            return
        
        self.state = SlingshotState.SAILING
        for fin in self.bola.grid_fins.values():
            fin.deploy()
        
        print("[SLINGSHOT] Grid fins DEPLOYED - sailing mode")
    
    def update_ballistic(self, dt: float, gravity: float = 9.81) -> np.ndarray:
        """
        Update bola in ballistic/sailing flight.
        
        Grid fins provide drag and steering.
        """
        if self.state not in [SlingshotState.BALLISTIC, SlingshotState.SAILING]:
            return self.bola.position
        
        # Gravity
        accel = np.array([0, 0, -gravity])
        
        # Aerodynamic forces
        vel = self.bola.velocity
        speed = np.linalg.norm(vel)
        
        if speed > 0.1:
            vel_dir = vel / speed
            rho = 1.225  # Air density
            
            # Sum grid fin contributions
            total_drag = 0.0
            total_lift = np.zeros(3)
            
            for fin in self.bola.grid_fins.values():
                fin.update(dt)
                cd = fin.get_drag_coefficient()
                
                # Approximate fin area
                fin_area = fin.config.span * fin.config.chord * fin.deploy_fraction
                
                # Drag force magnitude
                drag_force = 0.5 * rho * speed**2 * cd * fin_area
                total_drag += drag_force
                
                # Lift from deflection (simplified)
                if abs(fin.deflection) > 0.1:
                    cl = fin.get_lift_coefficient(fin.deflection)
                    lift_force = 0.5 * rho * speed**2 * cl * fin_area
                    
                    # Lift perpendicular to velocity
                    # Direction based on fin orientation
                    if "UP" in fin.fin_id or "DOWN" in fin.fin_id:
                        lift_dir = np.cross(vel_dir, [1, 0, 0])
                    else:
                        lift_dir = np.cross(vel_dir, [0, 0, 1])
                    
                    if np.linalg.norm(lift_dir) > 0.01:
                        lift_dir = lift_dir / np.linalg.norm(lift_dir)
                        total_lift += lift_dir * lift_force * np.sign(fin.deflection)
            
            # Apply forces
            drag_accel = -vel_dir * total_drag / self.bola.mass
            lift_accel = total_lift / self.bola.mass
            accel += drag_accel + lift_accel
        
        # Integrate
        self.bola.velocity += accel * dt
        self.bola.position += self.bola.velocity * dt
        
        return self.bola.position
    
    def steer_to_target(self, target: np.ndarray):
        """
        Command grid fins to steer toward target.
        
        DreamerV3 calls this with pursuit guidance commands.
        """
        if self.state != SlingshotState.SAILING:
            return
        
        # Vector to target
        to_target = target - self.bola.position
        to_target = to_target / (np.linalg.norm(to_target) + 0.001)
        
        # Current heading
        vel = self.bola.velocity
        speed = np.linalg.norm(vel)
        if speed < 0.1:
            return
        
        heading = vel / speed
        
        # Error (cross product gives rotation axis)
        error = np.cross(heading, to_target)
        
        # Command fins based on error components
        # Vertical error -> UP/DOWN fins
        # Horizontal error -> LEFT/RIGHT fins
        
        if "FIN_UP" in self.bola.grid_fins:
            self.bola.grid_fins["FIN_UP"].deflection_cmd = -error[0] * 30
            self.bola.grid_fins["FIN_DOWN"].deflection_cmd = error[0] * 30
        
        if "FIN_LEFT" in self.bola.grid_fins:
            self.bola.grid_fins["FIN_LEFT"].deflection_cmd = error[2] * 30
            self.bola.grid_fins["FIN_RIGHT"].deflection_cmd = -error[2] * 30
    
    def disperse_tabs(self):
        """Return to standard dispersed TAB mode."""
        self.state = SlingshotState.DISPERSED
        self.bola = BolaHead()
        self.swing_angular_velocity = 0.0
        self.swing_angle = 0.0
        print("[SLINGSHOT] Dispersed - returning to standard TAB mode")
    
    def get_release_velocity(self) -> float:
        """
        Calculate theoretical max release velocity.
        
        v = ω * r where ω is angular velocity, r is cable length
        """
        return abs(self.swing_angular_velocity) * self.consolidated_cable_length
    
    def get_state_for_dreamer(self) -> Dict:
        """
        Get state representation for DreamerV3 observation.
        """
        return {
            'slingshot_state': self.state.value,
            'swing_angle': self.swing_angle,
            'swing_angular_velocity': self.swing_angular_velocity,
            'bola_position': self.bola.position.tolist(),
            'bola_velocity': self.bola.velocity.tolist(),
            'bola_speed': np.linalg.norm(self.bola.velocity),
            'cable_tension': self.bola.cable_tension,
            'release_velocity': self.get_release_velocity(),
            'optimal_release_angle': self.optimal_release_angle,
            'is_consolidated': self.bola.is_consolidated,
            'grid_fins_deployed': any(
                f.deploy_state == AirfoilDeployState.DEPLOYED 
                for f in self.bola.grid_fins.values()
            ),
        }


class AirfoiledBuzzard:
    """
    Buzzard with deployable lifting surfaces.
    
    The Buzzard itself becomes a flying wing / lifting body with
    configurable airfoil surfaces for enhanced maneuverability.
    
    Like a manta ray that can reshape its wings for different
    flight regimes.
    """
    
    def __init__(self):
        # Lifting body properties
        self.body_lift_coefficient = 0.3    # Base body provides some lift
        self.reference_area = 4.0           # m^2
        
        # Deployable wing surfaces
        self.wing_extension = 0.0           # 0 = retracted, 1 = full span
        self.max_wingspan = 8.0             # meters at full extension
        self.min_wingspan = 2.0             # meters retracted
        
        # Airfoil configuration
        self.angle_of_attack = 0.0          # degrees
        self.bank_angle = 0.0               # degrees
        
        # Grid fin arrays (like Falcon 9)
        self.grid_fins = {
            'nose': DeployableGridFin(fin_id='nose', position_local=np.array([2, 0, 0])),
            'tail': DeployableGridFin(fin_id='tail', position_local=np.array([-2, 0, 0])),
            'left': DeployableGridFin(fin_id='left', position_local=np.array([0, -1.5, 0])),
            'right': DeployableGridFin(fin_id='right', position_local=np.array([0, 1.5, 0])),
        }
        
        # Corkscrew propulsion integration
        self.thrust_vector_angle = 0.0      # Vectored thrust for agility
    
    def get_current_wingspan(self) -> float:
        """Get current wingspan based on extension."""
        return self.min_wingspan + (self.max_wingspan - self.min_wingspan) * self.wing_extension
    
    def extend_wings(self, target_extension: float, dt: float, rate: float = 2.0):
        """Extend/retract wings toward target."""
        target_extension = np.clip(target_extension, 0, 1)
        delta = target_extension - self.wing_extension
        max_delta = rate * dt
        
        if abs(delta) > max_delta:
            self.wing_extension += np.sign(delta) * max_delta
        else:
            self.wing_extension = target_extension
    
    def compute_lift_drag(self, 
                          velocity: np.ndarray,
                          air_density: float = 1.225) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute lift and drag forces on the airfoiled Buzzard.
        
        Returns: (lift_vector, drag_vector)
        """
        speed = np.linalg.norm(velocity)
        if speed < 0.1:
            return np.zeros(3), np.zeros(3)
        
        vel_dir = velocity / speed
        q = 0.5 * air_density * speed ** 2  # Dynamic pressure
        
        # Wing area scales with extension
        current_span = self.get_current_wingspan()
        chord = 1.5  # Average chord
        wing_area = current_span * chord + self.reference_area
        
        # Lift coefficient (simplified)
        alpha = np.radians(self.angle_of_attack)
        cl = self.body_lift_coefficient + 2 * np.pi * alpha * self.wing_extension
        cl = np.clip(cl, -1.5, 1.5)
        
        # Drag coefficient (induced + parasitic)
        cd_parasitic = 0.03 + 0.02 * self.wing_extension  # More drag with wings out
        cd_induced = cl ** 2 / (np.pi * (current_span / chord) * 0.8)  # Oswald efficiency
        cd = cd_parasitic + cd_induced
        
        # Force magnitudes
        lift_mag = q * wing_area * cl
        drag_mag = q * wing_area * cd
        
        # Lift perpendicular to velocity, in the bank plane
        # Default up vector, rotated by bank angle
        up = np.array([0, 0, 1])
        bank_rad = np.radians(self.bank_angle)
        
        # Rotate up vector around velocity axis
        # (simplified - proper would use quaternions)
        right = np.cross(vel_dir, up)
        if np.linalg.norm(right) > 0.01:
            right = right / np.linalg.norm(right)
            lift_dir = np.cos(bank_rad) * up + np.sin(bank_rad) * right
        else:
            lift_dir = up
        
        lift_vector = lift_dir * lift_mag
        drag_vector = -vel_dir * drag_mag
        
        return lift_vector, drag_vector
    
    def deploy_all_fins(self):
        """Deploy all grid fins for maximum control."""
        for fin in self.grid_fins.values():
            fin.deploy()
    
    def retract_all_fins(self):
        """Retract grid fins for minimum drag."""
        for fin in self.grid_fins.values():
            fin.retract()
    
    def update(self, dt: float):
        """Update all fin states."""
        for fin in self.grid_fins.values():
            fin.update(dt)


# =============================================================================
# DREAMERV3 ACTION SPACE FOR SLINGSHOT MODE
# =============================================================================

SLINGSHOT_ACTIONS = {
    # Mode transitions
    'consolidate': 'Begin TAB consolidation into bola',
    'disperse': 'Return to standard TAB mode',
    
    # Swing control
    'begin_swing': 'Start nunchaku swing motion',
    'swing_torque': 'Apply torque to accelerate swing (-1 to 1)',
    'set_swing_plane': 'Set swing plane normal vector',
    
    # Release
    'release': 'Cut cable, release bola',
    
    # Grid fin control
    'deploy_fins': 'Deploy grid fins for steering',
    'retract_fins': 'Retract grid fins',
    'fin_deflections': 'Set fin deflection commands [up, down, left, right]',
    
    # Buzzard airfoil
    'wing_extension': 'Set Buzzard wing extension (0-1)',
    'angle_of_attack': 'Set Buzzard AoA',
    'bank_angle': 'Set Buzzard bank angle',
}

# Observation additions for slingshot mode
SLINGSHOT_OBSERVATIONS = {
    'slingshot_state': 'Current slingshot state enum',
    'swing_angle': 'Current swing angle (radians)',
    'swing_angular_velocity': 'Swing rate (rad/s)',
    'bola_position': 'Bola position [x, y, z]',
    'bola_velocity': 'Bola velocity [vx, vy, vz]',
    'bola_speed': 'Bola speed magnitude',
    'cable_tension': 'Current cable tension (N)',
    'release_velocity': 'Theoretical release velocity',
    'optimal_release_angle': 'Computed optimal release angle',
    'grid_fins_deployed': 'Whether fins are deployed',
    'wing_extension': 'Buzzard wing extension state',
}


if __name__ == "__main__":
    # Test slingshot dynamics
    print("=== Slingshot / Nunchaku Dynamics Test ===\n")
    
    controller = SlingshotController(cable_length=50.0)
    
    # Simulate some TABs
    tabs = {
        'UP': {'position': np.array([0, 0, 60]), 'velocity': np.zeros(3), 'mass': 5.0, 'is_attached': True},
        'DOWN': {'position': np.array([0, 0, -10]), 'velocity': np.zeros(3), 'mass': 5.0, 'is_attached': True},
        'LEFT': {'position': np.array([-30, 0, 25]), 'velocity': np.zeros(3), 'mass': 5.0, 'is_attached': True},
        'RIGHT': {'position': np.array([30, 0, 25]), 'velocity': np.zeros(3), 'mass': 5.0, 'is_attached': True},
    }
    
    # Consolidate
    print("Consolidating TABs into BOLA...")
    bola = controller.consolidate_tabs(tabs)
    print(f"  Bola mass: {bola.mass} kg")
    print(f"  Contributing TABs: {bola.contributing_tabs}")
    print(f"  Grid fins: {list(bola.grid_fins.keys())}")
    
    # Begin swing
    buzzard_pos = np.array([0, 0, 1000])
    buzzard_vel = np.array([0, 50, 0])
    
    print("\nBeginning swing...")
    controller.begin_swing(buzzard_pos, initial_angular_rate=1.0)
    
    # Simulate swing with torque
    print("\nSimulating swing acceleration...")
    dt = 0.02
    for i in range(100):
        controller.accelerate_swing(torque=5000.0, dt=dt)
        pos, vel, tension = controller.update_swing_physics(buzzard_pos, buzzard_vel, dt)
        
        if i % 25 == 0:
            speed = np.linalg.norm(vel)
            print(f"  t={i*dt:.1f}s: ω={controller.swing_angular_velocity:.2f} rad/s, "
                  f"v={speed:.1f} m/s, T={tension:.0f} N")
    
    # Compute optimal release
    target = np.array([500, 500, 900])
    opt_angle = controller.compute_optimal_release_angle(target)
    print(f"\nOptimal release angle for target: {np.degrees(opt_angle):.1f}°")
    
    # Release!
    print("\nRELEASING BOLA!")
    rel_pos, rel_vel, rel_speed = controller.release()
    print(f"  Release speed: {rel_speed:.1f} m/s")
    print(f"  Release velocity: [{rel_vel[0]:.1f}, {rel_vel[1]:.1f}, {rel_vel[2]:.1f}]")
    
    # Deploy fins and sail
    controller.deploy_grid_fins()
    
    print("\nSailing to target...")
    for i in range(50):
        controller.steer_to_target(target)
        pos = controller.update_ballistic(dt)
        
        if i % 10 == 0:
            dist = np.linalg.norm(pos - target)
            speed = np.linalg.norm(controller.bola.velocity)
            print(f"  t={i*dt:.1f}s: dist={dist:.1f}m, speed={speed:.1f} m/s")
    
    print("\n=== Test Complete ===")
