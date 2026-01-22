"""
Articulated Blade-Tether System
================================

The links between nodes are NOT passive cables - they are:
  - Flat-sided airfoil/hydrofoil segments
  - Articulated joints at each node
  - Pitch/yaw controllable by the quine agents
  - Act as propeller blades when rotated

Each segment catches the medium (air/water) and generates:
  - Lift (perpendicular to flow)
  - Drag (parallel to flow) 
  - Thrust (when pitched correctly during rotation)

The nodes at each end cooperatively control the blade angle,
turning the entire lattice into a distributed propeller system.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict
from enum import Enum, auto


class BladeProfile(Enum):
    """Cross-section profile of the blade segment"""
    FLAT_PLATE = auto()      # Simple flat surface (high drag, simple)
    SYMMETRIC = auto()       # NACA 0012-style (no camber, reversible)
    CAMBERED = auto()        # NACA 2412-style (more lift one direction)
    PADDLE = auto()          # Wide flat paddle (like oar)
    HELICAL = auto()         # Twisted along length (like propeller blade)


class JointType(Enum):
    """Type of articulation at node connection"""
    FIXED = auto()           # No articulation (rigid)
    PITCH = auto()           # Rotate around span axis (angle of attack)
    YAW = auto()             # Rotate around chord axis
    UNIVERSAL = auto()       # Both pitch and yaw (gimbal)
    FLEXIBLE = auto()        # Continuous bend (like fish fin)


@dataclass
class BladeGeometry:
    """Physical geometry of a blade segment"""
    length: float = 3.0           # Span length (m) - distance between nodes
    chord: float = 0.15           # Chord width (m) - blade width
    thickness: float = 0.02       # Maximum thickness (m)
    
    profile: BladeProfile = BladeProfile.FLAT_PLATE
    
    # Twist distribution (for helical blades)
    root_twist: float = 0.0       # Twist at parent node (radians)
    tip_twist: float = 0.0        # Twist at child node (radians)
    
    @property
    def aspect_ratio(self) -> float:
        """Span / chord - higher = more efficient, lower = more maneuverable"""
        return self.length / self.chord
    
    @property
    def surface_area(self) -> float:
        """Planform area (m²)"""
        return self.length * self.chord
    
    @property
    def wetted_area(self) -> float:
        """Total surface area exposed to flow (both sides)"""
        return 2 * self.surface_area


@dataclass
class BladeState:
    """Current kinematic state of a blade segment"""
    # Articulation angles (controlled by nodes)
    pitch_angle: float = 0.0      # Angle of attack (radians)
    yaw_angle: float = 0.0        # Twist around length axis
    
    # Rates (for damping)
    pitch_rate: float = 0.0
    yaw_rate: float = 0.0
    
    # Forces computed from flow
    lift: np.ndarray = field(default_factory=lambda: np.zeros(3))
    drag: np.ndarray = field(default_factory=lambda: np.zeros(3))
    moment: np.ndarray = field(default_factory=lambda: np.zeros(3))
    
    # Center of pressure (along blade, 0=root, 1=tip)
    cp_location: float = 0.25


@dataclass
class ArticulatedJoint:
    """Joint connecting blade to node"""
    joint_type: JointType = JointType.UNIVERSAL
    
    # Range of motion limits (radians)
    pitch_min: float = -np.pi / 3   # -60°
    pitch_max: float = np.pi / 3    # +60°
    yaw_min: float = -np.pi / 4     # -45°
    yaw_max: float = np.pi / 4      # +45°
    
    # Actuator properties
    max_pitch_torque: float = 5.0   # Nm
    max_yaw_torque: float = 3.0     # Nm
    
    # Stiffness/damping (for passive stability)
    pitch_stiffness: float = 10.0   # Nm/rad
    pitch_damping: float = 2.0      # Nm·s/rad
    yaw_stiffness: float = 8.0
    yaw_damping: float = 1.5
    
    def clamp_pitch(self, angle: float) -> float:
        return np.clip(angle, self.pitch_min, self.pitch_max)
    
    def clamp_yaw(self, angle: float) -> float:
        return np.clip(angle, self.yaw_min, self.yaw_max)


@dataclass
class BladeTether:
    """
    An articulated blade-tether segment between two nodes.
    
    This is the fundamental propulsion element - a controllable
    airfoil/hydrofoil that generates thrust when rotated.
    """
    segment_id: str
    
    # Endpoint node IDs
    parent_node_id: str
    child_node_id: str
    
    # Geometry
    geometry: BladeGeometry = field(default_factory=BladeGeometry)
    
    # Articulation
    parent_joint: ArticulatedJoint = field(default_factory=ArticulatedJoint)
    child_joint: ArticulatedJoint = field(default_factory=ArticulatedJoint)
    
    # Current state
    state: BladeState = field(default_factory=BladeState)
    
    # Physical properties
    mass: float = 0.1             # kg (blade mass)
    inertia: float = 0.01         # kg·m² (about pitch axis)
    
    # Medium properties (set based on environment)
    fluid_density: float = 1.225  # kg/m³ (air at sea level)
    
    def set_medium(self, medium: str) -> None:
        """Set fluid properties based on operating medium"""
        if medium.lower() == 'air':
            self.fluid_density = 1.225
        elif medium.lower() == 'water':
            self.fluid_density = 1000.0
        elif medium.lower() == 'vacuum':
            self.fluid_density = 0.0
        else:
            # Custom density
            self.fluid_density = float(medium)
    
    def compute_forces(self,
                       parent_pos: np.ndarray,
                       child_pos: np.ndarray,
                       parent_vel: np.ndarray,
                       child_vel: np.ndarray,
                       freestream: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute aerodynamic/hydrodynamic forces on the blade.
        
        Args:
            parent_pos: Position of parent node
            child_pos: Position of child node  
            parent_vel: Velocity of parent node
            child_vel: Velocity of child node
            freestream: Freestream flow velocity (if any)
        
        Returns:
            (force_on_parent, force_on_child) - forces to apply to each node
        """
        if freestream is None:
            freestream = np.zeros(3)
        
        # Blade geometry in world frame
        span_vec = child_pos - parent_pos
        span_length = np.linalg.norm(span_vec)
        if span_length < 0.001:
            return np.zeros(3), np.zeros(3)
        
        span_dir = span_vec / span_length
        
        # Average velocity at blade center
        blade_vel = 0.5 * (parent_vel + child_vel)
        
        # Relative flow velocity
        rel_vel = freestream - blade_vel
        rel_speed = np.linalg.norm(rel_vel)
        
        if rel_speed < 0.01 or self.fluid_density < 0.001:
            self.state.lift = np.zeros(3)
            self.state.drag = np.zeros(3)
            return np.zeros(3), np.zeros(3)
        
        flow_dir = rel_vel / rel_speed
        
        # Build blade coordinate frame
        # Normal = perpendicular to blade surface (affected by pitch)
        # Chord direction = perpendicular to span, in flow plane
        
        # Base normal (before pitch)
        chord_dir = np.cross(span_dir, flow_dir)
        chord_norm = np.linalg.norm(chord_dir)
        if chord_norm < 0.01:
            # Flow parallel to span - minimal forces
            chord_dir = np.array([1, 0, 0]) if abs(span_dir[0]) < 0.9 else np.array([0, 1, 0])
        else:
            chord_dir = chord_dir / chord_norm
        
        normal_base = np.cross(chord_dir, span_dir)
        normal_base = normal_base / np.linalg.norm(normal_base)
        
        # Apply pitch rotation
        pitch = self.state.pitch_angle
        cos_p, sin_p = np.cos(pitch), np.sin(pitch)
        normal = cos_p * normal_base + sin_p * chord_dir
        chord_rotated = -sin_p * normal_base + cos_p * chord_dir
        
        # Angle of attack
        flow_component_normal = np.dot(flow_dir, normal)
        flow_component_chord = np.dot(flow_dir, chord_rotated)
        alpha = np.arctan2(flow_component_normal, abs(flow_component_chord))
        
        # Aerodynamic coefficients (simplified flat plate model)
        # Real implementation would use lookup tables or panel methods
        
        if self.geometry.profile == BladeProfile.FLAT_PLATE:
            # Flat plate: CL ≈ 2π·sin(α)·cos(α), CD ≈ 2·sin²(α) + 0.02
            cl = 2 * np.pi * np.sin(alpha) * np.cos(alpha)
            cd = 2 * np.sin(alpha)**2 + 0.02
            
        elif self.geometry.profile == BladeProfile.SYMMETRIC:
            # Symmetric airfoil (NACA 0012-like)
            # Linear region + stall
            alpha_stall = np.radians(15)
            if abs(alpha) < alpha_stall:
                cl = 5.7 * alpha  # ~2π per radian in linear region
                cd = 0.01 + 0.05 * alpha**2
            else:
                # Post-stall
                cl = 1.2 * np.sign(alpha) * np.sin(2 * alpha)
                cd = 0.5 * np.sin(alpha)**2 + 0.02
                
        elif self.geometry.profile == BladeProfile.PADDLE:
            # Wide paddle - high drag, moderate lift
            cl = 1.5 * np.sin(2 * alpha)
            cd = 1.2 * np.sin(alpha)**2 + 0.1
            
        else:
            # Default
            cl = 2 * np.pi * np.sin(alpha) * np.cos(alpha)
            cd = 2 * np.sin(alpha)**2 + 0.02
        
        # Dynamic pressure
        q = 0.5 * self.fluid_density * rel_speed**2
        
        # Reference area
        S = self.geometry.surface_area
        
        # Lift and drag magnitudes
        L_mag = q * S * cl
        D_mag = q * S * cd
        
        # Lift direction (perpendicular to flow and span)
        lift_dir = np.cross(flow_dir, span_dir)
        lift_norm = np.linalg.norm(lift_dir)
        if lift_norm > 0.01:
            lift_dir = lift_dir / lift_norm
        else:
            lift_dir = normal
        
        # Force vectors
        lift = L_mag * lift_dir * np.sign(np.dot(normal, lift_dir))
        drag = D_mag * flow_dir
        
        self.state.lift = lift
        self.state.drag = drag
        
        # Total force
        total_force = lift + drag
        
        # Distribute to nodes based on center of pressure
        cp = self.state.cp_location  # 0 = parent, 1 = child
        force_on_parent = total_force * (1 - cp)
        force_on_child = total_force * cp
        
        return force_on_parent, force_on_child
    
    def compute_thrust(self,
                       rotation_axis: np.ndarray,
                       angular_velocity: float,
                       blade_center: np.ndarray,
                       freestream: np.ndarray = None) -> np.ndarray:
        """
        Compute thrust generated when blade rotates around an axis.
        
        This is the propeller effect - blade pitch + rotation = thrust.
        
        Args:
            rotation_axis: Axis of rotation (normalized)
            angular_velocity: Rotation rate (rad/s)
            blade_center: Position of blade center
            freestream: Incoming flow velocity
        
        Returns:
            Thrust force vector (along rotation axis ideally)
        """
        if freestream is None:
            freestream = np.zeros(3)
        
        # Blade velocity from rotation
        # v = ω × r (where r is from rotation axis to blade center)
        r = blade_center  # Simplified: assume axis passes through origin
        rotational_vel = angular_velocity * np.cross(rotation_axis, r)
        
        # Total relative velocity
        rel_vel = freestream - rotational_vel
        rel_speed = np.linalg.norm(rel_vel)
        
        if rel_speed < 0.01:
            return np.zeros(3)
        
        # Dynamic pressure
        q = 0.5 * self.fluid_density * rel_speed**2
        S = self.geometry.surface_area
        
        # Thrust coefficient depends on pitch angle and advance ratio
        # Simplified: thrust ∝ pitch angle × velocity²
        pitch = self.state.pitch_angle
        ct = 0.1 * np.sin(pitch) * np.cos(pitch)  # Peak at 45°
        
        # Thrust along rotation axis
        thrust_mag = q * S * ct
        thrust = thrust_mag * rotation_axis
        
        return thrust


class BladeController:
    """
    Controller for blade pitch/yaw based on desired thrust direction.
    
    The quine agents at each node use this to coordinate blade angles.
    """
    
    def __init__(self, blade: BladeTether):
        self.blade = blade
        
        # Control gains
        self.kp_pitch = 5.0
        self.kd_pitch = 1.0
        self.kp_yaw = 3.0
        self.kd_yaw = 0.5
    
    def compute_optimal_pitch(self,
                               desired_thrust_dir: np.ndarray,
                               rotation_axis: np.ndarray,
                               blade_position_angle: float) -> float:
        """
        Compute optimal pitch angle for thrust in desired direction.
        
        Like a helicopter collective/cyclic: varies pitch through rotation
        to achieve thrust in arbitrary direction.
        
        Args:
            desired_thrust_dir: Desired thrust direction (normalized)
            rotation_axis: Current rotation axis
            blade_position_angle: Current angular position in rotation (radians)
        
        Returns:
            Optimal pitch angle (radians)
        """
        # Decompose desired thrust into axial and radial components
        axial = np.dot(desired_thrust_dir, rotation_axis) * rotation_axis
        radial = desired_thrust_dir - axial
        radial_mag = np.linalg.norm(radial)
        
        # Base pitch for axial thrust
        axial_pitch = np.arctan2(np.dot(desired_thrust_dir, rotation_axis), 1.0) * 0.5
        
        if radial_mag < 0.01:
            # Pure axial thrust
            return axial_pitch
        
        # Cyclic pitch variation for radial thrust
        radial_dir = radial / radial_mag
        
        # Phase angle where radial thrust aligns with desired
        # Blade generates thrust perpendicular to its position
        phase_offset = np.arctan2(radial_dir[1], radial_dir[0])
        
        # Sinusoidal pitch variation
        cyclic_amplitude = np.arctan(radial_mag) * 0.5
        cyclic_pitch = cyclic_amplitude * np.sin(blade_position_angle - phase_offset)
        
        return axial_pitch + cyclic_pitch
    
    def update(self,
               desired_thrust: np.ndarray,
               rotation_state: dict,
               dt: float) -> Tuple[float, float]:
        """
        Update blade pitch/yaw commands.
        
        Returns:
            (pitch_command, yaw_command) in radians
        """
        if 'axis' not in rotation_state or 'angle' not in rotation_state:
            return 0.0, 0.0
        
        # Compute optimal pitch
        thrust_mag = np.linalg.norm(desired_thrust)
        if thrust_mag < 0.01:
            target_pitch = 0.0
        else:
            target_pitch = self.compute_optimal_pitch(
                desired_thrust / thrust_mag,
                rotation_state['axis'],
                rotation_state['angle']
            )
        
        # PD control to target
        pitch_error = target_pitch - self.blade.state.pitch_angle
        pitch_cmd = (
            self.kp_pitch * pitch_error -
            self.kd_pitch * self.blade.state.pitch_rate
        )
        
        # Clamp to joint limits
        pitch_cmd = self.blade.parent_joint.clamp_pitch(
            self.blade.state.pitch_angle + pitch_cmd * dt
        )
        
        # Yaw typically follows flow alignment (auto-weathervane)
        yaw_cmd = 0.0  # Simplified
        
        return pitch_cmd, yaw_cmd


class BladeArray:
    """
    Collection of blade-tethers forming a distributed propeller.
    
    Coordinates multiple blades rotating together to generate
    collective thrust like a propeller or rotor.
    """
    
    def __init__(self):
        self.blades: Dict[str, BladeTether] = {}
        self.controllers: Dict[str, BladeController] = {}
    
    def add_blade(self, blade: BladeTether) -> None:
        """Add a blade segment to the array"""
        self.blades[blade.segment_id] = blade
        self.controllers[blade.segment_id] = BladeController(blade)
    
    def set_all_pitch(self, pitch: float) -> None:
        """Set uniform pitch (collective) for all blades"""
        for blade in self.blades.values():
            blade.state.pitch_angle = pitch
    
    def compute_collective_thrust(self,
                                   node_positions: Dict[str, np.ndarray],
                                   node_velocities: Dict[str, np.ndarray],
                                   rotation_axis: np.ndarray,
                                   angular_velocity: float,
                                   freestream: np.ndarray = None) -> np.ndarray:
        """
        Compute total thrust from all blades rotating together.
        """
        total_thrust = np.zeros(3)
        
        for blade in self.blades.values():
            parent_pos = node_positions.get(blade.parent_node_id, np.zeros(3))
            child_pos = node_positions.get(blade.child_node_id, np.zeros(3))
            
            blade_center = 0.5 * (parent_pos + child_pos)
            
            thrust = blade.compute_thrust(
                rotation_axis,
                angular_velocity,
                blade_center,
                freestream
            )
            total_thrust += thrust
        
        return total_thrust


if __name__ == "__main__":
    print("=" * 60)
    print("ARTICULATED BLADE-TETHER DEMO")
    print("=" * 60)
    
    # Create a blade segment
    blade = BladeTether(
        segment_id="BLADE_0",
        parent_node_id="VERT_0",
        child_node_id="VERT_0.GYRO_UP",
        geometry=BladeGeometry(
            length=2.5,
            chord=0.2,
            thickness=0.015,
            profile=BladeProfile.SYMMETRIC
        )
    )
    
    print(f"\n=== BLADE GEOMETRY ===")
    print(f"Span: {blade.geometry.length} m")
    print(f"Chord: {blade.geometry.chord} m")
    print(f"Aspect Ratio: {blade.geometry.aspect_ratio:.1f}")
    print(f"Surface Area: {blade.geometry.surface_area:.4f} m²")
    print(f"Profile: {blade.geometry.profile.name}")
    
    # Simulate forces in airflow
    blade.set_medium('air')
    blade.state.pitch_angle = np.radians(15)  # 15° angle of attack
    
    parent_pos = np.array([0, 0, 0])
    child_pos = np.array([0, 0, -2.5])
    parent_vel = np.array([0, 0, 0])
    child_vel = np.array([0, 0, 0])
    freestream = np.array([10, 0, 0])  # 10 m/s wind
    
    f_parent, f_child = blade.compute_forces(
        parent_pos, child_pos,
        parent_vel, child_vel,
        freestream
    )
    
    print(f"\n=== AERODYNAMIC FORCES ===")
    print(f"Freestream: {freestream} m/s")
    print(f"Pitch angle: {np.degrees(blade.state.pitch_angle):.1f}°")
    print(f"Lift: {blade.state.lift} N")
    print(f"Drag: {blade.state.drag} N")
    print(f"Force on parent: {f_parent} N")
    print(f"Force on child: {f_child} N")
    
    # Simulate propeller thrust (rotating blade)
    print(f"\n=== PROPELLER THRUST ===")
    blade.state.pitch_angle = np.radians(25)  # Blade pitch
    
    for rpm in [100, 500, 1000]:
        omega = rpm * 2 * np.pi / 60  # rad/s
        thrust = blade.compute_thrust(
            rotation_axis=np.array([0, 0, 1]),
            angular_velocity=omega,
            blade_center=np.array([1.25, 0, 0]),  # 1.25m from axis
            freestream=np.array([0, 0, 0])
        )
        print(f"  {rpm:4d} RPM: Thrust = {np.linalg.norm(thrust):.2f} N along {thrust/np.linalg.norm(thrust+1e-9)}")
    
    # Array of 4 blades
    print(f"\n=== 4-BLADE ARRAY ===")
    array = BladeArray()
    
    for i in range(4):
        b = BladeTether(
            segment_id=f"BLADE_{i}",
            parent_node_id="HUB",
            child_node_id=f"TIP_{i}",
            geometry=BladeGeometry(length=2.0, chord=0.15, profile=BladeProfile.SYMMETRIC)
        )
        b.state.pitch_angle = np.radians(20)
        array.add_blade(b)
    
    # Node positions (4 blades in a cross)
    positions = {"HUB": np.array([0, 0, 0])}
    for i, angle in enumerate([0, np.pi/2, np.pi, 3*np.pi/2]):
        positions[f"TIP_{i}"] = np.array([2*np.cos(angle), 2*np.sin(angle), 0])
    
    velocities = {k: np.zeros(3) for k in positions}
    
    total = array.compute_collective_thrust(
        positions, velocities,
        rotation_axis=np.array([0, 0, 1]),
        angular_velocity=50.0,  # ~480 RPM
        freestream=np.zeros(3)
    )
    
    print(f"4 blades at 480 RPM:")
    print(f"  Total thrust: {np.linalg.norm(total):.1f} N")
    print(f"  Direction: {total / (np.linalg.norm(total) + 1e-9)}")
    
    print("\n" + "=" * 60)
    print("Blade-tethers convert rotation into directional thrust")
    print("Each quine node controls pitch angle for collective/cyclic")
    print("=" * 60)
