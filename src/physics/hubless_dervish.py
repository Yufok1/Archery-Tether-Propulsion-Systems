"""
Hubless Dervish - Decentralized Tether Constellation
=====================================================

No central body. No Buzzard. Only the orchestration of ends.

The "vehicle" is an emergent property of connected airfoiled nodes
spinning in coordinated tension. The center of mass is virtual -
it exists only where the tether forces balance.

Key principles:
1. Each node is an autonomous airfoil with tether connections
2. The network topology defines the shape
3. Collective spin creates the propulsion envelope  
4. Cyclic pitch coordination enables omnidirectional thrust
5. The "hull" is the centroid of tension - massless, virtual

Like a bola with no handle. A flying tensegrity.
A constellation that claws through space.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set
from enum import Enum


@dataclass
class AirfoilNode:
    """
    A single airfoiled node in the constellation.
    
    Each node is autonomous but connected to others via tethers.
    It has mass, aerodynamic surfaces, and control authority.
    """
    node_id: str
    
    # Physical state
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    mass: float = 2.0  # kg
    
    # Airfoil properties
    wing_area: float = 0.5        # m²
    aspect_ratio: float = 6.0
    
    # Control surfaces
    pitch: float = 0.0            # Angle of attack (rad)
    roll: float = 0.0             # Bank angle (rad)
    
    # Tether connections (node_id -> tether_length)
    connections: Dict[str, float] = field(default_factory=dict)
    
    # Computed forces this timestep
    aero_force: np.ndarray = field(default_factory=lambda: np.zeros(3))
    tether_force: np.ndarray = field(default_factory=lambda: np.zeros(3))
    
    def add_connection(self, other_id: str, length: float):
        """Add a tether connection to another node."""
        self.connections[other_id] = length


@dataclass 
class TetherLink:
    """A tether connecting two nodes."""
    node_a: str
    node_b: str
    rest_length: float
    stiffness: float = 50000.0    # N/m - MUCH stiffer
    damping: float = 500.0        # Ns/m - more damping
    
    # State
    current_length: float = 0.0
    tension: float = 0.0
    

class HublessDervish:
    """
    A flying constellation of connected airfoils.
    No central body - the vehicle IS the network.
    
    The center of mass emerges from the node positions.
    Propulsion comes from coordinated cyclic pitch.
    The shape morphs based on mission needs.
    """
    
    def __init__(self, 
                 air_density: float = 1.225,
                 gravity: float = 9.81):
        
        self.rho = air_density
        self.g = gravity
        
        # Nodes and tethers
        self.nodes: Dict[str, AirfoilNode] = {}
        self.tethers: Dict[Tuple[str, str], TetherLink] = {}
        
        # Collective state
        self.centroid = np.zeros(3)           # Virtual center
        self.angular_velocity = np.zeros(3)   # Spin rate vector
        self.spin_phase = 0.0
        
        # Control parameters  
        self.target_spin_rate = 2.0           # Hz
        self.collective_pitch = np.radians(5) # Base AoA
        self.cyclic_amplitude = 0.0
        self.cyclic_phase = 0.0
        
    def add_node(self, node_id: str, position: np.ndarray, mass: float = 2.0) -> AirfoilNode:
        """Add an airfoil node to the constellation."""
        node = AirfoilNode(
            node_id=node_id,
            position=position.copy(),
            mass=mass
        )
        self.nodes[node_id] = node
        return node
    
    def connect(self, node_a: str, node_b: str, length: Optional[float] = None):
        """Connect two nodes with a tether."""
        if node_a not in self.nodes or node_b not in self.nodes:
            raise ValueError(f"Nodes must exist: {node_a}, {node_b}")
        
        # Default length = current distance
        if length is None:
            pa = self.nodes[node_a].position
            pb = self.nodes[node_b].position
            length = np.linalg.norm(pb - pa)
        
        # Create bidirectional link
        key = (min(node_a, node_b), max(node_a, node_b))
        tether = TetherLink(node_a=node_a, node_b=node_b, rest_length=length)
        self.tethers[key] = tether
        
        # Record in nodes
        self.nodes[node_a].add_connection(node_b, length)
        self.nodes[node_b].add_connection(node_a, length)
        
    def compute_centroid(self) -> np.ndarray:
        """Compute mass-weighted centroid of constellation."""
        if not self.nodes:
            return np.zeros(3)
        
        total_mass = 0.0
        weighted_pos = np.zeros(3)
        
        for node in self.nodes.values():
            weighted_pos += node.mass * node.position
            total_mass += node.mass
            
        self.centroid = weighted_pos / total_mass if total_mass > 0 else np.zeros(3)
        return self.centroid
    
    def compute_angular_momentum(self) -> np.ndarray:
        """Compute total angular momentum about centroid."""
        L = np.zeros(3)
        c = self.compute_centroid()
        
        for node in self.nodes.values():
            r = node.position - c
            p = node.mass * node.velocity
            L += np.cross(r, p)
            
        return L
    
    def command_thrust(self, direction: np.ndarray, magnitude: float = 1.0):
        """
        Command thrust in a direction by setting cyclic pitch.
        
        The constellation will adjust individual node pitches
        cyclically to create asymmetric lift summing to thrust.
        """
        norm = np.linalg.norm(direction)
        if norm < 0.01:
            self.cyclic_amplitude = 0.0
            return
        
        # Cyclic amplitude proportional to thrust demand
        self.cyclic_amplitude = np.clip(magnitude / 50.0, 0, np.radians(15))
        
        # Cyclic phase: where in rotation to pitch up
        # 90° before desired thrust direction
        d = direction / norm
        self.cyclic_phase = np.arctan2(d[1], d[0]) - np.pi/2
    
    def get_node_pitch(self, node: AirfoilNode) -> float:
        """
        Compute pitch for a node based on its position in constellation.
        
        Pitch = collective + cyclic * sin(angular_position - cyclic_phase)
        """
        # Node's angular position relative to centroid
        r = node.position - self.centroid
        angular_pos = np.arctan2(r[1], r[0]) + self.spin_phase
        
        # Cyclic variation
        cyclic = self.cyclic_amplitude * np.sin(angular_pos - self.cyclic_phase)
        
        return self.collective_pitch + cyclic
    
    def compute_tether_forces(self):
        """Compute spring-damper forces for all tethers."""
        for key, tether in self.tethers.items():
            node_a = self.nodes[tether.node_a]
            node_b = self.nodes[tether.node_b]
            
            # Vector from A to B
            delta = node_b.position - node_a.position
            dist = np.linalg.norm(delta)
            
            if dist < 0.001:
                continue
                
            direction = delta / dist
            
            # Spring force (tension when stretched)
            stretch = dist - tether.rest_length
            spring_force = tether.stiffness * stretch
            
            # Damping force
            rel_vel = node_b.velocity - node_a.velocity
            vel_along = np.dot(rel_vel, direction)
            damp_force = tether.damping * vel_along
            
            # Total tension
            tension = spring_force + damp_force
            tether.tension = max(0, tension)  # Cables can't push
            tether.current_length = dist
            
            # Apply to nodes
            force_vec = tether.tension * direction
            node_a.tether_force += force_vec
            node_b.tether_force -= force_vec
            
            # POSITION CORRECTION: if stretched too far, pull back immediately
            max_stretch = 0.05 * tether.rest_length  # Allow 5% stretch max
            if stretch > max_stretch:
                correction = (stretch - max_stretch) * 0.5  # Split between nodes
                node_a.position += direction * correction
                node_b.position -= direction * correction
                # Also adjust velocities to prevent further stretch
                if vel_along > 0:
                    impulse = vel_along * 0.5 * direction
                    node_a.velocity += impulse
                    node_b.velocity -= impulse
    
    def compute_aero_forces(self, freestream: np.ndarray):
        """Compute aerodynamic forces on all nodes."""
        for node in self.nodes.values():
            # Apparent velocity (freestream + motion relative to air)
            v_apparent = freestream - node.velocity
            v_mag = np.linalg.norm(v_apparent)
            
            if v_mag < 0.1:
                node.aero_force = np.zeros(3)
                continue
            
            v_dir = v_apparent / v_mag
            
            # Dynamic pressure
            q = 0.5 * self.rho * v_mag**2
            
            # Pitch (angle of attack)
            pitch = self.get_node_pitch(node)
            
            # Lift: perpendicular to velocity, in vertical plane
            # Simplified: lift acts in Z direction modified by pitch
            Cl = 2 * np.pi * np.sin(pitch) * 0.8  # Thin airfoil theory with efficiency
            lift_mag = q * node.wing_area * Cl
            
            # Lift direction: perpendicular to v_dir, biased upward
            lift_dir = np.array([0, 0, 1]) - np.dot([0, 0, 1], v_dir) * v_dir
            lift_norm = np.linalg.norm(lift_dir)
            if lift_norm > 0.01:
                lift_dir = lift_dir / lift_norm
            else:
                lift_dir = np.array([0, 0, 1])
            
            lift = lift_mag * lift_dir
            
            # Drag
            Cd = 0.02 + Cl**2 / (np.pi * node.aspect_ratio * 0.8)  # Parasitic + induced
            drag_mag = q * node.wing_area * Cd
            drag = -drag_mag * v_dir
            
            node.aero_force = lift + drag
    
    def step(self, dt: float, freestream: np.ndarray = None):
        """
        Step the constellation physics.
        
        Each node moves according to:
        - Tether constraint forces
        - Aerodynamic forces
        - Gravity
        """
        if freestream is None:
            freestream = np.zeros(3)
        
        # Reset forces
        for node in self.nodes.values():
            node.tether_force = np.zeros(3)
            node.aero_force = np.zeros(3)
        
        # Compute forces
        self.compute_centroid()
        self.compute_tether_forces()
        self.compute_aero_forces(freestream)
        
        # Update spin phase based on angular momentum
        L = self.compute_angular_momentum()
        L_mag = np.linalg.norm(L)
        if L_mag > 0.1 and L_mag < 1e6:  # Sanity check
            # Estimate spin rate from angular momentum and moment of inertia
            I_approx = sum(node.mass * np.linalg.norm(node.position - self.centroid)**2 
                          for node in self.nodes.values())
            if I_approx > 0.1:
                omega = min(L_mag / I_approx, 100.0)  # Cap spin rate
                self.spin_phase += omega * dt
                self.spin_phase = self.spin_phase % (2 * np.pi)  # Keep bounded
        
        # Integrate each node with stability limits
        MAX_VELOCITY = 200.0    # m/s
        MAX_ACCEL = 500.0       # m/s²
        MAX_FORCE = 10000.0     # N per node
        
        for node in self.nodes.values():
            # Total force with clamping
            gravity = np.array([0, 0, -self.g * node.mass])
            total_force = node.tether_force + node.aero_force + gravity
            
            # Clamp force magnitude
            force_mag = np.linalg.norm(total_force)
            if force_mag > MAX_FORCE:
                total_force = total_force * (MAX_FORCE / force_mag)
            
            # Acceleration with clamping
            accel = total_force / node.mass
            accel_mag = np.linalg.norm(accel)
            if accel_mag > MAX_ACCEL:
                accel = accel * (MAX_ACCEL / accel_mag)
            
            # Semi-implicit Euler
            node.velocity += accel * dt
            
            # Clamp velocity
            vel_mag = np.linalg.norm(node.velocity)
            if vel_mag > MAX_VELOCITY:
                node.velocity = node.velocity * (MAX_VELOCITY / vel_mag)
            
            # Check for NaN
            if np.any(np.isnan(node.velocity)):
                node.velocity = np.zeros(3)
            if np.any(np.isnan(node.position)):
                node.position = self.centroid.copy()
            
            node.position += node.velocity * dt
            
            # Ground collision
            if node.position[2] < 0.5:
                node.position[2] = 0.5
                node.velocity[2] = abs(node.velocity[2]) * 0.3  # Bounce
    
    def get_state(self) -> dict:
        """Get current constellation state."""
        return {
            'centroid': self.centroid.copy(),
            'spin_phase': self.spin_phase,
            'nodes': {
                nid: {
                    'position': n.position.copy(),
                    'velocity': n.velocity.copy(),
                    'pitch': self.get_node_pitch(n),
                    'aero_force': n.aero_force.copy(),
                    'tether_force': n.tether_force.copy(),
                }
                for nid, n in self.nodes.items()
            },
            'tethers': {
                f"{t.node_a}-{t.node_b}": {
                    'tension': t.tension,
                    'length': t.current_length,
                }
                for t in self.tethers.values()
            }
        }


def create_ring_constellation(n_nodes: int = 4, radius: float = 20.0) -> HublessDervish:
    """
    Create a ring-shaped hubless constellation.
    
    Nodes arranged in a circle, each connected to neighbors.
    """
    dervish = HublessDervish()
    
    # Create nodes in a ring
    for i in range(n_nodes):
        angle = 2 * np.pi * i / n_nodes
        pos = np.array([
            radius * np.cos(angle),
            radius * np.sin(angle),
            50.0  # Altitude
        ])
        dervish.add_node(f"N{i}", pos)
    
    # Connect neighbors (ring topology)
    for i in range(n_nodes):
        next_i = (i + 1) % n_nodes
        dervish.connect(f"N{i}", f"N{next_i}")
    
    # Add cross-bracing for stability (connect opposites)
    if n_nodes >= 4:
        for i in range(n_nodes // 2):
            opposite = (i + n_nodes // 2) % n_nodes
            dervish.connect(f"N{i}", f"N{opposite}")
    
    return dervish


def create_tetrahedron_constellation(edge_length: float = 30.0) -> HublessDervish:
    """
    Create a tetrahedral hubless constellation.
    
    Four nodes at tetrahedron vertices, all connected.
    Most stable 3D configuration.
    """
    dervish = HublessDervish()
    
    # Tetrahedron vertices (centered at origin, altitude 50)
    h = edge_length * np.sqrt(2/3)
    r = edge_length / np.sqrt(3)
    
    vertices = [
        np.array([r, 0, 50]),
        np.array([-r/2, r * np.sqrt(3)/2, 50]),
        np.array([-r/2, -r * np.sqrt(3)/2, 50]),
        np.array([0, 0, 50 + h]),
    ]
    
    for i, pos in enumerate(vertices):
        dervish.add_node(f"T{i}", pos)
    
    # Fully connected
    for i in range(4):
        for j in range(i + 1, 4):
            dervish.connect(f"T{i}", f"T{j}")
    
    return dervish


# =============================================================================
# BOLA LAUNCH SYSTEM
# =============================================================================

class LaunchPhase(Enum):
    """Phases of bola launch."""
    COLLAPSED = "collapsed"      # Bundled in hand
    SPINNING_UP = "spinning_up"  # Human spinning overhead
    RELEASED = "released"        # In flight, deploying
    DEPLOYED = "deployed"        # Constellation spread, autonomous


@dataclass
class BolaLaunchState:
    """State of a hand-thrown bola launch."""
    phase: LaunchPhase = LaunchPhase.COLLAPSED
    
    # Pre-release (human spinning)
    spin_radius: float = 1.0          # Arm length
    spin_rate: float = 0.0            # rad/s (builds up)
    spin_angle: float = 0.0           # Current angle
    
    # Release parameters
    release_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    release_angular_momentum: float = 0.0
    release_height: float = 1.8       # Human shoulder height
    
    # Deployment progress
    deployment_fraction: float = 0.0  # 0 = collapsed, 1 = fully spread
    
    def __post_init__(self):
        if not isinstance(self.release_velocity, np.ndarray):
            self.release_velocity = np.zeros(3)


class BolaLauncher:
    """
    Simulates human bola throw to bootstrap the hubless constellation.
    
    The throw sequence:
    1. Hold collapsed constellation (all nodes at one point)
    2. Spin overhead - angular velocity builds
    3. Release at optimal angle - converts angular to linear momentum
    4. Centrifugal deployment - tethers extend, nodes spread
    5. Airfoil engagement - sustain spin via aerodynamics
    
    This is how the constellation comes to life from a human hand.
    """
    
    def __init__(self, 
                 dervish: HublessDervish,
                 human_arm_length: float = 0.8,
                 max_human_spin_rate: float = 3.0):  # ~3 Hz max human spin
        
        self.dervish = dervish
        self.arm_length = human_arm_length
        self.max_spin_rate = max_human_spin_rate * 2 * np.pi  # Convert to rad/s
        
        self.state = BolaLaunchState()
        
        # Deployment parameters
        self.full_radius = self._compute_full_radius()
        self.deployment_time = 2.0  # seconds to fully deploy
        
    def _compute_full_radius(self) -> float:
        """Compute the fully deployed constellation radius."""
        if not self.dervish.tethers:
            return 10.0
        
        # Average tether length
        total = sum(t.rest_length for t in self.dervish.tethers.values())
        return total / len(self.dervish.tethers) / 2
    
    def collapse_for_throw(self, hand_position: np.ndarray):
        """
        Collapse all nodes to hand position for throwing.
        """
        self.state.phase = LaunchPhase.COLLAPSED
        self.state.deployment_fraction = 0.0
        
        for node in self.dervish.nodes.values():
            node.position = hand_position.copy()
            node.velocity = np.zeros(3)
    
    def spin_up(self, dt: float, effort: float = 1.0):
        """
        Simulate human spinning the collapsed bola overhead.
        
        Args:
            dt: Time step
            effort: 0-1 how hard the human is spinning
        """
        if self.state.phase == LaunchPhase.COLLAPSED:
            self.state.phase = LaunchPhase.SPINNING_UP
        
        if self.state.phase != LaunchPhase.SPINNING_UP:
            return
        
        # Build up spin rate (human acceleration)
        target_rate = effort * self.max_spin_rate
        rate_accel = 5.0  # rad/s² - how fast human can accelerate spin
        
        if self.state.spin_rate < target_rate:
            self.state.spin_rate += rate_accel * dt
            self.state.spin_rate = min(self.state.spin_rate, target_rate)
        
        # Update angle
        self.state.spin_angle += self.state.spin_rate * dt
        
        # Move all nodes in a circle at arm's length
        # (collapsed into single point, orbiting overhead)
        center = np.array([0, 0, self.state.release_height + 0.3])  # Overhead
        radius = self.arm_length
        
        x = center[0] + radius * np.cos(self.state.spin_angle)
        y = center[1] + radius * np.sin(self.state.spin_angle)
        z = center[2]
        
        bundle_pos = np.array([x, y, z])
        
        # Tangential velocity
        tangent = np.array([
            -np.sin(self.state.spin_angle),
            np.cos(self.state.spin_angle),
            0
        ])
        bundle_vel = self.state.spin_rate * radius * tangent
        
        for node in self.dervish.nodes.values():
            node.position = bundle_pos.copy()
            node.velocity = bundle_vel.copy()
    
    def release(self, throw_angle: float = np.pi/4):
        """
        Release the bola at current spin state.
        
        Args:
            throw_angle: Release angle from horizontal (rad)
                        π/4 (45°) is optimal for range
        """
        if self.state.phase != LaunchPhase.SPINNING_UP:
            return
        
        self.state.phase = LaunchPhase.RELEASED
        
        # Current bundle state
        nodes = list(self.dervish.nodes.values())
        if not nodes:
            return
        
        bundle_pos = nodes[0].position.copy()
        bundle_vel = nodes[0].velocity.copy()
        
        # Add upward component based on throw angle
        speed = np.linalg.norm(bundle_vel)
        horizontal_dir = bundle_vel / speed if speed > 0.1 else np.array([1, 0, 0])
        
        # Redirect velocity at throw angle
        release_vel = speed * (
            np.cos(throw_angle) * horizontal_dir +
            np.sin(throw_angle) * np.array([0, 0, 1])
        )
        
        self.state.release_velocity = release_vel
        self.state.release_angular_momentum = (
            self.state.spin_rate * self.arm_length**2 * 
            sum(n.mass for n in self.dervish.nodes.values())
        )
        
        # Apply release velocity to all nodes
        for node in self.dervish.nodes.values():
            node.velocity = release_vel.copy()
        
        print(f"RELEASED! Speed: {speed:.1f} m/s, Angle: {np.degrees(throw_angle):.0f}°")
        print(f"  Angular momentum: {self.state.release_angular_momentum:.1f} kg·m²/s")
    
    def update_deployment(self, dt: float):
        """
        Update constellation deployment after release.
        
        Centrifugal force spreads the nodes outward.
        """
        if self.state.phase not in [LaunchPhase.RELEASED, LaunchPhase.DEPLOYED]:
            return
        
        # Deployment progress
        self.state.deployment_fraction += dt / self.deployment_time
        self.state.deployment_fraction = min(1.0, self.state.deployment_fraction)
        
        if self.state.deployment_fraction >= 1.0:
            self.state.phase = LaunchPhase.DEPLOYED
        
        # Current target radius
        target_radius = self.state.deployment_fraction * self.full_radius
        
        # Spread nodes outward from centroid
        centroid = self.dervish.compute_centroid()
        n_nodes = len(self.dervish.nodes)
        
        for i, node in enumerate(self.dervish.nodes.values()):
            # Target position: spread in a circle
            angle = 2 * np.pi * i / n_nodes + self.dervish.spin_phase
            
            target_pos = centroid + target_radius * np.array([
                np.cos(angle),
                np.sin(angle),
                0
            ])
            
            # Smoothly move toward target
            deployment_force = 50.0 * (target_pos - node.position)
            
            # Apply as velocity adjustment (simplified)
            node.velocity += deployment_force * dt / node.mass
    
    def step(self, dt: float, freestream: np.ndarray = None):
        """
        Step the launch simulation.
        """
        if self.state.phase == LaunchPhase.SPINNING_UP:
            # Just update spinning position (spin_up must be called separately)
            pass
        
        elif self.state.phase in [LaunchPhase.RELEASED, LaunchPhase.DEPLOYED]:
            # Update deployment
            self.update_deployment(dt)
            
            # Run constellation physics
            if freestream is None:
                freestream = np.zeros(3)
            self.dervish.step(dt, freestream)
    
    def get_state(self) -> dict:
        """Get launch state."""
        return {
            'phase': self.state.phase.value,
            'spin_rate_hz': self.state.spin_rate / (2 * np.pi),
            'deployment': self.state.deployment_fraction,
            'release_velocity': self.state.release_velocity.copy(),
            'constellation': self.dervish.get_state()
        }


# Demo
if __name__ == "__main__":
    print("=" * 60)
    print("HUBLESS DERVISH - BOLA LAUNCH SEQUENCE")
    print("=" * 60)
    print()
    print("A human throws a collapsed constellation like a bola.")
    print("The throw provides initial angular momentum.")
    print("Centrifugal force deploys it. Airfoils sustain the spin.")
    print()
    
    # Create constellation (collapsed - positions don't matter yet)
    dervish = create_ring_constellation(n_nodes=4, radius=10.0)
    
    # Create launcher
    launcher = BolaLauncher(dervish, human_arm_length=0.8)
    
    # PHASE 1: Collapse into hand
    print("PHASE 1: COLLAPSED IN HAND")
    hand_pos = np.array([0, 0, 1.8])  # Shoulder height
    launcher.collapse_for_throw(hand_pos)
    print(f"  All {len(dervish.nodes)} nodes at hand position: {hand_pos}")
    print()
    
    # PHASE 2: Spin up overhead
    print("PHASE 2: SPINNING UP OVERHEAD")
    dt = 0.02
    spinup_time = 2.0  # 2 seconds of spinning
    
    for t in np.arange(0, spinup_time, dt):
        launcher.spin_up(dt, effort=min(1.0, t / 1.0))  # Ramp up effort
        
        if int(t * 10) % 5 == 0 and t > 0:
            rpm = launcher.state.spin_rate / (2 * np.pi) * 60
            print(f"  t={t:.1f}s: Spin rate = {rpm:.0f} RPM")
    
    print()
    
    # PHASE 3: Release!
    print("PHASE 3: RELEASE!")
    launcher.release(throw_angle=np.radians(30))  # 30° upward
    print()
    
    # PHASE 4: Flight and deployment
    print("PHASE 4: DEPLOYMENT AND FLIGHT")
    
    flight_time = 5.0
    freestream = np.array([0, 0, 0])  # Still air
    
    # Command thrust once deployed
    dervish.command_thrust(np.array([1, 0, 0]), magnitude=30)
    dervish.collective_pitch = np.radians(10)
    
    for step in range(int(flight_time / dt)):
        launcher.step(dt, freestream)
        
        if step % 50 == 0:
            state = launcher.get_state()
            c = state['constellation']['centroid']
            phase = state['phase']
            deploy = state['deployment']
            
            # Compute constellation radius
            positions = [n['position'] for n in state['constellation']['nodes'].values()]
            if positions:
                mean_pos = np.mean(positions, axis=0)
                radius = np.mean([np.linalg.norm(p - mean_pos) for p in positions])
            else:
                radius = 0
            
            print(f"  t={step*dt:.1f}s | {phase:10s} | "
                  f"Deploy: {deploy*100:5.1f}% | "
                  f"Radius: {radius:5.1f}m | "
                  f"Alt: {c[2]:6.1f}m | "
                  f"Pos: [{c[0]:6.1f}, {c[1]:6.1f}]")
    
    print()
    print("=" * 60)
    print("The constellation flies from a human throw.")
    print("No engines. No batteries. Just physics.")
    print("The ends do the work. The center is imaginary.")
    print("=" * 60)
