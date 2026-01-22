"""
Corkscrew Propulsion Physics
============================

Implements centrifugal propulsion through coordinated rotation of tethered nodes.

The corkscrew mechanism works by:
1. Counter-rotating sibling nodes around the tether axis
2. Phase-offset rotations create directional thrust
3. Gyroscopic precession enables 3D maneuvering
4. Synchronized pulses multiply effective thrust

This creates a "mechanical turbine" effect where the lattice can propel
itself by leveraging internal angular momentum against the tethers.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
from enum import Enum, auto


class RotationDirection(Enum):
    CW = auto()   # Clockwise (viewed from parent)
    CCW = auto()  # Counter-clockwise


@dataclass
class CorkscrewState:
    """State of a single node's corkscrew rotation"""
    phase: float = 0.0              # Current rotation phase (radians)
    frequency: float = 2.0          # Rotation frequency (Hz)
    amplitude: float = 0.5          # Rotation radius (m)
    direction: RotationDirection = RotationDirection.CW
    
    # Computed thrust
    instantaneous_thrust: np.ndarray = None
    
    def __post_init__(self):
        if self.instantaneous_thrust is None:
            self.instantaneous_thrust = np.zeros(3)


@dataclass
class GyroState:
    """Gyroscopic stabilization state"""
    angular_momentum: np.ndarray = None
    precession_axis: np.ndarray = None
    precession_rate: float = 0.0
    stability_torque: np.ndarray = None
    
    def __post_init__(self):
        if self.angular_momentum is None:
            self.angular_momentum = np.zeros(3)
        if self.precession_axis is None:
            self.precession_axis = np.array([0, 0, 1])
        if self.stability_torque is None:
            self.stability_torque = np.zeros(3)


class CorkscrewEngine:
    """
    Physics engine for corkscrew propulsion dynamics.
    
    Manages coordinated rotation of tethered nodes to generate
    centrifugal thrust in arbitrary directions.
    """
    
    def __init__(self, 
                 base_frequency: float = 2.0,
                 phase_coupling: float = 0.3):
        """
        Args:
            base_frequency: Default rotation frequency (Hz)
            phase_coupling: How strongly siblings synchronize phases
        """
        self.base_frequency = base_frequency
        self.phase_coupling = phase_coupling
        
        # State tracking
        self._node_states: dict = {}  # node_id -> CorkscrewState
        self._gyro_states: dict = {}  # node_id -> GyroState
    
    def register_node(self, 
                      node_id: str, 
                      direction: RotationDirection = RotationDirection.CW,
                      phase_offset: float = 0.0) -> CorkscrewState:
        """Register a node for corkscrew dynamics"""
        state = CorkscrewState(
            phase=phase_offset,
            frequency=self.base_frequency,
            direction=direction
        )
        self._node_states[node_id] = state
        self._gyro_states[node_id] = GyroState()
        return state
    
    def compute_rotation_thrust(self,
                                 node_id: str,
                                 parent_position: np.ndarray,
                                 node_position: np.ndarray,
                                 node_mass: float,
                                 dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute centrifugal thrust from corkscrew rotation.
        
        Returns:
            (thrust_vector, angular_velocity)
        """
        if node_id not in self._node_states:
            return np.zeros(3), np.zeros(3)
        
        state = self._node_states[node_id]
        
        # Update phase
        omega = 2 * np.pi * state.frequency
        phase_delta = omega * dt
        if state.direction == RotationDirection.CCW:
            phase_delta = -phase_delta
        state.phase += phase_delta
        
        # Tether axis (from node toward parent)
        tether_vec = parent_position - node_position
        tether_length = np.linalg.norm(tether_vec)
        if tether_length < 0.01:
            return np.zeros(3), np.zeros(3)
        
        tether_axis = tether_vec / tether_length
        
        # Build orthonormal basis around tether axis
        # (for computing rotation in the perpendicular plane)
        if abs(tether_axis[0]) < 0.9:
            perp1 = np.cross(tether_axis, np.array([1, 0, 0]))
        else:
            perp1 = np.cross(tether_axis, np.array([0, 1, 0]))
        perp1 = perp1 / np.linalg.norm(perp1)
        perp2 = np.cross(tether_axis, perp1)
        
        # Centrifugal force: F = m * ω² * r
        # Direction rotates with phase
        centrifugal_mag = node_mass * omega**2 * state.amplitude
        
        # Rotating thrust vector in perpendicular plane
        thrust = centrifugal_mag * (
            np.cos(state.phase) * perp1 +
            np.sin(state.phase) * perp2
        )
        
        # Angular velocity around tether axis
        angular_vel = omega * tether_axis
        if state.direction == RotationDirection.CCW:
            angular_vel = -angular_vel
        
        state.instantaneous_thrust = thrust
        return thrust, angular_vel
    
    def compute_sibling_coordination(self,
                                      sibling_ids: List[str],
                                      target_thrust_direction: np.ndarray) -> dict:
        """
        Coordinate sibling phases to produce thrust in target direction.
        
        By phase-shifting rotations, we can sum individual centrifugal
        forces to create a net thrust vector.
        
        Returns:
            Dict of node_id -> optimal_phase_offset
        """
        if len(sibling_ids) == 0:
            return {}
        
        # Normalize target
        target_norm = np.linalg.norm(target_thrust_direction)
        if target_norm < 0.01:
            return {nid: 0.0 for nid in sibling_ids}
        
        target = target_thrust_direction / target_norm
        
        # Optimal phase offsets to sum toward target
        # For N siblings: offset_i = 2π * i / N + alignment_angle
        phase_offsets = {}
        n = len(sibling_ids)
        
        # Alignment angle: phase where thrust aligns with target
        # (simplified: use arctan2 of target's xy projection)
        alignment = np.arctan2(target[1], target[0])
        
        for i, node_id in enumerate(sibling_ids):
            # Distribute phases with alignment bias
            base_phase = 2 * np.pi * i / n
            optimal_phase = alignment + base_phase
            phase_offsets[node_id] = optimal_phase
            
            # Apply to state
            if node_id in self._node_states:
                current = self._node_states[node_id]
                # Smoothly blend toward optimal
                phase_error = optimal_phase - current.phase
                current.phase += self.phase_coupling * phase_error
        
        return phase_offsets
    
    def compute_gyroscopic_torque(self,
                                   node_id: str,
                                   children_positions: List[np.ndarray],
                                   children_velocities: List[np.ndarray],
                                   children_masses: List[float],
                                   node_position: np.ndarray) -> np.ndarray:
        """
        Compute gyroscopic stabilization torque from rotating children.
        
        The combined angular momentum of children creates a "virtual gyroscope"
        that can be used for attitude control.
        """
        if node_id not in self._gyro_states:
            self._gyro_states[node_id] = GyroState()
        
        gyro = self._gyro_states[node_id]
        
        # Total angular momentum of children
        L_total = np.zeros(3)
        for pos, vel, mass in zip(children_positions, children_velocities, children_masses):
            r = pos - node_position
            L = np.cross(r, mass * vel)
            L_total += L
        
        gyro.angular_momentum = L_total
        
        # Gyroscopic torque: τ = dL/dt ≈ -k * L for damping
        # (Real gyro would use precession, this is simplified stabilization)
        damping_coeff = 0.15
        gyro.stability_torque = -damping_coeff * L_total
        
        # Precession axis (perpendicular to L)
        L_mag = np.linalg.norm(L_total)
        if L_mag > 0.01:
            gyro.precession_axis = L_total / L_mag
            gyro.precession_rate = L_mag / (sum(children_masses) + 0.1)
        
        return gyro.stability_torque
    
    def compute_collective_thrust(self,
                                   node_ids: List[str],
                                   thrust_vectors: List[np.ndarray]) -> Tuple[np.ndarray, float]:
        """
        Sum individual thrusts to get net collective force.
        
        Returns:
            (net_thrust, efficiency) where efficiency is how well
            individual thrusts aligned (1.0 = perfect, 0.0 = cancelled)
        """
        if len(thrust_vectors) == 0:
            return np.zeros(3), 0.0
        
        # Sum thrusts
        net = sum(thrust_vectors)
        
        # Efficiency = |sum| / sum(|individual|)
        individual_sum = sum(np.linalg.norm(t) for t in thrust_vectors)
        if individual_sum < 0.01:
            return net, 0.0
        
        efficiency = np.linalg.norm(net) / individual_sum
        return net, efficiency
    
    def pulse_burst(self,
                    node_ids: List[str],
                    target_direction: np.ndarray,
                    pulse_duration: float = 0.1) -> dict:
        """
        Coordinate a synchronized thrust pulse.
        
        All nodes align phases simultaneously for maximum burst thrust
        in the target direction.
        
        Returns:
            Dict with pulse timing and expected thrust magnitude
        """
        # Align all phases instantly
        for nid in node_ids:
            if nid in self._node_states:
                state = self._node_states[nid]
                # Set phase to align with target
                alignment = np.arctan2(target_direction[1], target_direction[0])
                state.phase = alignment
        
        # Estimate burst magnitude
        total_mass = 0.5 * len(node_ids)  # Assume 0.5kg each
        omega = 2 * np.pi * self.base_frequency
        amplitude = 0.5  # Default amplitude
        
        # All forces aligned: F_total = N * m * ω² * r
        burst_magnitude = len(node_ids) * total_mass * omega**2 * amplitude
        
        return {
            'direction': target_direction / (np.linalg.norm(target_direction) + 0.001),
            'magnitude': burst_magnitude,
            'duration': pulse_duration,
            'nodes': node_ids,
            'efficiency': 1.0  # Perfect alignment during pulse
        }


class GyroscopicController:
    """
    High-level controller for gyroscopic attitude stabilization.
    
    Uses the lattice structure as a "distributed gyroscope" where
    counter-rotating sub-clusters provide 3-axis torque control.
    """
    
    def __init__(self, corkscrew_engine: CorkscrewEngine):
        self.engine = corkscrew_engine
        self.target_orientation = np.array([1, 0, 0, 0])  # Identity quaternion
        
        # PID gains
        self.kp = 5.0
        self.kd = 2.0
        self.ki = 0.1
        
        self._integral_error = np.zeros(3)
    
    def set_target_orientation(self, quaternion: np.ndarray) -> None:
        """Set desired attitude"""
        self.target_orientation = quaternion / np.linalg.norm(quaternion)
    
    def compute_correction_torque(self,
                                   current_orientation: np.ndarray,
                                   angular_velocity: np.ndarray,
                                   dt: float) -> np.ndarray:
        """
        Compute torque needed to reach target orientation.
        """
        # Quaternion error (simplified: use vector part)
        q_current = current_orientation / np.linalg.norm(current_orientation)
        q_target = self.target_orientation
        
        # Error quaternion: q_error = q_target * q_current^-1
        # For small angles, vector part ≈ axis * sin(θ/2) ≈ axis * θ/2
        # Simplified: just use difference of vector parts
        error = q_target[1:4] - q_current[1:4]
        
        # PID control
        self._integral_error += error * dt
        derivative = -angular_velocity  # Damping
        
        torque = (
            self.kp * error +
            self.kd * derivative +
            self.ki * self._integral_error
        )
        
        return torque
    
    def assign_cluster_rotations(self,
                                  child_clusters: List[List[str]],
                                  required_torque: np.ndarray) -> dict:
        """
        Assign rotation directions to child clusters to produce required torque.
        
        With 4 clusters (e.g., UP/DOWN/LEFT/RIGHT), we can produce torque
        about any axis by differential rotation.
        
        Returns:
            Dict of cluster_index -> RotationDirection
        """
        assignments = {}
        
        if len(child_clusters) < 2:
            return assignments
        
        # For 4 clusters, assign based on torque axis
        # Torque about X: clusters at ±Y rotate differentially
        # Torque about Y: clusters at ±X rotate differentially
        # Torque about Z: all clusters same direction (roll)
        
        # Simplified: first half CW, second half CCW, scaled by torque
        torque_mag = np.linalg.norm(required_torque)
        if torque_mag < 0.01:
            # No correction needed, balance rotations
            for i, cluster in enumerate(child_clusters):
                direction = RotationDirection.CW if i % 2 == 0 else RotationDirection.CCW
                assignments[i] = {
                    'direction': direction,
                    'frequency_scale': 1.0
                }
        else:
            # Apply differential based on torque direction
            torque_dir = required_torque / torque_mag
            
            for i, cluster in enumerate(child_clusters):
                # Cluster axis (simplified: based on index)
                cluster_angle = 2 * np.pi * i / len(child_clusters)
                cluster_axis = np.array([np.cos(cluster_angle), np.sin(cluster_angle), 0])
                
                # Alignment with torque determines direction/speed
                alignment = np.dot(cluster_axis, torque_dir)
                
                direction = RotationDirection.CW if alignment > 0 else RotationDirection.CCW
                freq_scale = 1.0 + abs(alignment) * 0.5  # Speed up aligned clusters
                
                assignments[i] = {
                    'direction': direction,
                    'frequency_scale': freq_scale,
                    'nodes': cluster
                }
        
        return assignments


if __name__ == "__main__":
    # Demo: Corkscrew thrust computation
    engine = CorkscrewEngine(base_frequency=3.0)
    
    # Register 4 sibling nodes
    for i in range(4):
        direction = RotationDirection.CW if i % 2 == 0 else RotationDirection.CCW
        engine.register_node(f"TAB_{i}", direction=direction, phase_offset=np.pi/2 * i)
    
    # Simulate thrust over time
    parent_pos = np.array([0, 0, 0])
    node_positions = [
        np.array([2, 0, -5]),
        np.array([-2, 0, -5]),
        np.array([0, 2, -5]),
        np.array([0, -2, -5])
    ]
    
    print("=== CORKSCREW THRUST SIMULATION ===")
    dt = 0.01
    total_thrust = np.zeros(3)
    
    for step in range(100):
        thrusts = []
        for i in range(4):
            thrust, ang_vel = engine.compute_rotation_thrust(
                f"TAB_{i}",
                parent_pos,
                node_positions[i],
                node_mass=0.5,
                dt=dt
            )
            thrusts.append(thrust)
        
        net, efficiency = engine.compute_collective_thrust([f"TAB_{i}" for i in range(4)], thrusts)
        total_thrust += net * dt
        
        if step % 25 == 0:
            print(f"Step {step:3d}: Net thrust = [{net[0]:7.3f}, {net[1]:7.3f}, {net[2]:7.3f}] | Eff = {efficiency:.2%}")
    
    print(f"\nTotal impulse over 1s: {total_thrust}")
    
    # Demo: Pulse burst
    print("\n=== PULSE BURST ===")
    burst = engine.pulse_burst(
        [f"TAB_{i}" for i in range(4)],
        target_direction=np.array([1, 0, 0]),
        pulse_duration=0.1
    )
    print(f"Burst magnitude: {burst['magnitude']:.1f} N")
    print(f"Direction: {burst['direction']}")
    print(f"Efficiency: {burst['efficiency']:.0%}")


# =============================================================================
# DERVISH TACKING SYSTEM
# =============================================================================
# Like sailing, but omnidirectional - "clawing" through air using cyclic
# airfoil pitch control at the spinning TAB ends.

@dataclass
class TackingState:
    """State of a single airfoil's tacking control."""
    collective_pitch: float = 0.0    # Base pitch angle (radians) - like helicopter collective
    cyclic_amplitude: float = 0.0    # Cyclic variation magnitude
    cyclic_phase: float = 0.0        # Where in rotation to max pitch
    current_pitch: float = 0.0       # Actual pitch this instant
    
    # Computed forces
    lift_vector: np.ndarray = None
    drag_vector: np.ndarray = None
    
    def __post_init__(self):
        if self.lift_vector is None:
            self.lift_vector = np.zeros(3)
        if self.drag_vector is None:
            self.drag_vector = np.zeros(3)


class DervishTackingEngine:
    """
    Omnidirectional propulsion through cyclic airfoil pitch control.
    
    The TABs spin around the Buzzard hull like propeller blades.
    By varying each TAB's pitch cyclically (based on rotation phase),
    we can generate thrust in ANY direction - like a helicopter's
    cyclic control, or a sailboat tacking against wind.
    
    Key insight: The airflow from spinning creates apparent wind.
    By adjusting pitch at specific phases of rotation, the airfoils
    generate asymmetric lift that sums to directional thrust.
    
    "Clawing through space using aerodynamics"
    """
    
    def __init__(self,
                 spin_radius: float = 30.0,      # TAB distance from hub (m)
                 airfoil_area: float = 2.0,      # TAB wing area (m²)
                 air_density: float = 1.225,     # kg/m³ at sea level
                 lift_coefficient: float = 1.2,
                 drag_coefficient: float = 0.1):
        
        self.spin_radius = spin_radius
        self.airfoil_area = airfoil_area
        self.air_density = air_density
        self.Cl = lift_coefficient
        self.Cd = drag_coefficient
        
        # Tacking states per TAB
        self._tack_states: dict = {}  # tab_id -> TackingState
        
        # Spin state
        self.spin_rate = 0.0          # rad/s
        self.spin_plane_normal = np.array([0, 0, 1])  # Axis of rotation
        self.spin_phase = 0.0         # Current rotation angle
        
        # Thrust command
        self.target_thrust = np.zeros(3)  # Desired thrust direction/magnitude
        
    def register_tab(self, tab_id: str, phase_offset: float = 0.0) -> TackingState:
        """Register a TAB for tacking control."""
        state = TackingState()
        self._tack_states[tab_id] = {
            'state': state,
            'phase_offset': phase_offset  # Where in circle this TAB sits
        }
        return state
    
    def set_spin_rate(self, rate_hz: float):
        """Set rotation frequency."""
        self.spin_rate = 2 * np.pi * rate_hz
        
    def set_spin_plane(self, normal: np.ndarray):
        """Set the plane of rotation (normal vector)."""
        n = np.linalg.norm(normal)
        if n > 0.01:
            self.spin_plane_normal = normal / n
            
    def command_thrust(self, direction: np.ndarray, magnitude: float = 1.0):
        """
        Command desired thrust direction.
        
        The tacking system will compute cyclic pitch to achieve this.
        """
        n = np.linalg.norm(direction)
        if n > 0.01:
            self.target_thrust = (direction / n) * magnitude
        else:
            self.target_thrust = np.zeros(3)
            
    def compute_cyclic_for_thrust(self) -> Tuple[float, float]:
        """
        Compute cyclic amplitude and phase to achieve target thrust.
        
        Returns:
            (cyclic_amplitude, cyclic_phase) in radians
        """
        thrust = self.target_thrust
        mag = np.linalg.norm(thrust)
        
        if mag < 0.01:
            return 0.0, 0.0
        
        # Project thrust onto spin plane
        # The cyclic phase determines WHERE in the rotation we pitch up
        # The cyclic amplitude determines HOW MUCH asymmetry
        
        # Build basis in spin plane
        up = self.spin_plane_normal
        if abs(up[0]) < 0.9:
            right = np.cross(up, [1, 0, 0])
        else:
            right = np.cross(up, [0, 1, 0])
        right = right / np.linalg.norm(right)
        forward = np.cross(right, up)
        
        # Project thrust onto spin plane
        thrust_right = np.dot(thrust, right)
        thrust_forward = np.dot(thrust, forward)
        
        # Cyclic phase: 90° before where we want thrust
        # (pitch up creates lift perpendicular to motion)
        cyclic_phase = np.arctan2(thrust_forward, thrust_right) - np.pi/2
        
        # Cyclic amplitude: proportional to desired lateral thrust
        lateral_mag = np.sqrt(thrust_right**2 + thrust_forward**2)
        cyclic_amplitude = np.clip(lateral_mag / 100.0, 0, np.radians(15))  # Max 15° cyclic
        
        return cyclic_amplitude, cyclic_phase
    
    def compute_tab_pitch(self, 
                          tab_id: str,
                          rotation_phase: float) -> float:
        """
        Compute instantaneous pitch for a TAB based on rotation phase.
        
        Like helicopter cyclic: pitch varies sinusoidally with rotation.
        """
        if tab_id not in self._tack_states:
            return 0.0
        
        entry = self._tack_states[tab_id]
        state = entry['state']
        phase_offset = entry['phase_offset']
        
        # Get cyclic from thrust command
        cyclic_amp, cyclic_phase = self.compute_cyclic_for_thrust()
        
        # Total phase = rotation + this TAB's offset
        total_phase = rotation_phase + phase_offset
        
        # Pitch = collective + cyclic * sin(phase - cyclic_phase)
        pitch = state.collective_pitch + cyclic_amp * np.sin(total_phase - cyclic_phase)
        
        state.current_pitch = pitch
        return pitch
    
    def compute_airfoil_forces(self,
                                tab_id: str,
                                tab_position: np.ndarray,
                                hub_position: np.ndarray,
                                freestream_velocity: np.ndarray,
                                rotation_phase: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute lift and drag on a spinning TAB airfoil.
        
        The apparent wind = freestream - rotational velocity
        
        Returns:
            (lift_force, drag_force) in Newtons
        """
        if tab_id not in self._tack_states:
            return np.zeros(3), np.zeros(3)
        
        entry = self._tack_states[tab_id]
        state = entry['state']
        
        # TAB's rotational velocity (tangent to circle)
        r = tab_position - hub_position
        r_mag = np.linalg.norm(r)
        if r_mag < 0.1:
            return np.zeros(3), np.zeros(3)
        
        # Tangent direction (perpendicular to radius, in spin plane)
        tangent = np.cross(self.spin_plane_normal, r / r_mag)
        tangent_mag = np.linalg.norm(tangent)
        if tangent_mag < 0.01:
            return np.zeros(3), np.zeros(3)
        tangent = tangent / tangent_mag
        
        # Rotational velocity
        v_rot = self.spin_rate * r_mag * tangent
        
        # Apparent wind = freestream - rotational velocity
        # (from TAB's perspective, rotation creates headwind)
        v_apparent = freestream_velocity - v_rot
        v_mag = np.linalg.norm(v_apparent)
        
        if v_mag < 0.1:
            return np.zeros(3), np.zeros(3)
        
        v_dir = v_apparent / v_mag
        
        # Dynamic pressure
        q = 0.5 * self.air_density * v_mag**2
        
        # Get pitch for this rotation phase
        pitch = self.compute_tab_pitch(tab_id, rotation_phase)
        
        # Effective angle of attack (simplified)
        alpha = pitch  # Assuming airfoil aligned with tangent
        
        # Lift (perpendicular to apparent wind, in spin plane)
        lift_dir = np.cross(v_dir, self.spin_plane_normal)
        lift_mag_n = lift_dir / (np.linalg.norm(lift_dir) + 0.001)
        
        # Lift magnitude: L = q * S * Cl * sin(alpha) (linearized)
        Cl_effective = self.Cl * np.sin(alpha)
        lift_mag = q * self.airfoil_area * Cl_effective
        lift = lift_mag * lift_mag_n
        
        # Drag (opposite to apparent wind)
        Cd_effective = self.Cd + 0.05 * alpha**2  # Induced drag
        drag_mag = q * self.airfoil_area * Cd_effective
        drag = -drag_mag * v_dir
        
        state.lift_vector = lift
        state.drag_vector = drag
        
        return lift, drag
    
    def step(self, 
             tab_positions: dict,  # tab_id -> position
             hub_position: np.ndarray,
             freestream: np.ndarray,
             dt: float) -> dict:
        """
        Step the dervish tacking simulation.
        
        Returns:
            Dict with net_thrust, torque, power, per-tab forces
        """
        # Update spin phase
        self.spin_phase += self.spin_rate * dt
        self.spin_phase = self.spin_phase % (2 * np.pi)
        
        net_thrust = np.zeros(3)
        net_torque = np.zeros(3)
        total_power = 0.0
        tab_forces = {}
        
        for tab_id, pos in tab_positions.items():
            if tab_id not in self._tack_states:
                continue
            
            entry = self._tack_states[tab_id]
            phase = self.spin_phase + entry['phase_offset']
            
            lift, drag = self.compute_airfoil_forces(
                tab_id, pos, hub_position, freestream, phase
            )
            
            total_force = lift + drag
            net_thrust += total_force
            
            # Torque about hub
            r = pos - hub_position
            torque = np.cross(r, total_force)
            net_torque += torque
            
            # Power = torque component along spin axis * spin rate
            power = abs(np.dot(torque, self.spin_plane_normal) * self.spin_rate)
            total_power += power
            
            tab_forces[tab_id] = {
                'lift': lift.copy(),
                'drag': drag.copy(),
                'total': total_force.copy(),
                'pitch': entry['state'].current_pitch
            }
        
        return {
            'net_thrust': net_thrust,
            'net_torque': net_torque,
            'power': total_power,
            'tab_forces': tab_forces,
            'spin_phase': self.spin_phase
        }


# Demo
if __name__ == "__main__":
    print("\n=== DERVISH TACKING DEMO ===")
    
    dervish = DervishTackingEngine(spin_radius=30.0)
    dervish.set_spin_rate(1.0)  # 1 Hz = 60 RPM
    
    # Register 4 TABs in cross formation
    for i, tab_id in enumerate(["UP", "DOWN", "LEFT", "RIGHT"]):
        dervish.register_tab(tab_id, phase_offset=i * np.pi / 2)
    
    # Set collective pitch for all
    for entry in dervish._tack_states.values():
        entry['state'].collective_pitch = np.radians(5)  # 5° base pitch
    
    # Command thrust forward
    dervish.command_thrust(np.array([0, 1, 0]), magnitude=100)
    
    # Simulate
    hub = np.array([0, 0, 50])
    freestream = np.array([0, 20, 0])  # 20 m/s forward flight
    
    print(f"Hub at {hub}, freestream = {freestream} m/s")
    print(f"Commanded thrust: {dervish.target_thrust}")
    
    dt = 0.01
    for step in range(100):
        # TAB positions (spinning)
        phase = dervish.spin_phase
        r = dervish.spin_radius
        positions = {
            "UP": hub + np.array([r * np.cos(phase), r * np.sin(phase), 10]),
            "DOWN": hub + np.array([r * np.cos(phase + np.pi), r * np.sin(phase + np.pi), -10]),
            "LEFT": hub + np.array([r * np.cos(phase + np.pi/2), r * np.sin(phase + np.pi/2), 0]),
            "RIGHT": hub + np.array([r * np.cos(phase + 3*np.pi/2), r * np.sin(phase + 3*np.pi/2), 0]),
        }
        
        result = dervish.step(positions, hub, freestream, dt)
        
        if step % 25 == 0:
            t = result['net_thrust']
            print(f"Step {step:3d} | Phase {np.degrees(result['spin_phase']):5.1f}° | "
                  f"Thrust [{t[0]:7.1f}, {t[1]:7.1f}, {t[2]:7.1f}] N | "
                  f"Power {result['power']/1000:.1f} kW")
    
    print("\nDervish tacking: Clawing through air omnidirectionally!")

