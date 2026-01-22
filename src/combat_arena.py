"""
Combat Arena - 3D Asteroid Defense
===================================

Incoming threats (asteroids, missiles, drones) spawn and attack the Buzzard.
The DreamerV3 quine lattice must intercept and destroy them using:
  - Blade-tether slashing (rotating arms)
  - Kinetic intercepts (released TABs)
  - Collective thrust dodging

This is the game loop that feeds observations to the champion brain
and executes its actions on the lattice.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum, auto
import time


class ThreatType(Enum):
    """Types of incoming threats"""
    ASTEROID = auto()       # Slow, dumb, tough
    MISSILE = auto()        # Fast, tracking, fragile
    ENEMY_DRONE = auto()    # Medium, evasive, armed
    DEBRIS = auto()         # Random, small, numerous
    SWARM = auto()          # Many tiny units


@dataclass
class Threat:
    """An incoming threat to intercept"""
    threat_id: str
    threat_type: ThreatType
    
    # State
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    
    # Properties
    radius: float = 1.0          # Collision radius (m)
    mass: float = 10.0           # kg
    health: float = 100.0        # Damage needed to destroy
    damage: float = 50.0         # Damage dealt on collision
    
    # Tracking (for missiles)
    target: Optional[np.ndarray] = None
    tracking_strength: float = 0.0  # 0 = dumb, 1 = perfect tracking
    
    # State flags
    alive: bool = True
    
    def update(self, dt: float, target_pos: np.ndarray = None) -> None:
        """Update threat position"""
        if not self.alive:
            return
        
        # Tracking behavior
        if self.tracking_strength > 0 and target_pos is not None:
            to_target = target_pos - self.position
            dist = np.linalg.norm(to_target)
            if dist > 0.1:
                desired_dir = to_target / dist
                current_dir = self.velocity / (np.linalg.norm(self.velocity) + 0.01)
                
                # Blend toward target
                new_dir = (1 - self.tracking_strength * dt) * current_dir + \
                          self.tracking_strength * dt * desired_dir
                new_dir = new_dir / (np.linalg.norm(new_dir) + 0.01)
                
                speed = np.linalg.norm(self.velocity)
                self.velocity = new_dir * speed
        
        # Move
        self.position += self.velocity * dt
    
    def take_damage(self, damage: float) -> bool:
        """Apply damage, return True if destroyed"""
        self.health -= damage
        if self.health <= 0:
            self.alive = False
            return True
        return False


class ThreatSpawner:
    """
    Spawns waves of threats around the arena.
    Difficulty escalates over time.
    """
    
    def __init__(self, 
                 arena_radius: float = 150.0,
                 target_position: np.ndarray = None):
        self.arena_radius = arena_radius
        self.target = target_position if target_position is not None else np.zeros(3)
        
        self.threats: List[Threat] = []
        self.next_threat_id = 0
        
        # Wave system
        self.wave_number = 0
        self.threats_per_wave = 5
        self.time_between_waves = 5.0  # seconds (faster waves)
        self.time_since_last_wave = 0.0
        
        # Difficulty scaling
        self.base_speed = 40.0  # m/s (doubled for faster action)
        self.speed_increase_per_wave = 5.0
        
    def spawn_threat(self, 
                     threat_type: ThreatType = ThreatType.ASTEROID,
                     position: np.ndarray = None,
                     velocity: np.ndarray = None) -> Threat:
        """Spawn a single threat"""
        
        # Random spawn position on sphere around arena
        if position is None:
            theta = np.random.uniform(0, 2 * np.pi)
            phi = np.random.uniform(-np.pi/2, np.pi/2)
            r = self.arena_radius
            position = np.array([
                r * np.cos(phi) * np.cos(theta),
                r * np.cos(phi) * np.sin(theta),
                r * np.sin(phi)
            ])
        
        # Velocity toward target (with some randomness)
        if velocity is None:
            to_target = self.target - position
            to_target = to_target / (np.linalg.norm(to_target) + 0.01)
            
            # Add randomness
            to_target += np.random.randn(3) * 0.2
            to_target = to_target / (np.linalg.norm(to_target) + 0.01)
            
            speed = self.base_speed + self.wave_number * self.speed_increase_per_wave
            
            # Type-specific speeds
            if threat_type == ThreatType.MISSILE:
                speed *= 2.0
            elif threat_type == ThreatType.DEBRIS:
                speed *= 0.5
            
            velocity = to_target * speed
        
        # Create threat with type-specific properties
        threat = Threat(
            threat_id=f"THREAT_{self.next_threat_id}",
            threat_type=threat_type,
            position=position.copy(),
            velocity=velocity.copy()
        )
        
        # Type-specific stats
        if threat_type == ThreatType.ASTEROID:
            threat.radius = np.random.uniform(2.0, 5.0)  # Smaller asteroids
            threat.mass = threat.radius ** 2 * 20  # Less mass
            threat.health = threat.mass * 0.3
            threat.damage = min(threat.mass * 0.05, 30.0)  # Cap damage at 30
            threat.tracking_strength = 0.0
            
        elif threat_type == ThreatType.MISSILE:
            threat.radius = 0.5
            threat.mass = 10.0
            threat.health = 20.0
            threat.damage = 50.0  # Reduced from 100
            threat.tracking_strength = 0.6  # Slightly less tracking
            threat.target = self.target.copy()
            
        elif threat_type == ThreatType.ENEMY_DRONE:
            threat.radius = 1.5
            threat.mass = 50.0
            threat.health = 80.0
            threat.damage = 40.0
            threat.tracking_strength = 0.3
            threat.target = self.target.copy()
            
        elif threat_type == ThreatType.DEBRIS:
            threat.radius = np.random.uniform(0.2, 1.0)
            threat.mass = threat.radius ** 3 * 50
            threat.health = threat.mass * 0.3
            threat.damage = threat.mass * 0.1
            threat.tracking_strength = 0.0
        
        self.threats.append(threat)
        self.next_threat_id += 1
        
        return threat
    
    def spawn_wave(self) -> List[Threat]:
        """Spawn a wave of threats"""
        self.wave_number += 1
        wave = []
        
        # More threats per wave as difficulty increases
        n_threats = self.threats_per_wave + self.wave_number
        
        for i in range(n_threats):
            # Mix of threat types
            roll = np.random.random()
            if roll < 0.5:
                threat_type = ThreatType.ASTEROID
            elif roll < 0.7:
                threat_type = ThreatType.DEBRIS
            elif roll < 0.9:
                threat_type = ThreatType.MISSILE
            else:
                threat_type = ThreatType.ENEMY_DRONE
            
            threat = self.spawn_threat(threat_type)
            wave.append(threat)
        
        print(f"[WAVE {self.wave_number}] Spawned {len(wave)} threats!")
        return wave
    
    def update(self, dt: float) -> List[Threat]:
        """Update all threats, spawn new waves, return active threats"""
        
        # Update existing threats
        for threat in self.threats:
            threat.update(dt, self.target)
        
        # Remove dead or out-of-bounds threats
        self.threats = [t for t in self.threats 
                        if t.alive and np.linalg.norm(t.position) < self.arena_radius * 1.5]
        
        # Check for wave spawn
        self.time_since_last_wave += dt
        if self.time_since_last_wave >= self.time_between_waves:
            self.spawn_wave()
            self.time_since_last_wave = 0.0
            # Decrease time between waves (harder!)
            self.time_between_waves = max(3.0, self.time_between_waves - 0.5)
        
        return self.threats
    
    def get_closest_threat(self, position: np.ndarray) -> Optional[Threat]:
        """Get the closest active threat to a position"""
        if not self.threats:
            return None
        
        distances = [np.linalg.norm(t.position - position) for t in self.threats if t.alive]
        if not distances:
            return None
        
        idx = np.argmin(distances)
        return [t for t in self.threats if t.alive][idx]


@dataclass
class CombatStats:
    """Track combat statistics"""
    threats_destroyed: int = 0
    threats_escaped: int = 0
    damage_taken: float = 0.0
    damage_dealt: float = 0.0
    blades_active: int = 0
    total_thrust: float = 0.0
    wave_survived: int = 0
    
    @property
    def score(self) -> int:
        return self.threats_destroyed * 100 - self.threats_escaped * 50 - int(self.damage_taken)


class CombatArena:
    """
    The main game loop for the combat simulation.
    Integrates:
      - Threat spawning
      - Lattice physics
      - Champion brain inference
      - Collision detection
    """
    
    def __init__(self, lattice, champion_brain=None):
        """
        Args:
            lattice: QuineLattice with blade-tethers
            champion_brain: The DreamerV3 champion for decision making
        """
        self.lattice = lattice
        self.brain = champion_brain
        
        # Spread out initial node positions
        self._initialize_node_positions()
        
        # Arena setup
        self.arena_radius = 150.0  # Smaller arena for faster action
        self.spawner = ThreatSpawner(
            arena_radius=self.arena_radius,
            target_position=lattice.root.position
        )
    
    def _initialize_node_positions(self):
        """Spread nodes out from origin based on their hierarchy."""
        def spread_recursive(node, parent_pos, direction, distance):
            node.position = parent_pos + direction * distance
            node.offset_direction = direction.copy()
            node.cable_length = distance
            node.corkscrew_frequency = 5.0  # Fast spin for intercepts
            node.corkscrew_phase = np.random.uniform(0, 2 * np.pi)
            
            for i, child in enumerate(node.children):
                # Spread children in different directions
                angle = 2 * np.pi * i / max(len(node.children), 1)
                child_dir = np.array([
                    np.cos(angle),
                    np.sin(angle),
                    -0.2  # Slight downward
                ])
                child_dir = child_dir / np.linalg.norm(child_dir)
                child_dist = distance * 0.7  # Children closer
                spread_recursive(child, node.position, child_dir, child_dist)
        
        # Root at origin
        self.lattice.root.position = np.zeros(3)
        
        # Spread children
        for i, child in enumerate(self.lattice.root.children):
            angle = 2 * np.pi * i / max(len(self.lattice.root.children), 1)
            direction = np.array([
                np.cos(angle),
                np.sin(angle),
                0
            ])
            spread_recursive(child, np.zeros(3), direction, 10.0)
        
        # Combat state
        self.stats = CombatStats()
        self.time = 0.0
        self.running = False
        
        # Physics settings
        self.dt = 1/60  # 60 FPS physics
        self.freestream = np.array([0, 0, 0])  # Wind/current
        
        # Blade control state
        self.collective_pitch = np.radians(15)  # Default blade pitch
        self.target_rotation_speed = 300  # RPM target
        
    def build_observation(self) -> Dict:
        """
        Build observation vector for the DreamerV3 brain.
        This is what the champion sees to make decisions.
        """
        # Lattice state
        all_nodes = self.lattice.get_all_nodes()
        node_positions = np.array([n.position for n in all_nodes])
        node_velocities = np.array([n.velocity for n in all_nodes])
        
        # Threat state
        threats = self.spawner.threats
        if threats:
            threat_positions = np.array([t.position for t in threats if t.alive])
            threat_velocities = np.array([t.velocity for t in threats if t.alive])
            threat_radii = np.array([t.radius for t in threats if t.alive])
        else:
            threat_positions = np.zeros((0, 3))
            threat_velocities = np.zeros((0, 3))
            threat_radii = np.zeros(0)
        
        # Closest threat info
        closest = self.spawner.get_closest_threat(self.lattice.root.position)
        if closest:
            closest_dir = closest.position - self.lattice.root.position
            closest_dist = np.linalg.norm(closest_dir)
            closest_dir = closest_dir / (closest_dist + 0.01)
        else:
            closest_dir = np.zeros(3)
            closest_dist = self.arena_radius
        
        return {
            'node_positions': node_positions,
            'node_velocities': node_velocities,
            'n_nodes': len(all_nodes),
            'threat_positions': threat_positions,
            'threat_velocities': threat_velocities,
            'threat_radii': threat_radii,
            'n_threats': len(threats),
            'closest_threat_direction': closest_dir,
            'closest_threat_distance': closest_dist,
            'time': self.time,
            'stats': self.stats
        }
    
    def apply_brain_action(self, action: np.ndarray) -> None:
        """
        Apply action from DreamerV3 brain to the lattice.
        
        Action space (8 dims from champion):
          [0-2]: thrust_xyz - collective thrust direction
          [3-5]: torque_xyz - desired rotation
          [6]: collective_pitch - blade angle
          [7]: attack_mode - 0=defensive, 1=aggressive
        """
        if action is None:
            return
        
        # Parse action
        thrust_dir = action[0:3]
        torque = action[3:6]
        pitch_cmd = action[6]
        attack_mode = action[7]
        
        # Apply collective pitch to all blades
        target_pitch = np.radians(10 + 20 * pitch_cmd)  # 10-30 degrees
        for node in self.lattice.get_all_nodes():
            node.set_all_blade_pitch(target_pitch)
        
        # Apply thrust to mother drone
        thrust_mag = 50.0  # Base thrust
        if attack_mode > 0.5:
            thrust_mag *= 1.5  # Aggressive = more thrust
        
        self.lattice.root.thrust_vector = thrust_dir * thrust_mag
        
        # Apply rotation (changes corkscrew speeds)
        for node in self.lattice.get_all_nodes():
            if node.role.name in ['GYRO_ARM', 'SUB_TAB']:
                # Differential speed based on torque
                node.corkscrew_frequency = 2.0 + np.dot(torque, [1, 1, 1]) * 0.5
    
    def check_collisions(self) -> List[Tuple[str, str]]:
        """
        Check for collisions between lattice nodes and threats.
        Returns list of (node_id, threat_id) collision pairs.
        """
        collisions = []
        
        all_nodes = self.lattice.get_all_nodes()
        
        for threat in self.spawner.threats:
            if not threat.alive:
                continue
            
            for node in all_nodes:
                dist = np.linalg.norm(threat.position - node.position)
                
                # Collision check
                collision_radius = threat.radius + 0.5  # Node radius ~0.5m
                if dist < collision_radius:
                    collisions.append((node.node_id, threat.threat_id))
        
        return collisions
    
    def check_blade_intercepts(self) -> List[Tuple[str, str, float]]:
        """
        Check if rotating blades hit threats.
        
        SIMPLIFIED: Treat the tether cable between parent-child as a blade.
        When nodes spin (corkscrew), the cable sweeps a cylinder.
        Any threat inside this swept volume gets hit.
        
        Returns list of (blade_id, threat_id, damage) intercepts.
        """
        intercepts = []
        
        all_nodes = self.lattice.get_all_nodes()
        
        for node in all_nodes:
            if node.parent is None:
                continue
            
            # EVERY parent-child link is effectively a blade-tether
            parent_pos = node.parent.position
            child_pos = node.position
            blade_center = 0.5 * (parent_pos + child_pos)
            
            # Cable length = blade length
            cable_length = np.linalg.norm(child_pos - parent_pos)
            if cable_length < 0.1:
                continue  # Nodes too close, skip
            
            # Blade properties based on node type
            blade_id = f"BLADE_{node.node_id}"
            blade_mass = getattr(node, 'mass', 1.0) * 0.1  # Cable mass ~10% of node
            
            # Check threats against swept volume
            for threat in self.spawner.threats:
                if not threat.alive:
                    continue
                
                # Distance to blade center
                dist_to_center = np.linalg.norm(threat.position - blade_center)
                
                # Swept radius = half cable length + spin radius
                # Fast spinning blades sweep a larger cylinder
                spin_freq = getattr(node, 'corkscrew_frequency', 3.0)
                sweep_radius = cable_length / 2 + spin_freq * 0.5  # Faster spin = wider sweep
                
                if dist_to_center < sweep_radius + threat.radius:
                    # HIT! Calculate damage based on rotation speed
                    rotation_speed = spin_freq * 2 * np.pi  # rad/s
                    tip_speed = rotation_speed * cable_length / 2  # m/s
                    
                    # Kinetic energy damage - very effective
                    damage = 0.5 * blade_mass * tip_speed ** 2 * 1.0 + 50  # Base 50 + kinetic
                    
                    intercepts.append((blade_id, threat.threat_id, damage))
        
        return intercepts
    
    def step(self) -> Dict:
        """
        Advance simulation by one timestep.
        Returns observation dict.
        """
        dt = self.dt
        self.time += dt
        
        # Update threats
        self.spawner.target = self.lattice.root.position
        self.spawner.update(dt)
        
        # Build observation
        obs = self.build_observation()
        
        # Spin all nodes (rotate the lattice)
        for node in self.lattice.get_all_nodes():
            freq = getattr(node, 'corkscrew_frequency', 2.0)
            phase = getattr(node, 'corkscrew_phase', 0.0)
            node.corkscrew_phase = phase + freq * dt * 2 * np.pi
            
            # Update child node positions based on rotation
            if node.parent is not None:
                parent_pos = node.parent.position
                # Compute orbital position
                offset_dir = getattr(node, 'offset_direction', np.array([1, 0, 0]))
                cable_len = getattr(node, 'cable_length', 5.0)
                
                # Rotate offset direction
                angle = node.corkscrew_phase
                c, s = np.cos(angle), np.sin(angle)
                # Rotate in XY plane
                new_offset = np.array([
                    offset_dir[0] * c - offset_dir[1] * s,
                    offset_dir[0] * s + offset_dir[1] * c,
                    offset_dir[2]
                ])
                node.position = parent_pos + new_offset * cable_len
        
        # Get brain action (if available)
        if self.brain is not None:
            action = self.brain.infer(obs)
            self.apply_brain_action(action)
        else:
            # Default behavior: track closest threat
            closest = self.spawner.get_closest_threat(self.lattice.root.position)
            if closest:
                # Point thrust toward threat
                to_threat = closest.position - self.lattice.root.position
                to_threat = to_threat / (np.linalg.norm(to_threat) + 0.01)
                self.lattice.root.thrust_vector = to_threat * 30
        
        # Update lattice physics
        self.lattice.step(dt)
        
        # Check blade intercepts (slashing damage)
        intercepts = self.check_blade_intercepts()
        for blade_id, threat_id, damage in intercepts:
            for threat in self.spawner.threats:
                if threat.threat_id == threat_id:
                    destroyed = threat.take_damage(damage)
                    self.stats.damage_dealt += damage
                    if destroyed:
                        self.stats.threats_destroyed += 1
                        print(f"[HIT] {blade_id} destroyed {threat_id}!")
        
        # Check collisions (damage to lattice)
        collisions = self.check_collisions()
        for node_id, threat_id in collisions:
            for threat in self.spawner.threats:
                if threat.threat_id == threat_id and threat.alive:
                    self.stats.damage_taken += threat.damage
                    threat.alive = False  # Threat consumed
                    print(f"[COLLISION] {threat_id} hit {node_id}! Damage: {threat.damage:.0f}")
        
        # Check for threats that escaped (passed the origin)
        for threat in self.spawner.threats:
            if threat.alive:
                dist_to_center = np.linalg.norm(threat.position - self.lattice.root.position)
                # Threat passed through
                if dist_to_center < 5.0:
                    threat.alive = False
                    self.stats.threats_escaped += 1
        
        self.stats.wave_survived = self.spawner.wave_number
        self.stats.blades_active = len(self.lattice.root.get_all_blades())
        
        return obs
    
    def run(self, max_time: float = 60.0, callback=None) -> CombatStats:
        """
        Run the combat simulation.
        
        Args:
            max_time: Maximum simulation time (seconds)
            callback: Function called each frame with (arena, obs)
        
        Returns:
            Final combat statistics
        """
        self.running = True
        self.time = 0.0
        
        # Initial wave
        self.spawner.spawn_wave()
        
        while self.running and self.time < max_time:
            obs = self.step()
            
            if callback:
                callback(self, obs)
            
            # Check end conditions
            if self.stats.damage_taken > 500:
                print("[GAME OVER] Too much damage!")
                break
        
        self.running = False
        return self.stats


def create_demo_arena():
    """Create a demo arena for testing"""
    import sys
    import os
    
    # Add project root to path
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    # Import the lattice factory
    from src.entities.quine_node import create_matryoshka_lattice
    
    # Create the lattice
    lattice = create_matryoshka_lattice(
        n_spines=4,
        vertebrae_per_spine=2  # Smaller for demo
    )
    
    # Create arena
    arena = CombatArena(lattice)
    
    return arena


if __name__ == "__main__":
    print("=" * 60)
    print("COMBAT ARENA - ASTEROID DEFENSE")
    print("=" * 60)
    
    arena = create_demo_arena()
    
    print(f"\nLattice: {arena.lattice.total_nodes} quine agents")
    print(f"Arena radius: {arena.arena_radius}m")
    print(f"Starting combat simulation...\n")
    
    # Run for 30 seconds (text mode)
    def print_status(arena, obs):
        if int(arena.time * 10) % 50 == 0:  # Every 5 seconds
            print(f"[t={arena.time:.1f}s] Threats: {obs['n_threats']} | "
                  f"Destroyed: {arena.stats.threats_destroyed} | "
                  f"Damage: {arena.stats.damage_taken:.0f} | "
                  f"Score: {arena.stats.score}")
    
    stats = arena.run(max_time=30.0, callback=print_status)
    
    print("\n" + "=" * 60)
    print("FINAL STATS")
    print("=" * 60)
    print(f"  Threats destroyed: {stats.threats_destroyed}")
    print(f"  Threats escaped: {stats.threats_escaped}")
    print(f"  Damage taken: {stats.damage_taken:.0f}")
    print(f"  Damage dealt: {stats.damage_dealt:.0f}")
    print(f"  Waves survived: {stats.wave_survived}")
    print(f"  FINAL SCORE: {stats.score}")
    print("=" * 60)
