"""
TETHER NETWORK INTELLIGENCE
============================

The wires ARE the brain. In a hubless dervish:
- Each sail node has a DreamerV3 replicant (local sensing/actuation)
- Tethers carry TENSION SIGNALS between nodes
- No central authority - intelligence EMERGES from tension patterns
- Champion model replicates to each node at spawn from GENESIS

Architecture:
                    ┌─────────────┐
                    │   GENESIS   │  ← Original champion model
                    │ (champion)  │
                    └──────┬──────┘
                           │ REPLICATES AT SPAWN
            ┌──────────────┼──────────────┐
            ▼              ▼              ▼
      ┌──────────┐   ┌──────────┐   ┌──────────┐
      │ Sail A   │   │ Sail B   │   │ Sail C   │
      │ Dreamer  │═══│ Dreamer  │═══│ Dreamer  │
      │ Replicant│   │ Replicant│   │ Replicant│
      └──────────┘   └──────────┘   └──────────┘
            ║              ║              ║
            ╚══════════════╩══════════════╝
                    TETHER TENSION BUS
                    (signals via strain)

Key insight: The tether tension IS the communication channel.
- High tension = "I'm pulling hard" = thrust signal
- Oscillating tension = phase sync for collective maneuvers
- Tension gradients = directional command propagation

The GENESIS champion is like DNA - copied to each cell at birth,
but then each cell senses and acts locally while coordinating
through the physical medium (tethers).
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any
from pathlib import Path
import hashlib
import time

# Try to import the champion DreamerV3
try:
    from champion_gen42 import DreamerV3RSSM, TrainingConfig
    CHAMPION_AVAILABLE = True
except ImportError:
    CHAMPION_AVAILABLE = False
    print("[TETHER-NET] Champion not available - using stub")


@dataclass
class TensionSignal:
    """A signal encoded in tether tension."""
    tension_magnitude: float      # N - raw tension
    tension_rate: float           # N/s - rate of change
    oscillation_freq: float       # Hz - if pulsing
    oscillation_phase: float      # rad - phase in oscillation
    
    # Decoded meaning
    thrust_request: float = 0.0   # -1 to 1 (relax to pull)
    phase_sync: float = 0.0       # Target phase for collective
    emergency: bool = False       # High-freq burst = emergency


@dataclass
class SailNode:
    """
    A sail with a local DreamerV3 replicant brain.
    
    Each sail:
    - Senses its local apparent wind, position, tension
    - Has a full DreamerV3 copy for local decisions
    - Communicates via tether tension modulation
    - Shares the same GENESIS origin as all siblings
    """
    node_id: str
    genesis_hash: str              # Hash of original champion
    
    # Physical state (from physics sim)
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    
    # Sail control
    pitch: float = 0.0             # Local sail pitch angle
    yaw: float = 0.0               # Local sail yaw
    
    # DreamerV3 replicant (local brain)
    brain: Any = None              # DreamerV3RSSM instance
    brain_state: np.ndarray = field(default_factory=lambda: np.zeros(256))
    
    # Tether connections (node_id -> TensionSignal)
    tether_signals: Dict[str, TensionSignal] = field(default_factory=dict)
    
    # Observations for learning
    last_observation: Optional[np.ndarray] = None
    last_action: Optional[np.ndarray] = None
    cumulative_reward: float = 0.0
    
    def sense(self, physics_node, connected_tensions: Dict[str, float]) -> np.ndarray:
        """
        Build observation from local sensing + tether tensions.
        
        The sail senses:
        - Its own velocity (apparent wind)
        - Tether tensions from neighbors (the distributed signal)
        - Its orientation
        """
        # Local apparent wind (velocity relative to air)
        apparent_wind = -self.velocity  # Simplified - no freestream
        wind_speed = np.linalg.norm(apparent_wind)
        wind_dir = apparent_wind / max(wind_speed, 0.1)
        
        # Tether tension signals (THE DISTRIBUTED INTELLIGENCE)
        tension_obs = []
        for neighbor_id, tension in connected_tensions.items():
            if neighbor_id not in self.tether_signals:
                self.tether_signals[neighbor_id] = TensionSignal(
                    tension_magnitude=tension,
                    tension_rate=0.0,
                    oscillation_freq=0.0,
                    oscillation_phase=0.0
                )
            else:
                # Update with rate
                old_tension = self.tether_signals[neighbor_id].tension_magnitude
                self.tether_signals[neighbor_id].tension_rate = (tension - old_tension) / 0.01
                self.tether_signals[neighbor_id].tension_magnitude = tension
            
            tension_obs.extend([
                tension / 1000.0,  # Normalized tension
                self.tether_signals[neighbor_id].tension_rate / 100.0
            ])
        
        # Pad to fixed size (max 6 neighbors)
        while len(tension_obs) < 12:
            tension_obs.append(0.0)
        
        # Build full observation
        obs = np.array([
            # Wind sensing
            wind_speed / 50.0,
            wind_dir[0], wind_dir[1], wind_dir[2],
            
            # Position (relative to origin)
            self.position[0] / 100.0,
            self.position[1] / 100.0,
            self.position[2] / 100.0,
            
            # Velocity
            self.velocity[0] / 50.0,
            self.velocity[1] / 50.0,
            self.velocity[2] / 50.0,
            
            # Current sail state
            np.sin(self.pitch),
            np.cos(self.pitch),
            np.sin(self.yaw),
            np.cos(self.yaw),
            
            # Tether tensions (THE KEY DISTRIBUTED SIGNAL)
            *tension_obs[:12]
        ], dtype=np.float32)
        
        self.last_observation = obs
        return obs
    
    def act(self, observation: np.ndarray) -> np.ndarray:
        """
        Local brain decides action based on observation.
        
        Action space:
        - pitch_delta: Change in sail pitch
        - yaw_delta: Change in sail yaw  
        - tension_signal: Outgoing tension modulation (for neighbors)
        """
        if self.brain is not None:
            # Use DreamerV3 replicant
            action, self.brain_state = self.brain.infer(observation, self.brain_state)
        else:
            # Simple reactive controller
            # Extract tension signals
            tension_avg = np.mean(observation[14:26])
            
            # If tensions are high, we're being pulled - adjust pitch to generate thrust
            pitch_delta = -tension_avg * 0.1
            
            # Maintain spin by yawing into apparent wind
            wind_dir = observation[1:4]
            yaw_delta = np.arctan2(wind_dir[1], wind_dir[0]) * 0.05
            
            # Signal neighbors through tension
            tension_signal = observation[6] * 0.1  # Altitude-based
            
            action = np.array([pitch_delta, yaw_delta, tension_signal])
        
        self.last_action = action
        return action
    
    def apply_action(self, action: np.ndarray):
        """Apply action to sail controls."""
        self.pitch += action[0] * 0.1
        self.pitch = np.clip(self.pitch, -np.pi/4, np.pi/4)
        
        self.yaw += action[1] * 0.1
        self.yaw = self.yaw % (2 * np.pi)


class TetherNetwork:
    """
    The distributed intelligence living in the tether network.
    
    This is NOT a central controller - it's a container for the
    network of sail nodes and their tether connections. Intelligence
    emerges from:
    
    1. Local decisions by each sail's DreamerV3 replicant
    2. Tension signal propagation through tethers
    3. Collective behavior from the interaction
    
    The GENESIS model is copied to each sail at spawn time,
    then each sail learns and adapts locally while sharing
    experience back to the collective.
    """
    
    def __init__(self, genesis_path: Optional[Path] = None):
        """
        Initialize network with GENESIS model.
        
        Args:
            genesis_path: Path to champion model file. All sails
                         get a copy of this at spawn.
        """
        self.nodes: Dict[str, SailNode] = {}
        self.tethers: Dict[str, List[str]] = {}  # node_id -> [connected_ids]
        
        # Load genesis model
        self.genesis_path = genesis_path
        self.genesis_hash = self._compute_genesis_hash()
        self.genesis_brain = None
        
        if genesis_path and CHAMPION_AVAILABLE:
            try:
                self.genesis_brain = DreamerV3RSSM(TrainingConfig())
                self.genesis_brain.load(genesis_path)
                print(f"[TETHER-NET] Loaded genesis from {genesis_path}")
            except Exception as e:
                print(f"[TETHER-NET] Could not load genesis: {e}")
        
        # Collective state
        self.collective_observations: List[Dict] = []
        self.step_count = 0
    
    def _compute_genesis_hash(self) -> str:
        """Compute hash of genesis model for provenance."""
        if self.genesis_path and self.genesis_path.exists():
            with open(self.genesis_path, 'rb') as f:
                return hashlib.sha256(f.read()).hexdigest()[:16]
        return "no_genesis_" + hashlib.sha256(str(time.time()).encode()).hexdigest()[:8]
    
    def spawn_sail(self, node_id: str, position: np.ndarray) -> SailNode:
        """
        Spawn a new sail with a REPLICATED brain.
        
        The brain is a copy of the genesis model - same weights,
        but independent state. Like cells with the same DNA.
        """
        node = SailNode(
            node_id=node_id,
            genesis_hash=self.genesis_hash,
            position=position.copy()
        )
        
        # Replicate genesis brain
        if self.genesis_brain is not None:
            node.brain = self.genesis_brain.clone()
            print(f"[TETHER-NET] Replicated brain to sail {node_id}")
        
        self.nodes[node_id] = node
        return node
    
    def connect_tether(self, node_a: str, node_b: str):
        """Add tether connection between two sails."""
        if node_a not in self.tethers:
            self.tethers[node_a] = []
        if node_b not in self.tethers:
            self.tethers[node_b] = []
        
        if node_b not in self.tethers[node_a]:
            self.tethers[node_a].append(node_b)
        if node_a not in self.tethers[node_b]:
            self.tethers[node_b].append(node_a)
    
    def step(self, physics_state: Dict[str, Any], dt: float) -> Dict[str, np.ndarray]:
        """
        Step the distributed intelligence.
        
        1. Each sail senses (including tether tensions)
        2. Each sail's local brain decides action
        3. Actions applied to physics
        4. Tension changes propagate through network
        
        Returns actions for each sail.
        """
        actions = {}
        
        # Get current tensions from physics
        tensions = physics_state.get('tensions', {})
        
        for node_id, node in self.nodes.items():
            # Update position/velocity from physics
            if node_id in physics_state.get('positions', {}):
                node.position = physics_state['positions'][node_id]
            if node_id in physics_state.get('velocities', {}):
                node.velocity = physics_state['velocities'][node_id]
            
            # Get tensions from connected tethers
            connected_tensions = {}
            for neighbor_id in self.tethers.get(node_id, []):
                tether_key = tuple(sorted([node_id, neighbor_id]))
                if tether_key in tensions:
                    connected_tensions[neighbor_id] = tensions[tether_key]
            
            # Sense
            obs = node.sense(None, connected_tensions)
            
            # Act (local brain decision)
            action = node.act(obs)
            
            # Apply locally
            node.apply_action(action)
            
            actions[node_id] = action
        
        self.step_count += 1
        return actions
    
    def get_collective_pitch_command(self) -> float:
        """
        Emergent collective pitch from all sails.
        
        The "collective" command isn't imposed from above -
        it EMERGES from the average of all local decisions.
        """
        if not self.nodes:
            return 0.0
        return np.mean([n.pitch for n in self.nodes.values()])
    
    def get_tension_pattern(self) -> np.ndarray:
        """
        Get the current tension pattern across network.
        
        This IS the distributed state - the pattern of tensions
        encodes the collective intention.
        """
        patterns = []
        for node_id, node in self.nodes.items():
            for neighbor_id, signal in node.tether_signals.items():
                patterns.append([
                    signal.tension_magnitude,
                    signal.tension_rate,
                    signal.thrust_request
                ])
        
        if not patterns:
            return np.zeros((1, 3))
        return np.array(patterns)
    
    def broadcast_command(self, command_type: str, value: float):
        """
        External command (like Yondu's whistle).
        
        This doesn't override local brains - it modulates
        the tension network which the brains then respond to.
        """
        # Encode command as tension oscillation
        freq = {
            'forward': 1.0,
            'back': 2.0,
            'left': 3.0,
            'right': 4.0,
            'up': 5.0,
            'down': 6.0,
            'attack': 10.0,
            'return': 0.5
        }.get(command_type, 1.0)
        
        # Set oscillation on all tethers
        for node in self.nodes.values():
            for signal in node.tether_signals.values():
                signal.oscillation_freq = freq
                signal.thrust_request = value
                signal.emergency = (command_type == 'attack')
        
        print(f"[TETHER-NET] Broadcast: {command_type} = {value:.2f}")
    
    def collect_experience(self) -> List[Dict]:
        """
        Collect observations from all sails for collective learning.
        
        All sails share experience back to improve the GENESIS model.
        """
        experiences = []
        for node_id, node in self.nodes.items():
            if node.last_observation is not None and node.last_action is not None:
                experiences.append({
                    'node_id': node_id,
                    'genesis_hash': node.genesis_hash,
                    'observation': node.last_observation.copy(),
                    'action': node.last_action.copy(),
                    'reward': node.cumulative_reward,
                    'timestamp': time.time()
                })
        return experiences


def create_dervish_network(n_sails: int = 6, genesis_path: Optional[Path] = None) -> TetherNetwork:
    """
    Create a hubless dervish with distributed sail intelligence.
    
    Each sail gets a replicated brain from genesis.
    Tethers connect in ring topology.
    """
    network = TetherNetwork(genesis_path=genesis_path)
    
    # Spawn sails in ring
    for i in range(n_sails):
        angle = 2 * np.pi * i / n_sails
        pos = np.array([
            8.0 * np.cos(angle),
            8.0 * np.sin(angle),
            50.0  # Starting altitude
        ])
        network.spawn_sail(f"sail_{i}", pos)
    
    # Connect tethers in ring
    for i in range(n_sails):
        next_i = (i + 1) % n_sails
        network.connect_tether(f"sail_{i}", f"sail_{next_i}")
    
    print(f"[TETHER-NET] Created dervish with {n_sails} sails")
    print(f"[TETHER-NET] Genesis: {network.genesis_hash}")
    print(f"[TETHER-NET] Topology: ring with {n_sails} tethers")
    
    return network


if __name__ == "__main__":
    # Test the network
    print("\n=== TETHER NETWORK TEST ===\n")
    
    network = create_dervish_network(n_sails=6)
    
    # Simulate some steps
    for step in range(10):
        # Mock physics state
        physics = {
            'positions': {f"sail_{i}": network.nodes[f"sail_{i}"].position + np.random.randn(3) * 0.1 
                         for i in range(6)},
            'velocities': {f"sail_{i}": np.array([0, 10, 0]) + np.random.randn(3) * 2 
                          for i in range(6)},
            'tensions': {
                (f"sail_{i}", f"sail_{(i+1)%6}"): 100 + np.random.randn() * 20
                for i in range(6)
            }
        }
        
        actions = network.step(physics, dt=0.01)
        
        if step % 3 == 0:
            print(f"Step {step}:")
            print(f"  Collective pitch: {np.degrees(network.get_collective_pitch_command()):.1f}°")
            pattern = network.get_tension_pattern()
            print(f"  Tension pattern: {pattern.shape[0]} signals, avg={pattern[:,0].mean():.1f}N")
    
    # Test command broadcast
    print("\n--- Broadcasting FORWARD command ---")
    network.broadcast_command('forward', 0.8)
    
    # More steps to see response
    for step in range(5):
        actions = network.step(physics, dt=0.01)
    
    print(f"\nFinal collective pitch: {np.degrees(network.get_collective_pitch_command()):.1f}°")
    print("\n[TETHER-NET] Test complete!")
