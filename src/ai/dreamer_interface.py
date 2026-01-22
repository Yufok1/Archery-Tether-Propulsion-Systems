"""
DREAMER BRAIN INTERFACE
=======================

Connects the champion_gen42 DreamerV3 quine brain to the combat arena.
The brain processes observations and outputs actions to control the
blade-tether lattice.

OBSERVATION SPACE:
  - Threat positions, velocities, radii (normalized)
  - Lattice node positions, velocities
  - Closest threat direction and distance
  - Time, score, damage

ACTION SPACE (8 continuous):
  [0-2]: thrust_xyz - collective thrust direction
  [3-5]: torque_xyz - desired rotation
  [6]: collective_pitch - blade angle (0-1)
  [7]: attack_mode - defensive (0) to aggressive (1)
"""

import numpy as np
import sys
import os
from pathlib import Path
from typing import Dict, Any, Optional

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class DreamerBrainInterface:
    """
    Interface between the DreamerV3 champion and the combat arena.
    
    Handles:
      - Observation encoding (3D world state -> flat vector)
      - Action decoding (8-dim action -> lattice commands)
      - State persistence (RSSM hidden state)
    """
    
    # Default search paths for champion capsules
    DEFAULT_CHAMPION_PATHS = [
        # HuggingFace key-data repo (evolved champions) - gen42 is latest with bootstrap fix
        Path("F:/End-Game/glassboxgames/children/key-data-repo/models/champion_gen42.py"),
        # Project root fallback
        PROJECT_ROOT / "champion_gen42.py",
    ]
    
    def __init__(self, champion_path: str = None):
        # Find champion path
        if champion_path:
            self.champion_path = Path(champion_path)
        else:
            # Search default locations
            self.champion_path = None
            for path in self.DEFAULT_CHAMPION_PATHS:
                if path.exists():
                    self.champion_path = path
                    break
            if self.champion_path is None:
                self.champion_path = self.DEFAULT_CHAMPION_PATHS[-1]  # fallback
        
        self.champion = None
        self.brain = None
        
        # State tracking
        self.hidden_state = None
        self.last_action = np.zeros(8)
        self.frame_count = 0
        
        # Observation normalization
        self.obs_mean = None
        self.obs_std = None
        
        # Action scaling
        self.action_scale = np.array([
            50.0, 50.0, 50.0,   # thrust xyz max magnitude
            2.0, 2.0, 2.0,      # torque xyz scaling
            1.0,                 # pitch (already 0-1)
            1.0                  # attack mode (0-1)
        ])
        
        # Performance tracking
        self.inference_times = []
        
        self._load_champion()
    
    def _load_champion(self):
        """Load the champion DreamerV3 brain from capsule file."""
        import importlib.util
        
        try:
            print(f"[BRAIN] Loading champion from: {self.champion_path}")
            
            if not self.champion_path.exists():
                raise FileNotFoundError(f"Champion not found: {self.champion_path}")
            
            # Dynamic import from file path
            spec = importlib.util.spec_from_file_location("champion", str(self.champion_path))
            champion_module = importlib.util.module_from_spec(spec)
            sys.modules["champion"] = champion_module
            spec.loader.exec_module(champion_module)
            self.champion = champion_module
            
            # Get the quine brain - try multiple interfaces
            if hasattr(champion_module, 'QuineBrain'):
                self.brain = champion_module.QuineBrain()
                print(f"[BRAIN] QuineBrain loaded from {self.champion_path.name}")
                if hasattr(self.brain, 'get_merkle_hash'):
                    print(f"[BRAIN] Merkle hash: {self.brain.get_merkle_hash()[:32]}...")
            elif hasattr(champion_module, 'get_quine_brain'):
                self.brain = champion_module.get_quine_brain()
                print(f"[BRAIN] Loaded DreamerV3 quine brain from {self.champion_path.name}")
            elif hasattr(champion_module, 'CapsuleAgent'):
                agent = champion_module.CapsuleAgent(observe=False, observe_visual=False)
                self.brain = agent.brain
                self._capsule_agent = agent  # Keep reference
                print(f"[BRAIN] CapsuleAgent loaded from {self.champion_path.name}")
            else:
                # Fallback: try to use module directly
                print("[BRAIN] Using champion forward() directly")
            
            # Show generation info if available
            if hasattr(champion_module, '_GENERATION'):
                print(f"[BRAIN] Generation: {champion_module._GENERATION}")
            if hasattr(champion_module, '_FITNESS'):
                print(f"[BRAIN] Fitness: {champion_module._FITNESS:.4f}")
            if hasattr(champion_module, '_QUINE_HASH'):
                print(f"[BRAIN] Quine hash: {champion_module._QUINE_HASH[:32]}...")
            
            # Verify integrity
            if hasattr(champion_module, 'verify_quine_integrity'):
                if champion_module.verify_quine_integrity():
                    print("[BRAIN] Quine integrity VERIFIED ✓")
                else:
                    print("[BRAIN] Quine integrity check FAILED!")
                
        except Exception as e:
            print(f"[BRAIN] Failed to load champion: {e}")
            print("[BRAIN] Using fallback reactive controller")
            self.champion = None
            self.brain = None
    
    def encode_observation(self, obs: Dict) -> np.ndarray:
        """
        Encode combat arena observation into flat vector for DreamerV3.
        
        Input obs dict:
          - node_positions: (N, 3)
          - node_velocities: (N, 3)
          - threat_positions: (M, 3)
          - threat_velocities: (M, 3)
          - threat_radii: (M,)
          - closest_threat_direction: (3,)
          - closest_threat_distance: float
          - time: float
          - stats: CombatStats
        
        Output: flat numpy array suitable for DreamerV3 input
        """
        # Fixed-size encoding for variable-length inputs
        max_nodes = 100
        max_threats = 50
        
        # Node features (position, velocity) - pad/truncate to max_nodes
        node_pos = obs.get('node_positions', np.zeros((0, 3)))
        node_vel = obs.get('node_velocities', np.zeros((0, 3)))
        
        n_nodes = min(len(node_pos), max_nodes)
        node_features = np.zeros((max_nodes, 6))
        if n_nodes > 0:
            node_features[:n_nodes, :3] = node_pos[:n_nodes] / 500.0  # Normalize to arena
            node_features[:n_nodes, 3:] = node_vel[:n_nodes] / 100.0  # Normalize velocity
        
        # Threat features (position, velocity, radius) - pad/truncate
        threat_pos = obs.get('threat_positions', np.zeros((0, 3)))
        threat_vel = obs.get('threat_velocities', np.zeros((0, 3)))
        threat_radii = obs.get('threat_radii', np.zeros(0))
        
        n_threats = min(len(threat_pos), max_threats)
        threat_features = np.zeros((max_threats, 7))
        if n_threats > 0:
            threat_features[:n_threats, :3] = threat_pos[:n_threats] / 500.0
            threat_features[:n_threats, 3:6] = threat_vel[:n_threats] / 100.0
            threat_features[:n_threats, 6] = threat_radii[:n_threats] / 10.0
        
        # Scalar features
        closest_dir = obs.get('closest_threat_direction', np.zeros(3))
        closest_dist = obs.get('closest_threat_distance', 500.0) / 500.0
        time_norm = obs.get('time', 0.0) / 60.0  # Normalize to 1 minute
        
        # Combat stats
        stats = obs.get('stats', None)
        if stats:
            score = float(stats.score) / 10000.0  # Normalize
            damage = float(stats.damage_taken) / 1000.0
            destroyed = float(stats.threats_destroyed) / 100.0
        else:
            score = 0.0
            damage = 0.0
            destroyed = 0.0
        
        # Flatten and concatenate
        encoded = np.concatenate([
            node_features.flatten(),    # max_nodes * 6 = 600
            threat_features.flatten(),  # max_threats * 7 = 350
            closest_dir,                # 3
            [closest_dist],             # 1
            [time_norm],                # 1
            [score, damage, destroyed], # 3
            [float(n_nodes) / max_nodes, float(n_threats) / max_threats],  # 2
            self.last_action            # 8 (previous action for temporal)
        ])
        
        return encoded.astype(np.float32)
    
    def decode_action(self, raw_action: np.ndarray) -> np.ndarray:
        """
        Decode raw network output to action commands.
        
        Raw action is typically in [-1, 1] or logits.
        Output is scaled action vector.
        """
        # Clamp to [-1, 1]
        action = np.clip(raw_action, -1, 1)
        
        # Scale actions
        scaled = np.zeros(8)
        
        # Thrust direction (normalize then scale)
        thrust = action[:3]
        thrust_norm = np.linalg.norm(thrust)
        if thrust_norm > 0.01:
            thrust = thrust / thrust_norm
        scaled[:3] = thrust * self.action_scale[:3]
        
        # Torque (direct scaling)
        scaled[3:6] = action[3:6] * self.action_scale[3:6]
        
        # Pitch and attack mode (0-1 range)
        scaled[6] = (action[6] + 1) / 2  # Map [-1,1] to [0,1]
        scaled[7] = (action[7] + 1) / 2
        
        return scaled
    
    def infer(self, obs: Dict) -> np.ndarray:
        """
        Run inference to get action from observation.
        
        Uses the DreamerV3 brain if available, otherwise falls back
        to reactive controller.
        """
        self.frame_count += 1
        
        # Encode observation
        encoded_obs = self.encode_observation(obs)
        
        # Try DreamerV3 inference
        if self.champion is not None:
            try:
                # Call champion forward
                if hasattr(self.champion, 'forward'):
                    raw_action = self.champion.forward(encoded_obs)
                    action = self.decode_action(raw_action)
                    self.last_action = action
                    return action
            except Exception as e:
                if self.frame_count % 100 == 0:
                    print(f"[BRAIN] Inference error: {e}")
        
        # Fallback: reactive controller
        return self._reactive_fallback(obs)
    
    def _reactive_fallback(self, obs: Dict) -> np.ndarray:
        """
        Reactive controller fallback when DreamerV3 unavailable.
        
        Simple rules:
        - Track closest threat
        - Defensive posture when threatened
        - Aggressive when safe
        """
        action = np.zeros(8)
        
        closest_dir = obs.get('closest_threat_direction', np.zeros(3))
        closest_dist = obs.get('closest_threat_distance', 500.0)
        
        # Normalize threat direction
        threat_norm = np.linalg.norm(closest_dir)
        if threat_norm > 0.01:
            closest_dir = closest_dir / threat_norm
        
        # Threat response based on distance
        danger_threshold = 100.0
        
        if closest_dist < danger_threshold:
            # DEFENSIVE MODE
            # Move perpendicular to threat (dodge)
            perp = np.array([-closest_dir[1], closest_dir[0], closest_dir[2] * 0.5])
            perp = perp / (np.linalg.norm(perp) + 0.01)
            
            action[0:3] = perp * 30  # Dodge thrust
            action[3:6] = closest_dir * 1.5  # Torque toward threat
            action[6] = 0.8  # High blade pitch for intercept
            action[7] = 0.3  # Defensive
        else:
            # PATROL MODE
            # Slow rotation, watch all directions
            t = self.frame_count * 0.01
            action[0:3] = np.array([
                np.cos(t) * 10,
                np.sin(t) * 10,
                np.sin(t * 0.5) * 5
            ])
            action[3:6] = np.array([0.5, 0.5, 0.2])  # Gentle spin
            action[6] = 0.5  # Medium pitch
            action[7] = 0.5  # Balanced
        
        self.last_action = action
        return action
    
    def reset(self):
        """Reset brain state for new episode."""
        self.hidden_state = None
        self.last_action = np.zeros(8)
        self.frame_count = 0


class AdaptiveController:
    """
    Higher-level controller that adapts the DreamerV3 output
    to the specific lattice configuration.
    
    Handles:
      - Formation-specific thrust distribution
      - Blade coordination across spines
      - Emergency override (HOLD system)
    """
    
    def __init__(self, brain: DreamerBrainInterface):
        self.brain = brain
        self.lattice = None
        
        # Formation modes
        self.formation = "defensive_sphere"  # or "attack_wedge", "patrol_spiral"
        
        # HOLD state
        self.hold_active = False
        self.hold_reason = None
    
    def set_lattice(self, lattice):
        """Connect to lattice for formation control."""
        self.lattice = lattice
    
    def activate_hold(self, reason: str = "User override"):
        """Activate HOLD - freeze all actions."""
        self.hold_active = True
        self.hold_reason = reason
        print(f"[HOLD ACTIVATED] {reason}")
    
    def release_hold(self):
        """Release HOLD - resume autonomous control."""
        self.hold_active = False
        self.hold_reason = None
        print("[HOLD RELEASED] Resuming autonomous control")
    
    def compute_action(self, obs: Dict) -> np.ndarray:
        """
        Compute action with formation and safety constraints.
        """
        # Check HOLD
        if self.hold_active:
            return np.zeros(8)  # No action during HOLD
        
        # Get base action from brain
        base_action = self.brain.infer(obs)
        
        # Apply formation modifiers
        if self.formation == "defensive_sphere":
            # Emphasize even coverage
            base_action[3:6] *= 0.5  # Reduce rapid rotation
        elif self.formation == "attack_wedge":
            # Concentrate forward
            base_action[0] *= 1.5
            base_action[7] = max(base_action[7], 0.7)  # Force aggressive
        
        # Safety limits
        base_action = self._apply_safety_limits(base_action, obs)
        
        return base_action
    
    def _apply_safety_limits(self, action: np.ndarray, obs: Dict) -> np.ndarray:
        """Apply safety constraints to action."""
        # Limit total thrust
        thrust_mag = np.linalg.norm(action[:3])
        if thrust_mag > 100:
            action[:3] = action[:3] / thrust_mag * 100
        
        # Limit rotation rate
        torque_mag = np.linalg.norm(action[3:6])
        if torque_mag > 3.0:
            action[3:6] = action[3:6] / torque_mag * 3.0
        
        return action


def create_brain_interface() -> DreamerBrainInterface:
    """Factory function to create brain interface."""
    return DreamerBrainInterface()


if __name__ == "__main__":
    print("=" * 60)
    print("DREAMER BRAIN INTERFACE TEST")
    print("=" * 60)
    
    # Create brain
    brain = create_brain_interface()
    
    # Create mock observation
    mock_obs = {
        'node_positions': np.random.randn(50, 3) * 50,
        'node_velocities': np.random.randn(50, 3) * 5,
        'n_nodes': 50,
        'threat_positions': np.array([[100, 50, 20], [80, -30, 10]]),
        'threat_velocities': np.array([[-20, -10, -5], [-15, 5, -2]]),
        'threat_radii': np.array([3.0, 5.0]),
        'n_threats': 2,
        'closest_threat_direction': np.array([0.8, 0.5, 0.2]),
        'closest_threat_distance': 120.0,
        'time': 15.0,
        'stats': type('Stats', (), {'score': 500, 'damage_taken': 50, 'threats_destroyed': 5})()
    }
    
    # Test inference
    print("\nRunning inference...")
    for i in range(5):
        action = brain.infer(mock_obs)
        print(f"  Frame {i}: thrust={action[:3]}, pitch={action[6]:.2f}, attack={action[7]:.2f}")
    
    print("\n" + "=" * 60)
    print("Brain interface ready for combat!")
    print("=" * 60)
