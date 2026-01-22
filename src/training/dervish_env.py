"""
Hubless Dervish Training Environment
=====================================

Gymnasium environment for training DreamerV3 on the hubless constellation.

The agent learns to:
1. Control collective pitch (altitude)
2. Control cyclic pitch (thrust direction)
3. Manage spin rate
4. Navigate to targets
5. Maintain constellation stability

Observation space: Node positions, velocities, tensions, spin phase
Action space: Collective pitch, cyclic amplitude, cyclic phase per node
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Tuple, Optional, Any
from dataclasses import dataclass

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.physics.hubless_dervish import (
    HublessDervish, 
    AirfoilNode,
    create_ring_constellation,
    BolaLauncher
)


@dataclass
class DervishEnvConfig:
    """Configuration for the training environment."""
    n_nodes: int = 4
    constellation_radius: float = 15.0
    max_episode_steps: int = 1000
    dt: float = 0.02
    
    # Reward weights
    altitude_weight: float = 1.0
    target_weight: float = 2.0
    stability_weight: float = 0.5
    energy_weight: float = 0.1
    
    # Initial conditions
    launch_from_throw: bool = True
    initial_altitude: float = 50.0
    
    # Target
    target_position: Optional[np.ndarray] = None


class HublessDervishEnv(gym.Env):
    """
    Gymnasium environment for training on the hubless dervish.
    
    The constellation is thrown (bola-style) and must learn to fly.
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    
    def __init__(self, config: DervishEnvConfig = None, render_mode: str = None):
        super().__init__()
        
        self.config = config or DervishEnvConfig()
        self.render_mode = render_mode
        
        # Create constellation
        self.dervish: HublessDervish = None
        self.launcher: BolaLauncher = None
        
        # State
        self.step_count = 0
        self.episode_reward = 0.0
        self.target = np.array([100, 100, 50])  # Default target
        
        # Define spaces
        n = self.config.n_nodes
        
        # Observation: per-node (pos, vel) + global (centroid, spin, target_dir)
        # Per node: 6 (pos xyz, vel xyz)
        # Global: 3 (centroid) + 1 (spin_phase) + 3 (target direction) + 1 (target distance)
        obs_dim = n * 6 + 8
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        
        # Action: collective pitch (1) + cyclic amplitude (1) + cyclic phase (1)
        # Could be per-node, but start simple
        self.action_space = spaces.Box(
            low=np.array([-1, -1, -1], dtype=np.float32),
            high=np.array([1, 1, 1], dtype=np.float32),
            dtype=np.float32
        )
        
        # For rendering
        self._renderer = None
        
    def _create_constellation(self):
        """Create and initialize the constellation."""
        self.dervish = create_ring_constellation(
            n_nodes=self.config.n_nodes,
            radius=self.config.constellation_radius
        )
        
        if self.config.launch_from_throw:
            self.launcher = BolaLauncher(self.dervish)
            
            # Simulate throw
            hand_pos = np.array([0, 0, 1.8])
            self.launcher.collapse_for_throw(hand_pos)
            
            # Spin up (more time for higher RPM)
            for _ in range(150):
                self.launcher.spin_up(0.02, effort=1.0)
            
            # Release at steeper angle for more altitude
            self.launcher.release(throw_angle=np.radians(60))
            
            # Let it deploy fully
            for _ in range(150):
                self.launcher.step(0.02, np.zeros(3))
            
            # Boost altitude if too low
            for node in self.dervish.nodes.values():
                if node.position[2] < 30:
                    node.position[2] += 30
                    node.velocity[2] = max(node.velocity[2], 5.0)
        else:
            # Start already deployed at altitude
            for node in self.dervish.nodes.values():
                node.position[2] = self.config.initial_altitude
                
            # Give initial spin
            spin_speed = 12.0
            centroid = self.dervish.compute_centroid()
            for node in self.dervish.nodes.values():
                r = node.position - centroid
                r[2] = 0
                r_mag = np.linalg.norm(r)
                if r_mag > 0.1:
                    tangent = np.array([-r[1], r[0], 0]) / r_mag
                    node.velocity = spin_speed * tangent
        
        # Set initial pitch for lift
        self.dervish.collective_pitch = np.radians(8)
    
    def _get_obs(self) -> np.ndarray:
        """Get observation vector."""
        obs = []
        
        # Per-node observations (normalized)
        centroid = self.dervish.compute_centroid()
        
        for node in self.dervish.nodes.values():
            # Position relative to centroid (normalized by constellation radius)
            rel_pos = (node.position - centroid) / self.config.constellation_radius
            obs.extend(rel_pos)
            
            # Velocity (normalized)
            vel_norm = node.velocity / 50.0  # Assume max ~50 m/s
            obs.extend(vel_norm)
        
        # Global observations
        # Centroid (normalized by typical scale)
        obs.extend(centroid / 100.0)
        
        # Spin phase (normalized to [-1, 1])
        obs.append(np.sin(self.dervish.spin_phase))
        
        # Target direction and distance
        to_target = self.target - centroid
        target_dist = np.linalg.norm(to_target)
        if target_dist > 0.1:
            target_dir = to_target / target_dist
        else:
            target_dir = np.zeros(3)
        
        obs.extend(target_dir)
        obs.append(np.clip(target_dist / 200.0, 0, 1))  # Normalized distance
        
        return np.array(obs, dtype=np.float32)
    
    def _compute_reward(self) -> Tuple[float, Dict]:
        """Compute reward for current state."""
        reward = 0.0
        info = {}
        
        centroid = self.dervish.centroid
        
        # 1. Altitude reward (stay airborne)
        altitude = centroid[2]
        if altitude > 5:
            alt_reward = self.config.altitude_weight * min(altitude / 50.0, 1.0)
        else:
            alt_reward = -5.0  # Crashed
        reward += alt_reward
        info['altitude_reward'] = alt_reward
        
        # 2. Target approach reward
        to_target = self.target - centroid
        target_dist = np.linalg.norm(to_target)
        target_reward = self.config.target_weight * (1.0 - np.clip(target_dist / 200.0, 0, 1))
        reward += target_reward
        info['target_reward'] = target_reward
        info['target_distance'] = target_dist
        
        # 3. Stability reward (nodes maintain formation)
        positions = [n.position for n in self.dervish.nodes.values()]
        if len(positions) > 1:
            mean_pos = np.mean(positions, axis=0)
            radii = [np.linalg.norm(p - mean_pos) for p in positions]
            radius_variance = np.var(radii)
            stability = 1.0 / (1.0 + radius_variance)
            stability_reward = self.config.stability_weight * stability
        else:
            stability_reward = 0
        reward += stability_reward
        info['stability_reward'] = stability_reward
        
        # 4. Energy efficiency (penalize excessive pitch)
        pitch_penalty = 0.01 * abs(self.dervish.collective_pitch)
        reward -= pitch_penalty
        info['energy_penalty'] = pitch_penalty
        
        return reward, info
    
    def reset(self, seed: int = None, options: Dict = None) -> Tuple[np.ndarray, Dict]:
        """Reset the environment."""
        super().reset(seed=seed)
        
        self.step_count = 0
        self.episode_reward = 0.0
        
        # Randomize target
        if self.config.target_position is not None:
            self.target = self.config.target_position.copy()
        else:
            self.target = np.array([
                self.np_random.uniform(50, 150),
                self.np_random.uniform(50, 150),
                self.np_random.uniform(30, 80)
            ])
        
        # Create new constellation
        self._create_constellation()
        
        obs = self._get_obs()
        info = {'target': self.target.copy()}
        
        return obs, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Take a step in the environment."""
        self.step_count += 1
        
        # Decode action
        collective = action[0] * np.radians(15)  # ±15° collective
        cyclic_amp = (action[1] + 1) / 2 * np.radians(10)  # 0-10° cyclic
        cyclic_phase = action[2] * np.pi  # ±180° phase
        
        # Apply to dervish
        self.dervish.collective_pitch = np.radians(5) + collective  # 5° base + control
        self.dervish.cyclic_amplitude = cyclic_amp
        self.dervish.cyclic_phase = cyclic_phase
        
        # Step physics (multiple substeps for stability)
        substeps = 4
        sub_dt = self.config.dt / substeps
        freestream = np.zeros(3)  # Could add wind
        
        for _ in range(substeps):
            self.dervish.step(sub_dt, freestream)
        
        # Get observation and reward
        obs = self._get_obs()
        reward, info = self._compute_reward()
        self.episode_reward += reward
        
        # Check termination
        centroid = self.dervish.centroid
        crashed = centroid[2] < 1.0
        reached_target = info['target_distance'] < 10.0
        timeout = self.step_count >= self.config.max_episode_steps
        
        terminated = crashed or reached_target
        truncated = timeout
        
        info['step'] = self.step_count
        info['episode_reward'] = self.episode_reward
        info['centroid'] = centroid.copy()
        
        if reached_target:
            reward += 100.0  # Big bonus
            info['success'] = True
        elif crashed:
            reward -= 50.0
            info['crashed'] = True
        
        return obs, reward, terminated, truncated, info
    
    def render(self):
        """Render the environment."""
        if self.render_mode == "human":
            # Could integrate with ModernGL renderer
            pass
        elif self.render_mode == "rgb_array":
            # Return image array
            pass
        return None
    
    def close(self):
        """Clean up."""
        if self._renderer:
            self._renderer = None


def test_env():
    """Test the environment."""
    print("=" * 60)
    print("HUBLESS DERVISH - TRAINING ENVIRONMENT TEST")
    print("=" * 60)
    
    env = HublessDervishEnv(DervishEnvConfig(
        n_nodes=4,
        constellation_radius=10.0,
        launch_from_throw=True,
        max_episode_steps=500
    ))
    
    obs, info = env.reset(seed=42)
    print(f"\nInitial observation shape: {obs.shape}")
    print(f"Action space: {env.action_space}")
    print(f"Target: {info['target']}")
    
    # Run episode with random actions
    total_reward = 0
    for step in range(200):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if step % 50 == 0:
            print(f"Step {step:3d} | Reward: {reward:6.2f} | "
                  f"Alt: {info['centroid'][2]:5.1f}m | "
                  f"Target dist: {info['target_distance']:6.1f}m")
        
        if terminated or truncated:
            print(f"\nEpisode ended: {'crashed' if info.get('crashed') else 'success' if info.get('success') else 'timeout'}")
            break
    
    print(f"\nTotal reward: {total_reward:.1f}")
    env.close()


if __name__ == "__main__":
    test_env()
