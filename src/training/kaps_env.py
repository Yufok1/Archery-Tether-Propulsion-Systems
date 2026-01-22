"""
KAPS Gym Environment
====================
Wraps KAPSSimulation as a Gymnasium environment for RL training.

Observation space: TAB positions, velocities, cable tensions, threats
Action space: Control surface deflections, release commands
Rewards: Intercepts, survival, formation quality
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, Tuple, Optional, Any

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ..main import KAPSSimulation
from ..ai.defensive_matrix import ThreatType


class KAPSEnv(gym.Env):
    """
    Gymnasium environment for training RL agents on KAPS.
    
    The agent controls:
    - Formation spread/mode
    - Individual TAB control surface commands
    - Release decisions (which TAB to slingshot)
    
    The agent observes:
    - Mother drone state (position, velocity, orientation)
    - All TAB states (positions, velocities, attached status)
    - Cable tensions
    - Threat positions and velocities (up to N threats)
    - Defensive bubble status
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    
    def __init__(self, 
                 render_mode: Optional[str] = None,
                 max_threats: int = 8,
                 episode_steps: int = 3000,  # 3 seconds at 1000Hz
                 threat_spawn_rate: float = 0.01):
        super().__init__()
        
        self.render_mode = render_mode
        self.max_threats = max_threats
        self.episode_steps = episode_steps
        self.threat_spawn_rate = threat_spawn_rate
        
        # Will be created on reset
        self.sim: Optional[KAPSSimulation] = None
        self.step_count = 0
        self.total_reward = 0
        
        # Statistics
        self.stats = {
            'threats_spawned': 0,
            'threats_intercepted': 0,
            'tabs_released': 0,
            'damage_taken': 0
        }
        
        # ===== OBSERVATION SPACE =====
        # Mother drone: pos(3) + vel(3) + orientation(3) = 9
        # 4 TABs: pos(3) + vel(3) + attached(1) + tension(1) = 8 each = 32
        # Threats: pos(3) + vel(3) + active(1) = 7 each × max_threats
        # Total: 9 + 32 + 7*max_threats
        obs_dim = 9 + 32 + 7 * max_threats
        
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )
        
        # ===== ACTION SPACE =====
        # Continuous actions:
        # - Formation spread adjustment: 1 (delta spread)
        # - Per-TAB elevator commands: 4
        # - Per-TAB rudder commands: 4
        # - Release decision: 4 (probability per TAB, threshold = 0.8)
        # Total: 13 continuous actions
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(13,),
            dtype=np.float32
        )
        
        # Reward shaping weights
        self.reward_weights = {
            'intercept': 100.0,       # Successful threat intercept
            'damage': -200.0,         # Threat hit mother drone
            'cable_snap': -50.0,      # Cable broke from tension
            'tab_lost': -25.0,        # TAB released (cost)
            'formation_bonus': 0.1,   # Per-step formation quality
            'survival': 1.0,          # Per-step survival bonus
            'threat_proximity': -0.01 # Per-step penalty for close threats
        }
        
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment to initial state"""
        super().reset(seed=seed)
        
        # Create fresh simulation
        self.sim = KAPSSimulation()
        self.step_count = 0
        self.total_reward = 0
        self.stats = {
            'threats_spawned': 0,
            'threats_intercepted': 0,
            'tabs_released': 0,
            'damage_taken': 0
        }
        
        # Get initial observation
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one environment step"""
        assert self.sim is not None, "Must call reset() before step()"
        
        # Parse action
        self._apply_action(action)
        
        # Maybe spawn threat
        if self.np_random.random() < self.threat_spawn_rate:
            self._spawn_random_threat()
        
        # Step simulation
        telemetry = self.sim.step()
        self.step_count += 1
        
        # Compute reward
        reward = self._compute_reward(telemetry)
        self.total_reward += reward
        
        # Check termination
        terminated = self._check_terminated(telemetry)
        truncated = self.step_count >= self.episode_steps
        
        # Get observation
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, reward, terminated, truncated, info
    
    def _get_observation(self) -> np.ndarray:
        """Build observation vector"""
        obs = []
        
        # Mother drone state (normalized)
        md = self.sim.mother_drone
        obs.extend(md.position / 1000.0)  # Scale positions
        obs.extend(md.velocity / 100.0)   # Scale velocities
        obs.extend(md.orientation / np.pi) # Normalize angles
        
        # TAB states
        for tab_id in ["UP", "DOWN", "LEFT", "RIGHT"]:
            tab = self.sim.tab_array.tabs[tab_id]
            obs.extend(tab.position / 1000.0)
            obs.extend(tab.velocity / 100.0)
            obs.append(1.0 if tab.is_attached else 0.0)
            
            # Cable tension (normalized)
            tension = self.sim.tether_array.get_tension(tab_id) if hasattr(self.sim.tether_array, 'get_tension') else 0
            obs.append(tension / 10000.0)
        
        # Threats (pad to max_threats)
        threats = self.sim.defensive_ai.get_active_threats() if hasattr(self.sim.defensive_ai, 'get_active_threats') else []
        for i in range(self.max_threats):
            if i < len(threats):
                t = threats[i]
                obs.extend(t.get('position', np.zeros(3)) / 1000.0)
                obs.extend(t.get('velocity', np.zeros(3)) / 100.0)
                obs.append(1.0)  # Active
            else:
                obs.extend([0.0] * 7)  # Inactive slot
        
        return np.array(obs, dtype=np.float32)
    
    def _apply_action(self, action: np.ndarray):
        """Apply agent's action to simulation"""
        # Action layout:
        # [0]: formation spread delta
        # [1:5]: elevator commands for UP, DOWN, LEFT, RIGHT
        # [5:9]: rudder commands for UP, DOWN, LEFT, RIGHT
        # [9:13]: release probabilities for UP, DOWN, LEFT, RIGHT
        
        # Formation spread (not directly controllable, skip for now)
        
        # Control surfaces
        tab_ids = ["UP", "DOWN", "LEFT", "RIGHT"]
        for i, tab_id in enumerate(tab_ids):
            if tab_id in self.sim.tab_array.tabs:
                tab = self.sim.tab_array.tabs[tab_id]
                if tab.is_attached:
                    # Scale action to control limits
                    elevator = float(action[1 + i]) * tab.config.elevator_max
                    rudder = float(action[5 + i]) * tab.config.rudder_max
                    tab.set_control_targets(elevator=elevator, rudder=rudder)
        
        # Release decisions
        for i, tab_id in enumerate(tab_ids):
            if action[9 + i] > 0.8:  # Threshold for release
                if tab_id in self.sim.tab_array.tabs:
                    tab = self.sim.tab_array.tabs[tab_id]
                    if tab.is_attached:
                        # Calculate best release velocity
                        threat_dir = self._get_closest_threat_direction()
                        if threat_dir is not None:
                            tab.execute_release(tab.velocity + threat_dir * 50)
                            self.stats['tabs_released'] += 1
    
    def _get_closest_threat_direction(self) -> Optional[np.ndarray]:
        """Get direction to closest threat"""
        threats = self.sim.defensive_ai.get_active_threats() if hasattr(self.sim.defensive_ai, 'get_active_threats') else []
        if not threats:
            return None
        
        md_pos = self.sim.mother_drone.position
        closest = None
        min_dist = float('inf')
        
        for t in threats:
            t_pos = t.get('position', np.zeros(3))
            dist = np.linalg.norm(t_pos - md_pos)
            if dist < min_dist:
                min_dist = dist
                closest = t_pos
        
        if closest is None:
            return None
        
        direction = closest - md_pos
        return direction / (np.linalg.norm(direction) + 1e-8)
    
    def _spawn_random_threat(self):
        """Spawn a random threat"""
        md_pos = self.sim.mother_drone.position
        
        # Random direction, 300-500m away
        theta = self.np_random.uniform(0, 2 * np.pi)
        phi = self.np_random.uniform(-np.pi/4, np.pi/4)
        dist = self.np_random.uniform(300, 500)
        
        offset = np.array([
            dist * np.cos(phi) * np.cos(theta),
            dist * np.cos(phi) * np.sin(theta),
            dist * np.sin(phi)
        ])
        
        threat_pos = md_pos + offset
        threat_vel = -offset / np.linalg.norm(offset) * self.np_random.uniform(100, 200)
        
        self.sim.inject_threat(
            position=threat_pos,
            velocity=threat_vel,
            threat_type=ThreatType.MISSILE_IR
        )
        self.stats['threats_spawned'] += 1
    
    def _compute_reward(self, telemetry: Dict) -> float:
        """Compute reward for this step"""
        reward = 0.0
        
        # Survival bonus
        reward += self.reward_weights['survival']
        
        # Formation quality bonus
        formation = telemetry.get('formation', {})
        quality = formation.get('quality', 0.5)
        reward += self.reward_weights['formation_bonus'] * quality
        
        # Check for intercepts (would need event tracking in sim)
        # For now, just track defense status
        defense = telemetry.get('defense', {})
        alert_level = defense.get('alert_level', 'GREEN')
        
        if alert_level == 'RED':
            reward += self.reward_weights['threat_proximity']
        
        # Damage check (would need impact detection in sim)
        # Placeholder for now
        
        return reward
    
    def _check_terminated(self, telemetry: Dict) -> bool:
        """Check if episode should terminate"""
        # Terminate if mother drone takes critical damage
        # For now, never terminate early (just truncate)
        return False
    
    def _get_info(self) -> Dict:
        """Get auxiliary information"""
        return {
            'step': self.step_count,
            'total_reward': self.total_reward,
            'tabs_attached': self.sim.tab_array.count_attached() if self.sim else 0,
            **self.stats
        }
    
    def render(self):
        """Render the environment"""
        if self.render_mode == "human":
            # Would launch Panda3D visualizer
            pass
        elif self.render_mode == "rgb_array":
            # Would return frame buffer
            pass
    
    def close(self):
        """Clean up"""
        self.sim = None


# Register the environment
gym.register(
    id='KAPS-v0',
    entry_point='src.training.kaps_env:KAPSEnv',
    max_episode_steps=3000,
)
