"""
Exploration-Focused KAPS Environment
=====================================
An environment designed to let DreamerV3 EXPLORE the physics possibilities.

Key Design Principles:
1. INTRINSIC CURIOSITY - Reward novel physics states, not just task completion
2. FREEDOM - Don't over-constrain what the agent can do
3. RICH DYNAMICS - Many interacting systems to discover
4. HOSTILE PRESSURE - Threats force creative solutions

The goal is for the dreamer to DISCOVER:
- Cable whip maneuvers
- Aero boost techniques
- Optimal release timing
- Sacrifice plays
- Multi-threat prioritization
- Formation exploitation

This is about EMERGENCE, not scripted behavior.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, Tuple, Optional, Any, List
from collections import deque

import sys
import os
# Add parent directories to path for imports
_parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_root = os.path.dirname(_parent)
if _parent not in sys.path:
    sys.path.insert(0, _parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

try:
    from src.main import KAPSSimulation
    from src.training.threat_environment import ThreatSpawner, ThreatType, Threat
    from src.physics.cable_geometry import (
        CableIntersectionDetector, 
        SectorConstrainedActionSpace,
        TangleState
    )
except ImportError:
    from main import KAPSSimulation
    from training.threat_environment import ThreatSpawner, ThreatType, Threat
    try:
        from physics.cable_geometry import (
            CableIntersectionDetector,
            SectorConstrainedActionSpace, 
            TangleState
        )
    except ImportError:
        # Fallback - no cable geometry enforcement
        CableIntersectionDetector = None
        SectorConstrainedActionSpace = None
        TangleState = None


class ExplorationKAPSEnv(gym.Env):
    """
    Exploration-focused KAPS environment with intrinsic curiosity.
    
    This environment is designed to let DreamerV3 discover the physics
    possibilities through exploration, not just optimize for a fixed reward.
    
    Intrinsic rewards:
    - State novelty (visiting new physics configurations)
    - Prediction error (surprising dynamics)
    - Empowerment (control over future states)
    
    Extrinsic rewards:
    - Threat intercepts
    - Survival
    - Damage avoidance
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    
    def __init__(self,
                 render_mode: Optional[str] = None,
                 episode_steps: int = 3000,
                 threat_spawn_interval: int = 100,  # Steps between waves
                 curiosity_weight: float = 0.5,     # Balance intrinsic/extrinsic
                 state_memory_size: int = 1000):    # For novelty detection
        super().__init__()
        
        self.render_mode = render_mode
        self.episode_steps = episode_steps
        self.threat_spawn_interval = threat_spawn_interval
        self.curiosity_weight = curiosity_weight
        
        # Simulation
        self.sim: Optional[KAPSSimulation] = None
        self.threat_spawner: Optional[ThreatSpawner] = None
        self.step_count = 0
        
        # Cable geometry constraints
        self.cable_detector: Optional[CableIntersectionDetector] = None
        self.action_constrainer: Optional[SectorConstrainedActionSpace] = None
        if CableIntersectionDetector is not None:
            self.cable_detector = CableIntersectionDetector(cable_length=30.0)
            self.action_constrainer = SectorConstrainedActionSpace(cable_length=30.0)
        
        # State memory for novelty detection
        self.state_memory = deque(maxlen=state_memory_size)
        self.state_visit_counts: Dict[tuple, int] = {}
        
        # Physics state tracking (for discovering interesting configurations)
        self.max_cable_tension_seen = 0
        self.max_speed_seen = 0
        self.max_altitude_change = 0
        self.releases_performed = 0
        self.intercepts_achieved = 0
        
        # Episode statistics
        self.episode_stats = {}
        
        # ===== OBSERVATION SPACE =====
        # Extended observation for richer learning:
        # Mother: pos(3) + vel(3) + orientation(3) + angular_vel(3) = 12
        # 4 TABs: pos(3) + vel(3) + attached(1) + tension(1) + ctrl_surfaces(3) + sector_pos(2) = 13 each = 52
        # Cable pairs (6 pairs): distance(1) + tangle_state(1) = 2 each = 12
        # Up to 8 threats: pos(3) + vel(3) + type_onehot(6) + distance(1) + closing_rate(1) = 14 each = 112
        # Physics state: max_tension(1) + formation_spread(1) + cable_drag(1) + time_since_release(1) = 4
        # Total: 12 + 52 + 12 + 112 + 4 = 192
        obs_dim = 192
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(obs_dim,), dtype=np.float32
        )
        
        # ===== ACTION SPACE =====
        # TABs are AIRFOILS - they fly and maneuver to protect the Buzzard
        # - Formation mode (continuous blend between tight/wide)
        # - Per-TAB elevator, aileron, rudder commands (4 * 3 = 12)
        # - Per-TAB release command (4) - release for intercept
        # - Emergency speed burst (1)
        # Total: 1 + 12 + 4 + 1 = 18
        self.action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(18,), dtype=np.float32
        )
        
        # ===== REWARD WEIGHTS =====
        # THE ONLY GOAL: PROTECT THE BUZZARD
        # Everything else is secondary. TABs exist to die for the Buzzard.
        self.reward_config = {
            # PRIMARY: Buzzard survival (the ONLY thing that matters)
            'buzzard_alive': 1.0,              # Per-step reward for survival
            'buzzard_damaged': -500.0,         # CATASTROPHIC - we failed
            'buzzard_destroyed': -10000.0,     # GAME OVER - total failure
            
            # SECONDARY: Successful defense (means Buzzard is safer)
            'threat_intercepted': 200.0,       # TAB killed a threat - great!
            'threat_expired': 10.0,            # Threat gave up/missed - good
            
            # TERTIARY: TAB sacrifice (acceptable cost for Buzzard safety)
            'tab_sacrificed': 50.0,            # TAB died killing threat - HEROIC
            'tab_lost_no_kill': -20.0,         # TAB died without helping - wasteful
            
            # Exploration (helps find better defense strategies)
            'state_novelty': 5.0,              # Exploring new configurations
            'physics_discovery': 25.0,         # Finding new capabilities
        }
        
        # Track Buzzard health
        self.buzzard_health = 100.0
        
        # Action history for diversity
        self.action_history = deque(maxlen=100)
        
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        
        # Fresh simulation
        self.sim = KAPSSimulation()
        self.threat_spawner = ThreatSpawner()
        self.step_count = 0
        
        # BUZZARD HEALTH - the only thing that truly matters
        self.buzzard_health = 100.0
        
        # Reset episode stats
        self.episode_stats = {
            'total_reward': 0,
            'intrinsic_reward': 0,
            'extrinsic_reward': 0,
            'buzzard_health': 100.0,
            'intercepts': 0,               # Threats killed by TABs
            'impacts': 0,                  # Threats that hit Buzzard
            'tabs_sacrificed': 0,          # TABs that died killing threats
            'tabs_lost': 0,                # TABs lost without a kill
            'tabs_released': 0,
            'max_tension': 0,
            'max_speed': 0,
            'novel_states': 0,
            'physics_discoveries': 0,
        }
        
        # Initial threat wave (give agent something to react to)
        self._spawn_threat_wave()
        
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        assert self.sim is not None
        
        # Record action for diversity
        self.action_history.append(action.copy())
        
        # Apply action
        self._apply_action(action)
        
        # Step simulation
        telemetry = self.sim.step()
        self.step_count += 1
        
        # Update threats
        md_pos = telemetry['mother_drone']['position']
        md_vel = telemetry['mother_drone'].get('velocity', np.zeros(3))
        self.threat_spawner.update_all(self.sim.dt, md_pos, md_vel)
        
        # Maybe spawn new threats
        if self.step_count % self.threat_spawn_interval == 0:
            self._spawn_threat_wave()
        
        # Check intercepts
        tab_positions = {
            tid: np.array(telemetry['tabs'][tid]['position'])
            for tid in telemetry['tabs']
        }
        intercepts = self.threat_spawner.check_intercepts(tab_positions)
        
        # Check impacts (threat hit mother drone)
        impacts = self.threat_spawner.check_impacts(md_pos)
        
        # Compute reward (both intrinsic and extrinsic)
        extrinsic_reward = self._compute_extrinsic_reward(telemetry, intercepts, impacts)
        intrinsic_reward = self._compute_intrinsic_reward(telemetry, action)
        
        total_reward = (
            (1 - self.curiosity_weight) * extrinsic_reward +
            self.curiosity_weight * intrinsic_reward
        )
        
        self.episode_stats['total_reward'] += total_reward
        self.episode_stats['extrinsic_reward'] += extrinsic_reward
        self.episode_stats['intrinsic_reward'] += intrinsic_reward
        self.episode_stats['intercepts'] += len(intercepts)
        self.episode_stats['impacts'] += len(impacts)
        
        # Termination
        terminated = self._check_terminated(telemetry, impacts)
        truncated = self.step_count >= self.episode_steps
        
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, total_reward, terminated, truncated, info
    
    def _get_observation(self) -> np.ndarray:
        """Build extended observation vector"""
        obs = []
        
        # Mother drone (12 values)
        md = self.sim.mother_drone
        md_pos = md.position
        md_vel = md.velocity
        
        obs.extend(md_pos / 1000.0)
        obs.extend(md_vel / 100.0)
        obs.extend(md.orientation / np.pi)
        obs.extend(md.angular_velocity / 10.0)
        
        # TABs (52 values - 13 per TAB)
        for tab_id in ["UP", "DOWN", "LEFT", "RIGHT"]:
            tab = self.sim.tab_array.tabs[tab_id]
            obs.extend(tab.position / 1000.0)
            obs.extend(tab.velocity / 100.0)
            obs.append(1.0 if tab.is_attached else 0.0)
            obs.append(self._get_cable_tension(tab_id) / 10000.0)
            obs.append(tab.elevator / tab.config.elevator_max)
            obs.append(tab.aileron / tab.config.aileron_max)
            obs.append(tab.rudder / tab.config.rudder_max)
            
            # Add sector position (normalized position within allowed wedge)
            rel_pos = tab.position - md_pos
            sector_y = rel_pos[1] / 30.0  # Normalize by cable length
            sector_z = rel_pos[2] / 30.0
            obs.append(sector_y)
            obs.append(sector_z)
        
        # Cable pair states (12 values - 2 per pair, 6 pairs)
        cable_pairs = [("UP", "DOWN"), ("UP", "LEFT"), ("UP", "RIGHT"), 
                       ("DOWN", "LEFT"), ("DOWN", "RIGHT"), ("LEFT", "RIGHT")]
        for id1, id2 in cable_pairs:
            if self.cable_detector is not None:
                # Get minimum distance between cables
                if id1 in self.cable_detector.cables and id2 in self.cable_detector.cables:
                    dist = self.cable_detector._compute_cable_distance(id1, id2)
                    obs.append(dist / 30.0)  # Normalized by cable length
                    
                    # Tangle state as float
                    tangle = self.cable_detector.get_tangle_state(id1, id2)
                    tangle_val = {
                        TangleState.CLEAR: 0.0,
                        TangleState.PROXIMITY: 0.25,
                        TangleState.CROSSED: 0.5,
                        TangleState.TANGLED: 0.75,
                        TangleState.LOCKED: 1.0,
                    }.get(tangle, 0.0)
                    obs.append(tangle_val)
                else:
                    obs.extend([1.0, 0.0])  # Far apart, clear
            else:
                obs.extend([1.0, 0.0])  # No detector, assume clear
        
        # Threats (112 values for 8 threats)
        threats = self.threat_spawner.get_active_threats() if self.threat_spawner else []
        
        for i in range(8):
            if i < len(threats):
                t = threats[i]
                obs.extend(t.position / 1000.0)
                obs.extend(t.velocity / 100.0)
                # Type one-hot (6 types)
                type_onehot = [0.0] * 6
                type_idx = list(ThreatType).index(t.profile.type)
                type_onehot[type_idx] = 1.0
                obs.extend(type_onehot)
                # Distance and closing rate
                dist = np.linalg.norm(t.position - md_pos)
                closing = -np.dot(t.velocity - md_vel, (t.position - md_pos) / (dist + 1e-8))
                obs.append(dist / 500.0)
                obs.append(closing / 100.0)
            else:
                obs.extend([0.0] * 14)
        
        # Physics state (4 values)
        max_tension = max(self._get_cable_tension(tid) for tid in ["UP", "DOWN", "LEFT", "RIGHT"])
        obs.append(max_tension / 10000.0)
        obs.append(self._get_formation_spread() / 50.0)
        
        # Cable drag penalty (from tangling)
        cable_drag = self.cable_detector.compute_drag_penalty() if self.cable_detector else 0.0
        obs.append(cable_drag)
        
        obs.append(min(self.step_count / 100.0, 1.0))  # Time since last release (simplified)
        
        return np.array(obs, dtype=np.float32)
    
    def _apply_action(self, action: np.ndarray):
        """Apply agent action with cable geometry constraints enforced"""
        # Action layout:
        # [0]: Formation mode (tight=-1 to wide=1)
        # [1:13]: Control surfaces (elevator, aileron, rudder) for 4 TABs
        # [13:17]: Release commands for 4 TABs
        # [17]: Emergency speed burst
        
        tab_ids = ["UP", "DOWN", "LEFT", "RIGHT"]
        md_pos = self.sim.mother_drone.position
        
        # First, update cable geometry
        if self.cable_detector is not None:
            for tab_id in tab_ids:
                tab = self.sim.tab_array.tabs[tab_id]
                if tab.is_attached:
                    tension = self._get_cable_tension(tab_id)
                    self.cable_detector.update_cable(tab_id, md_pos, tab.position, tension)
        
        # Check for cable intersections
        cable_penalty = 0.0
        forced_releases = []
        if self.cable_detector is not None:
            distances = self.cable_detector.check_all_intersections()
            
            # Check for tangles
            tangled_pairs = self.cable_detector.get_tangled_pairs()
            crossed_pairs = self.cable_detector.get_crossed_pairs()
            
            # Apply drag penalty for tangled cables
            cable_penalty = self.cable_detector.compute_drag_penalty()
            if cable_penalty > 0:
                self.episode_stats['cable_drag_penalty'] = self.episode_stats.get('cable_drag_penalty', 0) + cable_penalty
            
            # Get forced releases (locked cables must release one)
            forced_releases = self.cable_detector.get_forced_releases()
        
        # Control surfaces - constrained to valid sectors
        for i, tab_id in enumerate(tab_ids):
            tab = self.sim.tab_array.tabs[tab_id]
            if tab.is_attached:
                # Raw action values
                raw_elevator = float(action[1 + i*3])
                raw_aileron = float(action[2 + i*3])
                raw_rudder = float(action[3 + i*3])
                
                # Convert to YZ motion and constrain to sector
                if self.action_constrainer is not None:
                    # Get current YZ position relative to mother drone
                    rel_pos = tab.position - md_pos
                    current_yz = np.array([rel_pos[1], rel_pos[2]])
                    
                    # Raw action implies a target motion
                    # rudder -> Y motion, elevator -> Z motion
                    raw_motion = np.array([
                        raw_rudder * 5.0,   # Scale to meters
                        raw_elevator * 5.0
                    ])
                    
                    # Constrain to sector
                    constrained_motion = self.action_constrainer.constrain_action(
                        tab_id, raw_motion, current_yz
                    )
                    
                    # Convert back to control surface commands
                    # This limits the agent to ONLY valid maneuvers
                    scale = np.linalg.norm(raw_motion) if np.linalg.norm(raw_motion) > 0.1 else 1.0
                    constrained_scale = np.linalg.norm(constrained_motion)
                    damping = min(1.0, constrained_scale / scale) if scale > 0.1 else 1.0
                    
                    elevator = raw_elevator * damping * tab.config.elevator_max
                    rudder = raw_rudder * damping * tab.config.rudder_max
                    aileron = raw_aileron * tab.config.aileron_max  # Aileron less constrained
                else:
                    elevator = raw_elevator * tab.config.elevator_max
                    aileron = raw_aileron * tab.config.aileron_max
                    rudder = raw_rudder * tab.config.rudder_max
                
                tab.set_control_targets(elevator=elevator, rudder=rudder, aileron=aileron)
        
        # Release commands (threshold 0.7 to trigger)
        # Also handle forced releases from cable tangles
        for i, tab_id in enumerate(tab_ids):
            should_release = action[13 + i] > 0.7 or tab_id in forced_releases
            if should_release:
                tab = self.sim.tab_array.tabs[tab_id]
                if tab.is_attached:
                    # Calculate release velocity based on current physics state
                    release_vel = self._calculate_release_velocity(tab_id)
                    tab.execute_release(release_vel)
                    self.releases_performed += 1
                    self.episode_stats['tabs_released'] += 1
                    
                    if tab_id in forced_releases:
                        self.episode_stats['forced_releases'] = self.episode_stats.get('forced_releases', 0) + 1
                        print(f"[CABLE TANGLE] Forced release of {tab_id}!")
        
        # Speed burst
        if action[17] > 0.9:
            self.sim.execute_speed_burst()
        
        # Store cable penalty for reward calculation
        self._current_cable_penalty = cable_penalty
    
    def _calculate_release_velocity(self, tab_id: str) -> np.ndarray:
        """Calculate optimal release velocity for intercept"""
        tab = self.sim.tab_array.tabs[tab_id]
        threats = self.threat_spawner.get_active_threats() if self.threat_spawner else []
        
        if not threats:
            # No threats - release in current direction
            return tab.velocity + tab.velocity / (np.linalg.norm(tab.velocity) + 1e-8) * 30
        
        # Aim at closest threat
        closest = min(threats, key=lambda t: np.linalg.norm(t.position - tab.position))
        direction = closest.position - tab.position
        direction = direction / (np.linalg.norm(direction) + 1e-8)
        
        # Add cable-stored momentum
        tension = self._get_cable_tension(tab_id)
        boost = tension / 1000.0  # Higher tension = more stored energy
        
        return tab.velocity + direction * (30 + boost)
    
    def _compute_extrinsic_reward(self, telemetry: Dict, 
                                   intercepts: List, impacts: List) -> float:
        """
        Compute task-based reward.
        
        THE ONLY GOAL: PROTECT THE BUZZARD
        
        Priority:
        1. Buzzard survival (paramount)
        2. Threat elimination (enables survival)
        3. TAB sacrifice (acceptable cost)
        """
        reward = 0.0
        
        # ===== PRIMARY: BUZZARD SURVIVAL =====
        # Every step the Buzzard survives is a win
        reward += self.reward_config['buzzard_alive']
        
        # ===== CATASTROPHIC: BUZZARD DAMAGED =====
        for impact in impacts:
            damage = impact.profile.damage
            self.buzzard_health -= damage
            self.episode_stats['buzzard_health'] = self.buzzard_health
            self.episode_stats['impacts'] += 1
            
            # Massive penalty - we failed to protect
            reward += self.reward_config['buzzard_damaged'] * (damage / 100.0)
            print(f"[BUZZARD HIT] Damage: {damage}, Health: {self.buzzard_health:.0f}%")
        
        # ===== SECONDARY: SUCCESSFUL INTERCEPTS =====
        for intercept in intercepts:
            reward += self.reward_config['threat_intercepted']
            self.episode_stats['intercepts'] += 1
            
            # Check if TAB was sacrificed in the intercept
            if hasattr(intercept, 'tab_destroyed') and intercept.tab_destroyed:
                # TAB died killing the threat - HEROIC
                reward += self.reward_config['tab_sacrificed']
                self.episode_stats['tabs_sacrificed'] += 1
                print(f"[HEROIC] TAB sacrificed to destroy threat!")
        
        # Cable tangle penalty (tangled cables = reduced defense capability)
        cable_penalty = getattr(self, '_current_cable_penalty', 0.0)
        if cable_penalty > 0:
            # Tangled cables hurt our ability to defend
            reward -= cable_penalty * 20.0
        
        return reward
    
    def _compute_intrinsic_reward(self, telemetry: Dict, action: np.ndarray) -> float:
        """
        Compute curiosity/exploration reward.
        
        Exploration helps find better DEFENSE strategies.
        It's not the goal - defending the Buzzard is.
        But exploring helps us find creative ways to defend.
        """
        reward = 0.0
        
        # === STATE NOVELTY ===
        # Visiting new configurations might reveal better defense positions
        state_hash = self._hash_physics_state(telemetry)
        visit_count = self.state_visit_counts.get(state_hash, 0)
        novelty_bonus = 1.0 / (1.0 + np.sqrt(visit_count))
        reward += self.reward_config['state_novelty'] * novelty_bonus
        
        if visit_count == 0:
            self.episode_stats['novel_states'] += 1
        self.state_visit_counts[state_hash] = visit_count + 1
        
        # === PHYSICS DISCOVERY ===
        # Finding new physical capabilities (tension, speed) might enable new defense tactics
        max_tension = max(self._get_cable_tension(tid) for tid in ["UP", "DOWN", "LEFT", "RIGHT"])
        speed = telemetry['mother_drone']['speed']
        
        if max_tension > self.max_cable_tension_seen * 1.1:
            reward += self.reward_config['physics_discovery']
            self.max_cable_tension_seen = max_tension
            self.episode_stats['physics_discoveries'] += 1
        
        if speed > self.max_speed_seen * 1.1:
            reward += self.reward_config['physics_discovery']
            self.max_speed_seen = speed
            self.episode_stats['physics_discoveries'] += 1
        
        return reward
    
    def _hash_physics_state(self, telemetry: Dict) -> tuple:
        """Create discrete hash of physics state for novelty detection"""
        # Discretize key physics values
        md = telemetry['mother_drone']
        
        # Position bucket (100m resolution)
        pos_bucket = tuple((np.array(md['position']) / 100).astype(int))
        
        # Speed bucket (10 m/s resolution)
        speed_bucket = int(md['speed'] / 10)
        
        # Tension buckets (1000N resolution)
        tension_buckets = tuple(
            int(self._get_cable_tension(tid) / 1000)
            for tid in ["UP", "DOWN", "LEFT", "RIGHT"]
        )
        
        # Attached status
        attached = tuple(
            self.sim.tab_array.tabs[tid].is_attached
            for tid in ["UP", "DOWN", "LEFT", "RIGHT"]
        )
        
        return pos_bucket + (speed_bucket,) + tension_buckets + attached
    
    def _get_cable_tension(self, tab_id: str) -> float:
        """Get cable tension for a TAB"""
        if hasattr(self.sim.tether_array, 'get_tension'):
            return self.sim.tether_array.get_tension(tab_id)
        return 0.0
    
    def _get_formation_spread(self) -> float:
        """Get current formation spread"""
        if hasattr(self.sim.formation_ctrl, 'get_spread'):
            return self.sim.formation_ctrl.get_spread()
        return 30.0  # Default
    
    def _spawn_threat_wave(self):
        """Spawn a wave of threats"""
        if self.threat_spawner is None:
            return
        
        md_pos = self.sim.mother_drone.position
        md_vel = self.sim.mother_drone.velocity
        
        new_threats = self.threat_spawner.spawn_scenario(md_pos, md_vel)
        print(f"[WAVE {self.threat_spawner.wave_counter}] Spawned {len(new_threats)} threats")
    
    def _check_terminated(self, telemetry: Dict, impacts: List) -> bool:
        """
        Check for episode termination.
        
        The episode ends when the BUZZARD IS DESTROYED.
        That's it. That's the only failure condition.
        """
        # BUZZARD DESTROYED - TOTAL MISSION FAILURE
        if self.buzzard_health <= 0:
            print(f"[MISSION FAILED] Buzzard destroyed!")
            return True
        
        # Buzzard critically damaged - barely holding on
        if self.buzzard_health <= 20:
            print(f"[CRITICAL] Buzzard at {self.buzzard_health:.0f}% health!")
        
        return False
    
    def _get_info(self) -> Dict:
        """Get auxiliary information"""
        return {
            'step': self.step_count,
            'buzzard_health': self.buzzard_health,
            **self.episode_stats,
            'threats_active': len(self.threat_spawner.get_active_threats()) if self.threat_spawner else 0,
        }


# Register
gym.register(
    id='KAPS-Explore-v0',
    entry_point='src.training.exploration_env:ExplorationKAPSEnv',
    max_episode_steps=3000,
)
