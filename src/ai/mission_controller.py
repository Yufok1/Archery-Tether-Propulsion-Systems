"""
KAPS Unified Mission Controller
================================

The singular purpose: PROTECT THE BUZZARD.

This controller unifies all subsystems under one coherent objective.
DreamerV3's world model learns this objective implicitly through
reward shaping - the goal exists in latent space, not as explicit
rules that could be observed or exploited by adversaries.

SUBSYSTEMS COMMANDED:
--------------------
1. TAB Array - Individual airfoil control (dispersed mode)
2. Slingshot Controller - Bola/nunchaku dynamics (consolidated mode)
3. Airfoiled Buzzard - Deployable wings and grid fins
4. Threat Response - Intercept coordination
5. Formation Control - Defensive geometry

THE KEY INSIGHT:
---------------
The objective (protect Buzzard) emerges from reward shaping, not 
explicit programming. DreamerV3's RSSM encodes the goal in its
latent state. An adversary observing behavior cannot easily
extract the decision logic - it's distributed across the
world model's learned representations.

"The omission of its physical form could stand to retain control"
- The AI's intentions are implicit in behavior, not explicit rules.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum, auto


class MissionMode(Enum):
    """Overall mission operational modes."""
    PATROL = auto()           # No threats, maintain formation
    ALERT = auto()            # Threats detected, preparing defense
    ENGAGE = auto()           # Active threat engagement
    EVASIVE = auto()          # Buzzard taking damage, emergency evasion
    SLINGSHOT = auto()        # Consolidated bola mode
    RECOVERY = auto()         # Post-engagement, reforming


class ThreatPriority(Enum):
    """Threat engagement priority levels."""
    CRITICAL = auto()         # Will hit Buzzard in <2s
    HIGH = auto()             # Closing fast, <5s to impact
    MEDIUM = auto()           # Approaching, <10s
    LOW = auto()              # Distant, tracking
    NONE = auto()             # No threat


@dataclass
class MissionState:
    """Complete mission state for DreamerV3 observation."""
    # Buzzard state
    buzzard_health: float = 100.0
    buzzard_position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    buzzard_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    buzzard_wing_extension: float = 0.0
    
    # TAB states
    tabs_attached: Dict[str, bool] = field(default_factory=lambda: {
        "UP": True, "DOWN": True, "LEFT": True, "RIGHT": True
    })
    tabs_positions: Dict[str, np.ndarray] = field(default_factory=dict)
    tabs_available: int = 4
    
    # Slingshot state
    slingshot_active: bool = False
    slingshot_state: str = "DISPERSED"
    bola_position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    bola_velocity: float = 0.0
    swing_angular_velocity: float = 0.0
    
    # Threat state
    threat_count: int = 0
    threat_priority: ThreatPriority = ThreatPriority.NONE
    nearest_threat_distance: float = float('inf')
    nearest_threat_time_to_impact: float = float('inf')
    
    # Mission state
    mode: MissionMode = MissionMode.PATROL
    time_in_mode: float = 0.0
    intercepts_this_episode: int = 0
    damage_taken: float = 0.0


class RewardShaper:
    """
    Shapes rewards to implicitly encode the mission objective.
    
    DreamerV3 learns the objective through these signals, NOT through
    explicit goal programming. The world model's latent state
    comes to represent "protect Buzzard" as an emergent property.
    """
    
    def __init__(self):
        # PRIMARY: Buzzard survival (everything else is secondary)
        self.buzzard_alive_per_step = 1.0
        self.buzzard_damage_penalty = -100.0  # Per health point lost
        self.buzzard_destroyed = -10000.0      # Game over
        
        # SECONDARY: Successful defense
        self.threat_intercepted = 200.0        # TAB killed threat
        self.threat_evaded = 50.0              # Threat missed without TAB loss
        
        # TERTIARY: TAB sacrifice (acceptable for Buzzard safety)
        self.tab_sacrifice_for_kill = 100.0    # Died killing threat - heroic
        self.tab_lost_no_value = -50.0         # Died without contribution
        
        # SLINGSHOT mode rewards
        self.slingshot_consolidate = 5.0       # Learning to use bola mode
        self.slingshot_release_hit = 300.0     # Bola hit a threat
        self.slingshot_release_miss = -20.0    # Wasted the bola
        self.swing_momentum_bonus = 0.1        # Per unit of angular velocity
        
        # AIRFOILED BUZZARD rewards
        self.wing_deploy_tactical = 10.0       # Wings helped evade
        self.grid_fin_steer_success = 5.0      # Fins helped targeting
        
        # Exploration (helps discover better strategies)
        self.state_novelty = 2.0
        self.physics_discovery = 20.0
        self.new_maneuver = 50.0               # First time doing something
    
    def compute_reward(self, 
                       prev_state: MissionState, 
                       curr_state: MissionState,
                       events: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
        """
        Compute shaped reward from state transition.
        
        Returns: (total_reward, reward_breakdown)
        """
        breakdown = {}
        total = 0.0
        
        # PRIMARY: Buzzard survival
        if curr_state.buzzard_health > 0:
            breakdown['alive'] = self.buzzard_alive_per_step
            total += self.buzzard_alive_per_step
        else:
            breakdown['destroyed'] = self.buzzard_destroyed
            return self.buzzard_destroyed, breakdown
        
        # Damage taken
        damage = prev_state.buzzard_health - curr_state.buzzard_health
        if damage > 0:
            penalty = damage * self.buzzard_damage_penalty
            breakdown['damage'] = penalty
            total += penalty
        
        # Threat intercepts
        intercepts = events.get('intercepts', 0)
        if intercepts > 0:
            breakdown['intercepts'] = intercepts * self.threat_intercepted
            total += intercepts * self.threat_intercepted
        
        # TAB sacrifices
        sacrifices = events.get('heroic_sacrifices', 0)
        if sacrifices > 0:
            breakdown['sacrifices'] = sacrifices * self.tab_sacrifice_for_kill
            total += sacrifices * self.tab_sacrifice_for_kill
        
        # Slingshot mode
        if curr_state.slingshot_active:
            # Reward for building momentum
            momentum_reward = curr_state.swing_angular_velocity * self.swing_momentum_bonus
            breakdown['swing_momentum'] = momentum_reward
            total += momentum_reward
            
            # Reward for bola hit
            if events.get('bola_hit', False):
                breakdown['bola_hit'] = self.slingshot_release_hit
                total += self.slingshot_release_hit
        
        return total, breakdown


class UnifiedMissionController:
    """
    Unified controller that commands all KAPS subsystems.
    
    DreamerV3 interfaces with this controller, which translates
    latent decisions into coordinated actions across:
    - TAB array
    - Slingshot dynamics
    - Buzzard airfoil surfaces
    - Threat response
    
    The mission objective (PROTECT BUZZARD) is not explicitly
    programmed here - it emerges from the reward shaper and
    DreamerV3's learned world model.
    """
    
    def __init__(self):
        self.state = MissionState()
        self.reward_shaper = RewardShaper()
        self.mode = MissionMode.PATROL
        
        # Mode timing
        self.mode_start_time = 0.0
        self.time_in_mode = 0.0
        
        # Action history for learning
        self.action_history = []
        
        # Subsystem references (set by environment)
        self.slingshot_controller = None
        self.airfoiled_buzzard = None
        self.tab_array = None
        self.threat_spawner = None
    
    def register_subsystems(self,
                           slingshot=None,
                           buzzard_airfoil=None,
                           tab_array=None,
                           threats=None):
        """Register all subsystems for unified control."""
        self.slingshot_controller = slingshot
        self.airfoiled_buzzard = buzzard_airfoil
        self.tab_array = tab_array
        self.threat_spawner = threats
    
    def update_state(self, sim_state: Dict) -> MissionState:
        """
        Update mission state from simulation.
        
        This creates the observation that DreamerV3 sees.
        """
        prev_state = self.state
        
        # Buzzard state
        self.state.buzzard_health = sim_state.get('buzzard_health', 100.0)
        self.state.buzzard_position = np.array(sim_state.get('buzzard_position', [0, 0, 1000]))
        self.state.buzzard_velocity = np.array(sim_state.get('buzzard_velocity', [0, 50, 0]))
        
        if self.airfoiled_buzzard:
            self.state.buzzard_wing_extension = self.airfoiled_buzzard.wing_extension
        
        # TAB states
        if self.tab_array:
            for tab_id in ["UP", "DOWN", "LEFT", "RIGHT"]:
                tab = self.tab_array.tabs.get(tab_id)
                if tab:
                    self.state.tabs_attached[tab_id] = tab.is_attached
                    self.state.tabs_positions[tab_id] = tab.position
            self.state.tabs_available = sum(1 for v in self.state.tabs_attached.values() if v)
        
        # Slingshot state
        if self.slingshot_controller:
            self.state.slingshot_active = self.slingshot_controller.state.name != "DISPERSED"
            self.state.slingshot_state = self.slingshot_controller.state.name
            self.state.bola_position = self.slingshot_controller.bola.position
            self.state.bola_velocity = np.linalg.norm(self.slingshot_controller.bola.velocity)
            self.state.swing_angular_velocity = self.slingshot_controller.swing_angular_velocity
        
        # Threat state
        if self.threat_spawner:
            threats = self.threat_spawner.get_active_threats()
            self.state.threat_count = len(threats)
            
            if threats:
                # Find nearest/most urgent threat
                min_dist = float('inf')
                min_tti = float('inf')
                
                for threat in threats:
                    dist = np.linalg.norm(threat.position - self.state.buzzard_position)
                    if dist < min_dist:
                        min_dist = dist
                    
                    # Time to impact
                    closing = np.dot(
                        threat.velocity - self.state.buzzard_velocity,
                        self.state.buzzard_position - threat.position
                    ) / (dist + 0.001)
                    if closing > 0:
                        tti = dist / closing
                        if tti < min_tti:
                            min_tti = tti
                
                self.state.nearest_threat_distance = min_dist
                self.state.nearest_threat_time_to_impact = min_tti
                
                # Set priority
                if min_tti < 2.0:
                    self.state.threat_priority = ThreatPriority.CRITICAL
                elif min_tti < 5.0:
                    self.state.threat_priority = ThreatPriority.HIGH
                elif min_tti < 10.0:
                    self.state.threat_priority = ThreatPriority.MEDIUM
                else:
                    self.state.threat_priority = ThreatPriority.LOW
            else:
                self.state.threat_priority = ThreatPriority.NONE
        
        # Determine mission mode
        self._update_mode()
        
        return self.state
    
    def _update_mode(self):
        """Update mission mode based on state."""
        old_mode = self.mode
        
        # Mode transitions
        if self.state.buzzard_health < 50:
            self.mode = MissionMode.EVASIVE
        elif self.state.slingshot_active:
            self.mode = MissionMode.SLINGSHOT
        elif self.state.threat_priority in [ThreatPriority.CRITICAL, ThreatPriority.HIGH]:
            self.mode = MissionMode.ENGAGE
        elif self.state.threat_priority in [ThreatPriority.MEDIUM, ThreatPriority.LOW]:
            self.mode = MissionMode.ALERT
        elif self.state.tabs_available < 4:
            self.mode = MissionMode.RECOVERY
        else:
            self.mode = MissionMode.PATROL
        
        # Track time in mode
        if old_mode != self.mode:
            self.mode_start_time = 0.0
            self.time_in_mode = 0.0
        
        self.state.mode = self.mode
    
    def decode_action(self, action: np.ndarray) -> Dict[str, Any]:
        """
        Decode DreamerV3 action into subsystem commands.
        
        The action is a continuous vector that gets translated
        into specific commands for each subsystem.
        
        Action space layout (26 dimensions):
        [0-15]   - TAB controls (4 TABs × 4 channels each)
                   [elevator, aileron, rudder, cable_tension]
        [16-19]  - TAB release commands (4)
        [20]     - Slingshot mode toggle
        [21]     - Slingshot swing torque
        [22]     - Slingshot release
        [23]     - Buzzard wing extension
        [24-25]  - Buzzard pitch/yaw bias
        """
        commands = {}
        
        # TAB controls
        commands['tabs'] = {}
        for i, tab_id in enumerate(["UP", "DOWN", "LEFT", "RIGHT"]):
            base = i * 4
            commands['tabs'][tab_id] = {
                'elevator': action[base + 0] if len(action) > base else 0,
                'aileron': action[base + 1] if len(action) > base + 1 else 0,
                'rudder': action[base + 2] if len(action) > base + 2 else 0,
                'tension': action[base + 3] if len(action) > base + 3 else 0.5,
            }
        
        # TAB releases
        commands['releases'] = {}
        for i, tab_id in enumerate(["UP", "DOWN", "LEFT", "RIGHT"]):
            idx = 16 + i
            commands['releases'][tab_id] = action[idx] > 0.5 if len(action) > idx else False
        
        # Slingshot
        if len(action) > 20:
            commands['slingshot'] = {
                'toggle': action[20] > 0.5,
                'torque': action[21] if len(action) > 21 else 0,
                'release': action[22] > 0.5 if len(action) > 22 else False,
            }
        
        # Buzzard airfoil
        if len(action) > 23:
            commands['buzzard'] = {
                'wing_extension': (action[23] + 1) / 2,  # Map -1,1 to 0,1
                'pitch_bias': action[24] if len(action) > 24 else 0,
                'yaw_bias': action[25] if len(action) > 25 else 0,
            }
        
        return commands
    
    def execute_commands(self, commands: Dict[str, Any], dt: float):
        """
        Execute decoded commands across all subsystems.
        """
        # TAB controls
        if self.tab_array and 'tabs' in commands:
            for tab_id, ctrl in commands['tabs'].items():
                tab = self.tab_array.tabs.get(tab_id)
                if tab and tab.is_attached:
                    tab.set_control_surfaces(
                        elevator=ctrl['elevator'],
                        aileron=ctrl['aileron'],
                        rudder=ctrl['rudder']
                    )
                    tab.set_cable_tension_target(ctrl['tension'])
        
        # TAB releases
        if self.tab_array and 'releases' in commands:
            for tab_id, should_release in commands['releases'].items():
                if should_release:
                    tab = self.tab_array.tabs.get(tab_id)
                    if tab and tab.is_attached:
                        tab.release()
        
        # Slingshot
        if self.slingshot_controller and 'slingshot' in commands:
            ss = commands['slingshot']
            
            if ss.get('toggle', False):
                # Toggle slingshot mode
                if self.slingshot_controller.state.name == "DISPERSED":
                    # Consolidate TABs
                    tab_states = {
                        tid: {
                            'position': t.position,
                            'velocity': t.velocity,
                            'mass': 5.0,
                            'is_attached': t.is_attached
                        }
                        for tid, t in self.tab_array.tabs.items()
                    } if self.tab_array else {}
                    self.slingshot_controller.consolidate_tabs(tab_states)
                    self.slingshot_controller.begin_swing(
                        self.state.buzzard_position,
                        initial_angular_rate=1.0
                    )
            
            if ss.get('torque', 0) != 0:
                self.slingshot_controller.accelerate_swing(
                    torque=ss['torque'] * 5000,  # Scale to Nm
                    dt=dt
                )
            
            if ss.get('release', False):
                self.slingshot_controller.release()
                self.slingshot_controller.deploy_grid_fins()
        
        # Buzzard airfoil
        if self.airfoiled_buzzard and 'buzzard' in commands:
            buz = commands['buzzard']
            self.airfoiled_buzzard.extend_wings(
                target_extension=buz.get('wing_extension', 0),
                dt=dt
            )
            self.airfoiled_buzzard.angle_of_attack = buz.get('pitch_bias', 0) * 15
            self.airfoiled_buzzard.bank_angle = buz.get('yaw_bias', 0) * 30
    
    def get_observation_vector(self) -> np.ndarray:
        """
        Create observation vector for DreamerV3.
        
        This is what the world model sees. The objective (protect Buzzard)
        is NOT explicitly encoded - it emerges from learning with rewards.
        """
        obs = []
        
        # Buzzard state (9)
        obs.extend(self.state.buzzard_position)
        obs.extend(self.state.buzzard_velocity)
        obs.append(self.state.buzzard_health / 100.0)
        obs.append(self.state.buzzard_wing_extension)
        obs.append(float(self.mode.value) / 6.0)  # Normalized mode
        
        # TAB states (4 × 8 = 32)
        for tab_id in ["UP", "DOWN", "LEFT", "RIGHT"]:
            if tab_id in self.state.tabs_positions:
                obs.extend(self.state.tabs_positions[tab_id])
            else:
                obs.extend([0, 0, 0])
            obs.append(1.0 if self.state.tabs_attached.get(tab_id, False) else 0.0)
            # Relative position to buzzard
            if tab_id in self.state.tabs_positions:
                rel = self.state.tabs_positions[tab_id] - self.state.buzzard_position
                obs.extend(rel / 100.0)  # Normalized
            else:
                obs.extend([0, 0, 0])
            obs.append(0)  # Placeholder for cable tension
        
        # Slingshot state (6)
        obs.append(1.0 if self.state.slingshot_active else 0.0)
        obs.append(self.state.swing_angular_velocity / 10.0)
        obs.extend(self.state.bola_position / 100.0)
        obs.append(self.state.bola_velocity / 100.0)
        
        # Threat state (8)
        obs.append(self.state.threat_count / 10.0)
        obs.append(float(self.state.threat_priority.value) / 5.0)
        obs.append(min(1.0, self.state.nearest_threat_distance / 500.0))
        obs.append(min(1.0, self.state.nearest_threat_time_to_impact / 10.0))
        obs.extend([0, 0, 0, 0])  # Placeholder for threat positions
        
        return np.array(obs, dtype=np.float32)
    
    def get_extended_action_space_dim(self) -> int:
        """Return dimension of extended action space."""
        return 26  # TABs(16) + releases(4) + slingshot(3) + buzzard(3)


# =============================================================================
# INTEGRATION HELPERS
# =============================================================================

def create_unified_observation_space():
    """Create gym observation space for unified controller."""
    try:
        from gymnasium import spaces
    except ImportError:
        from gym import spaces
    
    # 9 + 32 + 6 + 8 = 55 base
    obs_dim = 55
    
    return spaces.Box(
        low=-np.inf, high=np.inf,
        shape=(obs_dim,), dtype=np.float32
    )


def create_unified_action_space():
    """Create gym action space for unified controller."""
    try:
        from gymnasium import spaces
    except ImportError:
        from gym import spaces
    
    return spaces.Box(
        low=-1.0, high=1.0,
        shape=(26,), dtype=np.float32
    )


if __name__ == "__main__":
    print("=== Unified Mission Controller ===\n")
    
    controller = UnifiedMissionController()
    
    print("Mission Modes:")
    for mode in MissionMode:
        print(f"  {mode.name}")
    
    print("\nThreat Priority Levels:")
    for prio in ThreatPriority:
        print(f"  {prio.name}")
    
    print("\nObservation space:", create_unified_observation_space())
    print("Action space:", create_unified_action_space())
    
    print("\nReward shaping (implicit objective encoding):")
    shaper = RewardShaper()
    print(f"  Buzzard alive: +{shaper.buzzard_alive_per_step}/step")
    print(f"  Buzzard damage: {shaper.buzzard_damage_penalty}/point")
    print(f"  Threat intercepted: +{shaper.threat_intercepted}")
    print(f"  Bola hit: +{shaper.slingshot_release_hit}")
    
    print("\n=== Test Complete ===")
