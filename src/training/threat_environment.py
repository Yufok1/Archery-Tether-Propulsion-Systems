"""
Hostile Threat Environment
===========================
Realistic threat scenarios for DreamerV3 to learn defensive maneuvers.

Threat Types:
- IR Missiles (heat-seeking, fast, single)
- Radar-guided missiles (predictive intercept)
- Enemy drones (slower, maneuvering)
- Swarm attacks (multiple simultaneous)
- Loitering munitions (slow approach, sudden sprint)

The environment is designed for EXPLORATION - the dreamer should discover
what physics maneuvers are possible, not just react to threats.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum
import random


class ThreatBehavior(Enum):
    """How threats approach and maneuver"""
    DIRECT = "direct"              # Straight line to target
    LEAD_PURSUIT = "lead_pursuit"  # Aims ahead of target
    PROPORTIONAL_NAV = "pro_nav"   # Missile guidance law
    WEAVING = "weaving"            # Evasive approach
    LOITER_SPRINT = "loiter_sprint"  # Slow then fast
    SWARM_COORDINATED = "swarm"    # Multiple threats coordinate


class ThreatType(Enum):
    """Realistic threat categories"""
    IR_MISSILE = "ir_missile"           # Fast, heat-seeking
    RADAR_MISSILE = "radar_missile"     # Very fast, predictive
    ATTACK_DRONE = "attack_drone"       # Slow, maneuverable
    LOITERING_MUNITION = "loiterer"     # Slow approach, sprint attack
    SWARM_ELEMENT = "swarm"             # Coordinated group
    BALLISTIC = "ballistic"             # Unguided, predictable


@dataclass
class ThreatProfile:
    """Physical characteristics of a threat"""
    type: ThreatType
    speed_min: float      # m/s
    speed_max: float      # m/s
    turn_rate: float      # deg/s max turn
    size: float           # m radius (for intercept detection)
    behavior: ThreatBehavior
    damage: float         # Damage dealt on impact
    
    # Behavior parameters
    lead_time: float = 0.0      # Seconds to lead target
    weave_amplitude: float = 0.0
    weave_frequency: float = 0.0
    sprint_trigger_dist: float = 0.0  # Distance to trigger sprint


# Threat library
THREAT_PROFILES = {
    ThreatType.IR_MISSILE: ThreatProfile(
        type=ThreatType.IR_MISSILE,
        speed_min=300, speed_max=400,
        turn_rate=20, size=0.15,
        behavior=ThreatBehavior.PROPORTIONAL_NAV,
        damage=100,
        lead_time=0.5
    ),
    ThreatType.RADAR_MISSILE: ThreatProfile(
        type=ThreatType.RADAR_MISSILE,
        speed_min=500, speed_max=700,
        turn_rate=30, size=0.2,
        behavior=ThreatBehavior.LEAD_PURSUIT,
        damage=150,
        lead_time=1.0
    ),
    ThreatType.ATTACK_DRONE: ThreatProfile(
        type=ThreatType.ATTACK_DRONE,
        speed_min=50, speed_max=80,
        turn_rate=45, size=0.5,
        behavior=ThreatBehavior.WEAVING,
        damage=75,
        weave_amplitude=30, weave_frequency=0.5
    ),
    ThreatType.LOITERING_MUNITION: ThreatProfile(
        type=ThreatType.LOITERING_MUNITION,
        speed_min=30, speed_max=150,
        turn_rate=60, size=0.3,
        behavior=ThreatBehavior.LOITER_SPRINT,
        damage=80,
        sprint_trigger_dist=100
    ),
    ThreatType.SWARM_ELEMENT: ThreatProfile(
        type=ThreatType.SWARM_ELEMENT,
        speed_min=40, speed_max=60,
        turn_rate=90, size=0.2,
        behavior=ThreatBehavior.SWARM_COORDINATED,
        damage=30  # Low individual damage, high collective
    ),
    ThreatType.BALLISTIC: ThreatProfile(
        type=ThreatType.BALLISTIC,
        speed_min=200, speed_max=300,
        turn_rate=0, size=0.1,
        behavior=ThreatBehavior.DIRECT,
        damage=50
    ),
}


@dataclass
class Threat:
    """Active threat in the environment"""
    threat_id: str
    profile: ThreatProfile
    position: np.ndarray
    velocity: np.ndarray
    alive: bool = True
    age: float = 0.0
    
    # State for complex behaviors
    phase: str = "approach"  # approach, sprint, terminal
    weave_phase: float = 0.0
    
    def update(self, dt: float, target_pos: np.ndarray, target_vel: np.ndarray):
        """Update threat position based on behavior"""
        if not self.alive:
            return
        
        self.age += dt
        
        # Get desired direction based on behavior
        if self.profile.behavior == ThreatBehavior.DIRECT:
            desired_dir = self._direct_guidance(target_pos)
        elif self.profile.behavior == ThreatBehavior.LEAD_PURSUIT:
            desired_dir = self._lead_pursuit(target_pos, target_vel)
        elif self.profile.behavior == ThreatBehavior.PROPORTIONAL_NAV:
            desired_dir = self._proportional_nav(target_pos, target_vel)
        elif self.profile.behavior == ThreatBehavior.WEAVING:
            desired_dir = self._weaving_approach(target_pos, dt)
        elif self.profile.behavior == ThreatBehavior.LOITER_SPRINT:
            desired_dir = self._loiter_sprint(target_pos)
        else:
            desired_dir = self._direct_guidance(target_pos)
        
        # Apply turn rate limit
        current_dir = self.velocity / (np.linalg.norm(self.velocity) + 1e-8)
        max_turn = np.radians(self.profile.turn_rate) * dt
        
        # Angle between current and desired
        cos_angle = np.clip(np.dot(current_dir, desired_dir), -1, 1)
        angle = np.arccos(cos_angle)
        
        if angle > max_turn:
            # Limited turn
            axis = np.cross(current_dir, desired_dir)
            axis_norm = np.linalg.norm(axis)
            if axis_norm > 1e-8:
                axis = axis / axis_norm
                # Rodrigues rotation
                new_dir = (current_dir * np.cos(max_turn) + 
                          np.cross(axis, current_dir) * np.sin(max_turn))
            else:
                new_dir = desired_dir
        else:
            new_dir = desired_dir
        
        # Update velocity with speed
        speed = np.linalg.norm(self.velocity)
        self.velocity = new_dir * speed
        
        # Update position
        self.position += self.velocity * dt
    
    def _direct_guidance(self, target_pos: np.ndarray) -> np.ndarray:
        """Point directly at target"""
        direction = target_pos - self.position
        return direction / (np.linalg.norm(direction) + 1e-8)
    
    def _lead_pursuit(self, target_pos: np.ndarray, target_vel: np.ndarray) -> np.ndarray:
        """Aim ahead of target"""
        lead_pos = target_pos + target_vel * self.profile.lead_time
        direction = lead_pos - self.position
        return direction / (np.linalg.norm(direction) + 1e-8)
    
    def _proportional_nav(self, target_pos: np.ndarray, target_vel: np.ndarray) -> np.ndarray:
        """Proportional navigation (missile guidance)"""
        # Line of sight rate
        los = target_pos - self.position
        los_dist = np.linalg.norm(los)
        los_unit = los / (los_dist + 1e-8)
        
        # Closing velocity
        rel_vel = target_vel - self.velocity
        los_rate = np.cross(los, rel_vel) / (los_dist * los_dist + 1e-8)
        
        # Pro-nav command (N * Vc * los_rate)
        N = 4  # Navigation constant
        closing_speed = -np.dot(rel_vel, los_unit)
        command = N * closing_speed * np.cross(los_unit, los_rate)
        
        # Add to current direction
        current_dir = self.velocity / (np.linalg.norm(self.velocity) + 1e-8)
        new_dir = current_dir + command * 0.01
        return new_dir / (np.linalg.norm(new_dir) + 1e-8)
    
    def _weaving_approach(self, target_pos: np.ndarray, dt: float) -> np.ndarray:
        """Approach with evasive weaving"""
        self.weave_phase += self.profile.weave_frequency * dt * 2 * np.pi
        
        # Base direction to target
        base_dir = self._direct_guidance(target_pos)
        
        # Add perpendicular weave
        perp = np.array([-base_dir[1], base_dir[0], 0])
        perp = perp / (np.linalg.norm(perp) + 1e-8)
        
        weave = perp * np.sin(self.weave_phase) * np.radians(self.profile.weave_amplitude)
        new_dir = base_dir + weave
        return new_dir / (np.linalg.norm(new_dir) + 1e-8)
    
    def _loiter_sprint(self, target_pos: np.ndarray) -> np.ndarray:
        """Loiter at distance, then sprint to attack"""
        dist = np.linalg.norm(target_pos - self.position)
        
        if dist < self.profile.sprint_trigger_dist:
            # Sprint phase - max speed direct approach
            self.phase = "sprint"
            speed = self.profile.speed_max
        else:
            # Loiter phase - slow circling approach
            self.phase = "loiter"
            speed = self.profile.speed_min
        
        # Update speed
        current_speed = np.linalg.norm(self.velocity)
        new_speed = current_speed + (speed - current_speed) * 0.1  # Smooth transition
        current_dir = self.velocity / (current_speed + 1e-8)
        self.velocity = current_dir * new_speed
        
        return self._direct_guidance(target_pos)


class ThreatSpawner:
    """Spawns threats with realistic scenarios"""
    
    def __init__(self, 
                 spawn_radius_min: float = 300,
                 spawn_radius_max: float = 600,
                 max_threats: int = 12):
        self.spawn_radius_min = spawn_radius_min
        self.spawn_radius_max = spawn_radius_max
        self.max_threats = max_threats
        
        self.threats: List[Threat] = []
        self.threat_counter = 0
        self.wave_counter = 0
        
        # Scenario weights (can be adjusted for curriculum)
        self.scenario_weights = {
            'single_missile': 0.3,
            'double_missile': 0.15,
            'drone_attack': 0.2,
            'swarm': 0.1,
            'loiterer': 0.15,
            'mixed': 0.1,
        }
    
    def spawn_scenario(self, target_pos: np.ndarray, target_vel: np.ndarray) -> List[Threat]:
        """Spawn a threat scenario based on weights"""
        scenario = random.choices(
            list(self.scenario_weights.keys()),
            weights=list(self.scenario_weights.values())
        )[0]
        
        self.wave_counter += 1
        
        if scenario == 'single_missile':
            return self._spawn_missiles(target_pos, target_vel, count=1)
        elif scenario == 'double_missile':
            return self._spawn_missiles(target_pos, target_vel, count=2)
        elif scenario == 'drone_attack':
            return self._spawn_drones(target_pos, target_vel, count=2)
        elif scenario == 'swarm':
            return self._spawn_swarm(target_pos, target_vel, count=5)
        elif scenario == 'loiterer':
            return self._spawn_loiterers(target_pos, target_vel, count=2)
        elif scenario == 'mixed':
            return self._spawn_mixed(target_pos, target_vel)
        
        return []
    
    def _spawn_threat(self, 
                      target_pos: np.ndarray,
                      target_vel: np.ndarray,
                      threat_type: ThreatType,
                      angle_offset: float = 0) -> Threat:
        """Spawn a single threat"""
        self.threat_counter += 1
        
        profile = THREAT_PROFILES[threat_type]
        
        # Random spawn position on sphere around target
        theta = random.uniform(0, 2 * np.pi) + angle_offset
        phi = random.uniform(-np.pi/3, np.pi/3)  # Favor horizontal approaches
        dist = random.uniform(self.spawn_radius_min, self.spawn_radius_max)
        
        offset = np.array([
            dist * np.cos(phi) * np.cos(theta),
            dist * np.cos(phi) * np.sin(theta),
            dist * np.sin(phi)
        ])
        
        position = target_pos + offset
        
        # Initial velocity toward target
        direction = -offset / (np.linalg.norm(offset) + 1e-8)
        speed = random.uniform(profile.speed_min, profile.speed_max)
        velocity = direction * speed
        
        threat = Threat(
            threat_id=f"THREAT-{self.threat_counter:04d}",
            profile=profile,
            position=position,
            velocity=velocity
        )
        
        self.threats.append(threat)
        return threat
    
    def _spawn_missiles(self, target_pos, target_vel, count: int) -> List[Threat]:
        """Spawn missile attack"""
        threats = []
        for i in range(count):
            t_type = random.choice([ThreatType.IR_MISSILE, ThreatType.RADAR_MISSILE])
            angle_offset = i * (2 * np.pi / count)  # Spread around
            threats.append(self._spawn_threat(target_pos, target_vel, t_type, angle_offset))
        return threats
    
    def _spawn_drones(self, target_pos, target_vel, count: int) -> List[Threat]:
        """Spawn attack drones"""
        threats = []
        for i in range(count):
            angle_offset = i * (2 * np.pi / count)
            threats.append(self._spawn_threat(target_pos, target_vel, 
                                             ThreatType.ATTACK_DRONE, angle_offset))
        return threats
    
    def _spawn_swarm(self, target_pos, target_vel, count: int) -> List[Threat]:
        """Spawn coordinated swarm"""
        threats = []
        base_angle = random.uniform(0, 2 * np.pi)
        for i in range(count):
            angle_offset = base_angle + (i - count//2) * 0.2  # Tight formation
            threats.append(self._spawn_threat(target_pos, target_vel,
                                             ThreatType.SWARM_ELEMENT, angle_offset))
        return threats
    
    def _spawn_loiterers(self, target_pos, target_vel, count: int) -> List[Threat]:
        """Spawn loitering munitions"""
        threats = []
        for i in range(count):
            angle_offset = i * (2 * np.pi / count)
            threats.append(self._spawn_threat(target_pos, target_vel,
                                             ThreatType.LOITERING_MUNITION, angle_offset))
        return threats
    
    def _spawn_mixed(self, target_pos, target_vel) -> List[Threat]:
        """Spawn mixed threat scenario"""
        threats = []
        threats.append(self._spawn_threat(target_pos, target_vel, ThreatType.IR_MISSILE))
        threats.append(self._spawn_threat(target_pos, target_vel, ThreatType.ATTACK_DRONE, np.pi/2))
        threats.append(self._spawn_threat(target_pos, target_vel, ThreatType.LOITERING_MUNITION, np.pi))
        return threats
    
    def update_all(self, dt: float, target_pos: np.ndarray, target_vel: np.ndarray):
        """Update all active threats"""
        for threat in self.threats:
            if threat.alive:
                threat.update(dt, target_pos, target_vel)
    
    def get_active_threats(self) -> List[Threat]:
        """Get list of active threats"""
        return [t for t in self.threats if t.alive]
    
    def remove_dead(self):
        """Clean up dead threats"""
        self.threats = [t for t in self.threats if t.alive]
    
    def check_intercepts(self, 
                         interceptor_positions: Dict[str, np.ndarray],
                         intercept_radius: float = 2.0) -> List[Tuple[str, str]]:
        """Check if any interceptors hit threats"""
        intercepts = []
        
        for threat in self.threats:
            if not threat.alive:
                continue
            
            for tab_id, tab_pos in interceptor_positions.items():
                dist = np.linalg.norm(threat.position - tab_pos)
                hit_radius = intercept_radius + threat.profile.size
                
                if dist < hit_radius:
                    threat.alive = False
                    intercepts.append((tab_id, threat.threat_id))
                    print(f"[INTERCEPT] {tab_id} destroyed {threat.threat_id}!")
        
        return intercepts
    
    def check_impacts(self, target_pos: np.ndarray, impact_radius: float = 5.0) -> List[Threat]:
        """Check if any threats hit the target (mother drone)"""
        impacts = []
        
        for threat in self.threats:
            if not threat.alive:
                continue
            
            dist = np.linalg.norm(threat.position - target_pos)
            if dist < impact_radius + threat.profile.size:
                threat.alive = False
                impacts.append(threat)
                print(f"[IMPACT] {threat.threat_id} hit target! Damage: {threat.profile.damage}")
        
        return impacts
