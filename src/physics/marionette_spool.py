"""
MARIONETTE SPOOL - Central Authority for Sail Control
======================================================

The SPOOL is the physical manifestation of the champion brain.
It's a coiled mechanism that:

1. STORES cables wound together (collapsed state)
2. UNRAVELS during throw (centrifugal deployment)  
3. CONTROLS via differential cable lengths (marionette mode)
4. SENSES via cable tensions (proprioception)

Physical Architecture:
                    
    COLLAPSED (pre-throw)          DEPLOYED (marionette mode)
    
         ┌───────┐                      SAILS
         │███████│                    ⛵     ⛵
         │███████│◄── All cables        \   /
         │███████│    wound tight        \ /
         │SPOOL  │                    ┌───┴───┐
         └───────┘                    │ SPOOL │◄── Champion brain
              │                       │CONTROL│    runs HERE
              ▼                       └───┬───┘
           Human                          │
           hand                      individual
                                     cable control
                                     
The spool mechanism contains:
- N cable drums (one per sail)
- Tension sensors on each drum
- Motor/brake for each drum (or passive ratchet)
- The CHAMPION DreamerV3 as the control authority
- Cascade-lattice as the computational substrate

Tension sensing flows: Sails → Cables → Spool drums → Champion
Control flows: Champion → Spool drums → Cable lengths → Sail pitch
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
from pathlib import Path
import time

# Cascade-lattice is the computational substrate
try:
    import cascade
    from cascade import Hold, HoldAwareMixin
    CASCADE_AVAILABLE = True
except ImportError:
    CASCADE_AVAILABLE = False
    class HoldAwareMixin:
        pass

# Champion brain
try:
    from champion_gen42 import DreamerV3RSSM, TrainingConfig
    CHAMPION_AVAILABLE = True
except ImportError:
    CHAMPION_AVAILABLE = False


class SpoolState(Enum):
    """Operating state of the spool mechanism."""
    WOUND = "wound"           # Cables coiled, ready for throw
    UNRAVELING = "unraveling" # Actively spinning out
    TENSIONED = "tensioned"   # Deployed, under centrifugal tension
    CONTROL = "control"       # Marionette mode - active control


@dataclass
class CableDrum:
    """
    A single drum in the spool that controls one sail.
    
    Physical mechanism:
    - Wound cable around drum
    - Tension sensor (strain gauge)
    - Brake/motor for length control
    """
    drum_id: str
    sail_id: str
    
    # Cable properties
    total_length: float = 15.0      # meters of cable on drum
    deployed_length: float = 0.0    # current extension
    max_length: float = 12.0        # max deployment
    
    # Physical state
    tension: float = 0.0            # N - current tension reading
    tension_rate: float = 0.0       # N/s - rate of change
    angular_velocity: float = 0.0   # rad/s - drum spin rate
    
    # Control
    brake_engaged: float = 0.0      # 0-1 brake force
    motor_torque: float = 0.0       # Nm - active reel/unreel
    
    # Drum geometry
    drum_radius: float = 0.03       # 3cm drum radius
    cable_diameter: float = 0.002   # 2mm cable
    
    def wound_radius(self) -> float:
        """Current radius of wound cable on drum."""
        remaining = self.total_length - self.deployed_length
        # Approximate: layers of cable
        layers = remaining / (2 * np.pi * self.drum_radius)
        return self.drum_radius + layers * self.cable_diameter
    
    def unreel(self, dt: float, spin_rate: float) -> float:
        """
        Unreel cable based on drum spin.
        Returns cable paid out this step.
        """
        if self.deployed_length >= self.max_length:
            return 0.0
        
        # Centrifugal unreeling
        effective_radius = self.wound_radius()
        payout = spin_rate * effective_radius * dt * (1 - self.brake_engaged)
        
        self.deployed_length = min(self.deployed_length + payout, self.max_length)
        return payout
    
    def adjust_length(self, delta: float, dt: float):
        """
        Active length adjustment (marionette control).
        Positive = reel in, negative = pay out.
        """
        max_rate = 2.0  # m/s max reel rate
        actual_delta = np.clip(delta, -max_rate * dt, max_rate * dt)
        
        new_length = self.deployed_length - actual_delta  # Reel in = shorter deployed
        self.deployed_length = np.clip(new_length, 0.0, self.max_length)
    
    def update_tension(self, force_on_sail: np.ndarray, cable_direction: np.ndarray):
        """Update tension from physics."""
        old_tension = self.tension
        self.tension = max(0, np.dot(force_on_sail, cable_direction))
        self.tension_rate = (self.tension - old_tension) / 0.01


@dataclass 
class MarionetteSpool(HoldAwareMixin):
    """
    The central spool mechanism - WHERE THE CHAMPION LIVES.
    
    This is the physical controller that:
    1. Holds all cables wound together
    2. Unravels during throw
    3. Becomes the marionette master at max extension
    4. Uses cascade-lattice as computational substrate
    5. Runs the champion DreamerV3 for control decisions
    """
    
    # Identity
    spool_id: str = "marionette_alpha"
    
    # Drums (one per sail)
    drums: Dict[str, CableDrum] = field(default_factory=dict)
    
    # Physical state
    position: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    spin_rate: float = 0.0          # rad/s - overall spin
    spin_phase: float = 0.0         # rad - current phase
    
    # Operating state
    state: SpoolState = SpoolState.WOUND
    deployment_progress: float = 0.0  # 0-1 how deployed
    
    # Champion brain (THE CENTRAL AUTHORITY)
    champion: Optional[any] = None
    champion_state: np.ndarray = field(default_factory=lambda: np.zeros(256))
    
    # Cascade substrate
    cascade_enabled: bool = CASCADE_AVAILABLE
    
    # Control outputs
    collective_command: float = 0.0   # Collective cable adjustment
    cyclic_amplitude: float = 0.0     # Cyclic variation amplitude
    cyclic_phase: float = 0.0         # Cyclic phase offset
    
    # External command (Yondu whistle)
    external_command: Optional[np.ndarray] = None
    
    def __post_init__(self):
        """Initialize cascade substrate if available."""
        if CASCADE_AVAILABLE:
            try:
                # Register with cascade for computational substrate
                cascade.register_component(self.spool_id, self)
            except:
                pass
    
    def add_drum(self, sail_id: str, cable_length: float = 15.0) -> CableDrum:
        """Add a drum for a sail."""
        drum = CableDrum(
            drum_id=f"drum_{sail_id}",
            sail_id=sail_id,
            total_length=cable_length,
            max_length=cable_length * 0.8  # 80% max deployment
        )
        self.drums[sail_id] = drum
        return drum
    
    def load_champion(self, champion_path: Path):
        """
        Load the champion brain into the spool.
        
        The champion becomes the MARIONETTE MASTER -
        it reads tension from drums and commands cable lengths.
        """
        if CHAMPION_AVAILABLE and champion_path.exists():
            config = TrainingConfig()
            self.champion = DreamerV3RSSM(config)
            self.champion.load(champion_path)
            print(f"[SPOOL] Champion loaded from {champion_path}")
            print(f"[SPOOL] Marionette control ACTIVE")
        else:
            print(f"[SPOOL] No champion - using reactive control")
    
    def throw(self, throw_velocity: np.ndarray, initial_spin: float):
        """
        Human throws the spool.
        
        Transitions from WOUND to UNRAVELING state.
        """
        self.velocity = np.array(throw_velocity, dtype=np.float64)
        self.spin_rate = float(initial_spin)
        self.state = SpoolState.UNRAVELING
        
        # All drums start unreeling
        for drum in self.drums.values():
            drum.brake_engaged = 0.0
        
        print(f"[SPOOL] THROWN! v={np.linalg.norm(throw_velocity):.1f}m/s, spin={initial_spin:.1f}rad/s")
    
    def step_unraveling(self, dt: float) -> Dict[str, float]:
        """
        Step during unraveling phase.
        
        Cables pay out due to centrifugal force.
        Returns cable lengths for physics.
        """
        cable_lengths = {}
        total_deployed = 0
        total_max = 0
        
        for sail_id, drum in self.drums.items():
            payout = drum.unreel(dt, self.spin_rate)
            cable_lengths[sail_id] = drum.deployed_length
            total_deployed += drum.deployed_length
            total_max += drum.max_length
        
        self.deployment_progress = total_deployed / max(total_max, 1)
        
        # Transition to TENSIONED when mostly deployed
        if self.deployment_progress > 0.95:
            self.state = SpoolState.TENSIONED
            print(f"[SPOOL] DEPLOYED - entering tension mode")
            
            # Brief tensioning, then control
            self._transition_to_control()
        
        return cable_lengths
    
    def _transition_to_control(self):
        """Transition from passive tension to active control."""
        self.state = SpoolState.CONTROL
        
        # Engage light brake on all drums for control authority
        for drum in self.drums.values():
            drum.brake_engaged = 0.3
        
        print(f"[SPOOL] MARIONETTE MODE ACTIVE")
        if self.champion:
            print(f"[SPOOL] Champion brain controlling {len(self.drums)} sails")
    
    def sense(self) -> np.ndarray:
        """
        Build observation from all drum tensions.
        
        The spool FEELS the sails through cable tension.
        """
        obs_parts = []
        
        # Spool state
        obs_parts.extend([
            self.spin_rate / 20.0,
            np.sin(self.spin_phase),
            np.cos(self.spin_phase),
            self.position[2] / 100.0,  # Altitude
            self.velocity[2] / 20.0,   # Vertical velocity
        ])
        
        # Each drum's tension (THIS IS THE KEY SENSING)
        for sail_id in sorted(self.drums.keys()):
            drum = self.drums[sail_id]
            obs_parts.extend([
                drum.tension / 500.0,          # Normalized tension
                drum.tension_rate / 100.0,      # Rate
                drum.deployed_length / drum.max_length,  # Extension
            ])
        
        # External command if present
        if self.external_command is not None:
            obs_parts.extend(list(self.external_command / 10.0))
        else:
            obs_parts.extend([0, 0, 0])
        
        # Pad to fixed size
        while len(obs_parts) < 32:
            obs_parts.append(0.0)
        
        return np.array(obs_parts[:32], dtype=np.float32)
    
    def decide(self, observation: np.ndarray) -> Dict[str, float]:
        """
        Champion brain decides cable adjustments.
        
        Returns length delta for each cable.
        """
        if self.champion is not None:
            # Full DreamerV3 inference
            action, self.champion_state = self.champion.infer(
                observation, self.champion_state
            )
        else:
            # Reactive fallback - basic hover control
            action = self._reactive_control(observation)
        
        # Decode action to cable commands
        # Action: [collective, cyclic_amp, cyclic_phase, ...]
        self.collective_command = action[0] if len(action) > 0 else 0
        self.cyclic_amplitude = action[1] * 0.1 if len(action) > 1 else 0
        self.cyclic_phase = action[2] * np.pi if len(action) > 2 else 0
        
        # Generate per-drum commands
        commands = {}
        n_drums = len(self.drums)
        
        for i, sail_id in enumerate(sorted(self.drums.keys())):
            # Angular position of this sail
            angular_pos = 2 * np.pi * i / n_drums + self.spin_phase
            
            # Collective + cyclic (like helicopter)
            cyclic = self.cyclic_amplitude * np.sin(angular_pos - self.cyclic_phase)
            
            # Length delta: positive = reel in = pitch up
            commands[sail_id] = self.collective_command * 0.01 + cyclic * 0.005
        
        return commands
    
    def _reactive_control(self, observation: np.ndarray) -> np.ndarray:
        """Simple reactive control when no champion."""
        # Extract from observation
        altitude = observation[3] * 100
        vertical_vel = observation[4] * 20
        
        # Altitude hold
        target_alt = 30.0
        alt_error = target_alt - altitude
        collective = np.clip(alt_error * 0.01 + vertical_vel * 0.05, -1, 1)
        
        # Respond to external command
        if self.external_command is not None:
            cyclic_amp = np.linalg.norm(self.external_command[:2]) * 0.1
            cyclic_phase = np.arctan2(self.external_command[1], self.external_command[0])
        else:
            cyclic_amp = 0
            cyclic_phase = 0
        
        return np.array([collective, cyclic_amp, cyclic_phase / np.pi])
    
    def step(self, dt: float, sail_forces: Dict[str, np.ndarray], 
             sail_positions: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Main step function.
        
        1. Update tensions from sail forces
        2. Sense through drums
        3. Champion decides
        4. Command cable adjustments
        
        Returns cable length adjustments.
        """
        # Update spool physics
        gravity = np.array([0, 0, -9.81])
        
        # Compute net force from all cables
        net_cable_force = np.zeros(3)
        spool_pos = self.position
        
        for sail_id, drum in self.drums.items():
            if sail_id in sail_positions:
                cable_dir = sail_positions[sail_id] - spool_pos
                cable_dist = np.linalg.norm(cable_dir)
                if cable_dist > 0.1:
                    cable_dir = cable_dir / cable_dist
                    
                    # Update drum tension
                    if sail_id in sail_forces:
                        drum.update_tension(sail_forces[sail_id], cable_dir)
                    
                    # Cable pulls spool toward sail
                    net_cable_force += cable_dir * drum.tension
        
        # Spool mass (small - it's just the mechanism)
        spool_mass = 0.5  # kg
        accel = (gravity * spool_mass + net_cable_force * 0.01) / spool_mass
        
        self.velocity += accel * dt
        self.position += self.velocity * dt
        
        # Update spin
        self.spin_phase += self.spin_rate * dt
        self.spin_phase = self.spin_phase % (2 * np.pi)
        
        # State machine
        if self.state == SpoolState.UNRAVELING:
            return self.step_unraveling(dt)
        elif self.state in [SpoolState.TENSIONED, SpoolState.CONTROL]:
            # Sense and decide
            obs = self.sense()
            commands = self.decide(obs)
            
            # Apply cable adjustments
            lengths = {}
            for sail_id, delta in commands.items():
                self.drums[sail_id].adjust_length(delta, dt)
                lengths[sail_id] = self.drums[sail_id].deployed_length
            
            return lengths
        else:
            # Wound state - no action
            return {s: 0.0 for s in self.drums}
    
    def command(self, direction: np.ndarray, magnitude: float = 1.0):
        """
        External command (Yondu whistle style).
        
        Sets a direction for the marionette to fly toward.
        """
        self.external_command = direction * magnitude
        print(f"[SPOOL] Command: direction={direction}, mag={magnitude:.2f}")
    
    def hold(self, reason: str = "human_override"):
        """
        Halt control for human intervention.
        
        Uses cascade HOLD if available.
        """
        if CASCADE_AVAILABLE:
            try:
                cascade.hold(self.spool_id, reason=reason)
            except:
                pass
        
        # Engage all brakes
        for drum in self.drums.values():
            drum.brake_engaged = 1.0
        
        print(f"[SPOOL] HOLD - {reason}")


def create_marionette_system(n_sails: int = 6, 
                             champion_path: Optional[Path] = None) -> MarionetteSpool:
    """
    Create a complete marionette spool system.
    
    The spool is the CENTRAL AUTHORITY - it contains the champion
    and controls all sails through cable tension.
    """
    spool = MarionetteSpool(spool_id="yaka_marionette")
    
    # Add drums for each sail
    for i in range(n_sails):
        spool.add_drum(f"sail_{i}", cable_length=15.0)
    
    # Load champion if available
    if champion_path:
        spool.load_champion(champion_path)
    
    print(f"\n=== MARIONETTE SPOOL SYSTEM ===")
    print(f"Sails: {n_sails}")
    print(f"Cable per sail: 15m (12m max deploy)")
    print(f"Champion: {'LOADED' if spool.champion else 'reactive fallback'}")
    print(f"Cascade substrate: {'ACTIVE' if CASCADE_AVAILABLE else 'not available'}")
    print(f"================================\n")
    
    return spool


if __name__ == "__main__":
    print("\n=== MARIONETTE SPOOL TEST ===\n")
    
    spool = create_marionette_system(n_sails=6)
    
    # Simulate throw
    spool.throw(
        throw_velocity=np.array([0, 20, 15]),
        initial_spin=10.0
    )
    
    # Simulate unraveling
    print("\n--- Unraveling Phase ---")
    for step in range(100):
        dt = 0.02
        
        # Mock sail positions (spreading out)
        deployment = min(step / 50.0, 1.0)
        sail_positions = {}
        sail_forces = {}
        
        for i in range(6):
            angle = 2 * np.pi * i / 6 + spool.spin_phase
            radius = 1.0 + deployment * 11.0  # 1m to 12m
            sail_positions[f"sail_{i}"] = spool.position + np.array([
                radius * np.cos(angle),
                radius * np.sin(angle),
                0
            ])
            # Centrifugal + lift
            sail_forces[f"sail_{i}"] = np.array([0, 0, 50 + deployment * 100])
        
        lengths = spool.step(dt, sail_forces, sail_positions)
        
        if step % 20 == 0:
            print(f"Step {step}: state={spool.state.value}, "
                  f"deployment={spool.deployment_progress:.0%}, "
                  f"alt={spool.position[2]:.1f}m")
    
    # Control phase
    print("\n--- Control Phase ---")
    for step in range(50):
        dt = 0.02
        
        # Command: fly forward
        if step == 10:
            spool.command(np.array([0, 1, 0]), magnitude=0.5)
        
        lengths = spool.step(dt, sail_forces, sail_positions)
        
        if step % 10 == 0:
            obs = spool.sense()
            print(f"Step {step}: collective={spool.collective_command:.3f}, "
                  f"cyclic_amp={spool.cyclic_amplitude:.3f}")
    
    print("\n[SPOOL] Test complete!")
