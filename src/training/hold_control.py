"""
HOLD Control System for KAPS Visual Trainer
============================================

Integrates the CASCADE-LATTICE Hold system with KAPS training.

HOLD MODES:
-----------
1. CONTINUOUS - Full speed training, no pauses
2. AI_PAUSE   - Environment runs, AI is frozen (reactive mode)
3. FULL_PAUSE - Everything stops, inspect state
4. STEP       - Advance one decision at a time

SHOWCASE MODES:
---------------
Interactive demonstrations of KAPS capabilities.
These are scripted maneuvers that show what the system CAN do.

The DreamerV3 brain is TRAINABLE - these showcases help it
learn what good behavior looks like.
"""

import numpy as np
import time
import threading
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass, field
from enum import Enum, auto


class HoldMode(Enum):
    """Control modes for the simulation."""
    CONTINUOUS = auto()    # Full speed, no intervention
    AI_PAUSE = auto()      # Environment runs, AI frozen - forces reactive control
    FULL_PAUSE = auto()    # Everything frozen
    STEP = auto()          # Single-step mode


class ShowcaseMode(Enum):
    """Showcase demonstration modes."""
    NONE = auto()              # Normal operation
    SYSTEMS_CHECK = auto()     # Test all systems - extend/retract all TABs
    DEFENSIVE_SPHERE = auto()  # Maximum coverage formation
    INTERCEPTION_DRILL = auto() # Practice threat intercepts
    MAX_EXTENSION = auto()     # Push cables to limits
    CORKSCREW_SPIN = auto()    # Test corkscrew propulsion
    FORMATION_PARADE = auto()  # Cycle through formations
    THROTTLE_TEST = auto()     # 0 to max speed demonstration
    EVASIVE_DEMO = auto()      # Aggressive evasive maneuvers
    SACRIFICE_DRILL = auto()   # Practice TAB sacrifice for Buzzard protection
    SLINGSHOT_DEMO = auto()    # Nunchaku/bola mode demonstration


@dataclass
class HoldState:
    """Current state of the HOLD system."""
    mode: HoldMode = HoldMode.CONTINUOUS
    showcase: ShowcaseMode = ShowcaseMode.NONE
    
    # Timing
    paused_at: Optional[float] = None
    pause_duration: float = 0.0
    
    # Speed control
    time_scale: float = 1.0  # 0.1 = 10%, 2.0 = 2x speed
    
    # Step control
    steps_remaining: int = 0  # For STEP mode
    
    # AI state
    ai_frozen: bool = False
    last_ai_action: Optional[np.ndarray] = None
    
    # Showcase progress
    showcase_step: int = 0
    showcase_phase: str = ""


class ShowcaseController:
    """
    Generates showcase actions to demonstrate KAPS capabilities.
    
    These are NOT random - they're designed maneuvers that show
    what the airfoils can do. The DreamerV3 can LEARN from these.
    """
    
    def __init__(self):
        self.step = 0
        self.phase = ""
    
    def get_action(self, mode: ShowcaseMode, obs: np.ndarray) -> np.ndarray:
        """Generate showcase action for current mode."""
        self.step += 1
        
        if mode == ShowcaseMode.SYSTEMS_CHECK:
            return self._systems_check()
        elif mode == ShowcaseMode.DEFENSIVE_SPHERE:
            return self._defensive_sphere()
        elif mode == ShowcaseMode.INTERCEPTION_DRILL:
            return self._interception_drill(obs)
        elif mode == ShowcaseMode.MAX_EXTENSION:
            return self._max_extension()
        elif mode == ShowcaseMode.CORKSCREW_SPIN:
            return self._corkscrew_spin()
        elif mode == ShowcaseMode.FORMATION_PARADE:
            return self._formation_parade()
        elif mode == ShowcaseMode.THROTTLE_TEST:
            return self._throttle_test()
        elif mode == ShowcaseMode.EVASIVE_DEMO:
            return self._evasive_demo()
        elif mode == ShowcaseMode.SACRIFICE_DRILL:
            return self._sacrifice_drill()
        elif mode == ShowcaseMode.SLINGSHOT_DEMO:
            return self._slingshot_demo()
        else:
            return np.zeros(18)
    
    def reset(self):
        """Reset showcase state."""
        self.step = 0
        self.phase = ""
    
    # =========================================================================
    # SHOWCASE MANEUVERS
    # =========================================================================
    
    def _systems_check(self) -> np.ndarray:
        """
        SYSTEMS CHECK: Test all actuators systematically.
        
        Sequence:
        1. Extend all cables to max
        2. Retract all cables
        3. Sweep all yaw controls
        4. Sweep all pitch controls
        5. Roll test
        6. Corkscrew test
        """
        action = np.zeros(18)
        cycle = self.step % 300
        
        if cycle < 50:
            # Phase 1: EXTEND ALL
            self.phase = "EXTEND ALL"
            for i in range(4):
                action[i*4 + 0] = 1.0  # Max extension
        
        elif cycle < 100:
            # Phase 2: RETRACT ALL
            self.phase = "RETRACT ALL"
            for i in range(4):
                action[i*4 + 0] = -1.0  # Max retraction
        
        elif cycle < 150:
            # Phase 3: YAW SWEEP
            self.phase = "YAW SWEEP"
            phase = (cycle - 100) / 50 * 2 * np.pi
            for i in range(4):
                action[i*4 + 0] = 0.5  # Half extension
                action[i*4 + 1] = np.sin(phase + i * np.pi/2)  # Yaw
        
        elif cycle < 200:
            # Phase 4: PITCH SWEEP
            self.phase = "PITCH SWEEP"
            phase = (cycle - 150) / 50 * 2 * np.pi
            for i in range(4):
                action[i*4 + 0] = 0.5
                action[i*4 + 2] = np.sin(phase + i * np.pi/2)  # Pitch
        
        elif cycle < 250:
            # Phase 5: ROLL TEST
            self.phase = "ROLL TEST"
            phase = (cycle - 200) / 50 * 2 * np.pi
            for i in range(4):
                action[i*4 + 0] = 0.6
                action[i*4 + 3] = np.sin(phase)  # All roll together
        
        else:
            # Phase 6: CORKSCREW
            self.phase = "CORKSCREW TEST"
            action[16] = np.sin((cycle - 250) / 50 * 2 * np.pi)
            for i in range(4):
                action[i*4 + 0] = 0.7
        
        return action
    
    def _defensive_sphere(self) -> np.ndarray:
        """
        DEFENSIVE SPHERE: Maximum coverage formation.
        
        TABs spread out to cover all approach angles.
        This is the ideal defensive posture.
        """
        action = np.zeros(18)
        self.phase = "DEFENSIVE SPHERE"
        
        # Each TAB covers its quadrant
        # UP: covers above
        # DOWN: covers below
        # LEFT: covers port
        # RIGHT: covers starboard
        
        positions = [
            (0.8, 0.0, 0.3),   # UP: extended, level, slight roll
            (0.8, 0.0, -0.3),  # DOWN: extended, level, opposite roll
            (0.8, 0.0, 0.0),   # LEFT: extended, level
            (0.8, 0.0, 0.0),   # RIGHT: extended, level
        ]
        
        for i, (ext, yaw, roll) in enumerate(positions):
            action[i*4 + 0] = ext   # Extension
            action[i*4 + 1] = yaw   # Yaw
            action[i*4 + 2] = 0.0   # Pitch
            action[i*4 + 3] = roll  # Roll
        
        # Gentle oscillation to show readiness
        osc = np.sin(self.step * 0.1) * 0.1
        for i in range(4):
            action[i*4 + 2] += osc
        
        return action
    
    def _interception_drill(self, obs: np.ndarray) -> np.ndarray:
        """
        INTERCEPTION DRILL: Practice threat intercepts.
        
        Aggressively steer TABs toward any detected threats.
        This shows what successful defense looks like.
        """
        action = np.zeros(18)
        self.phase = "INTERCEPTION DRILL"
        
        # All TABs extended and hunting
        for i in range(4):
            action[i*4 + 0] = 0.9  # Near-max extension
            
            # Aggressive sweeping to find targets
            sweep = np.sin(self.step * 0.2 + i * np.pi/2)
            action[i*4 + 1] = sweep * 0.7  # Yaw sweep
            action[i*4 + 2] = np.cos(self.step * 0.15 + i * np.pi/2) * 0.5  # Pitch
            action[i*4 + 3] = 0.3  # Roll for agility
        
        return action
    
    def _max_extension(self) -> np.ndarray:
        """
        MAX EXTENSION: Push cables to their limits.
        
        Shows the maximum operational envelope.
        """
        action = np.zeros(18)
        self.phase = "MAX EXTENSION"
        
        # ALL cables to maximum
        for i in range(4):
            action[i*4 + 0] = 1.0  # MAX
            # Gentle positioning to show they're at full reach
            action[i*4 + 1] = np.sin(self.step * 0.05 + i * np.pi/2) * 0.2
            action[i*4 + 2] = np.cos(self.step * 0.03 + i * np.pi/2) * 0.2
        
        return action
    
    def _corkscrew_spin(self) -> np.ndarray:
        """
        CORKSCREW SPIN: Demonstrate corkscrew propulsion.
        
        The mother drone's unique propulsion system.
        """
        action = np.zeros(18)
        cycle = self.step % 200
        
        if cycle < 100:
            # Spin up
            self.phase = "CORKSCREW: SPIN UP"
            action[16] = min(1.0, cycle / 50)  # Ramp up
        else:
            # Spin down
            self.phase = "CORKSCREW: SPIN DOWN"
            action[16] = max(-1.0, 1.0 - (cycle - 100) / 50)
        
        # TABs trail during spin
        for i in range(4):
            action[i*4 + 0] = 0.6
            action[i*4 + 3] = action[16] * 0.3  # Roll with spin
        
        return action
    
    def _formation_parade(self) -> np.ndarray:
        """
        FORMATION PARADE: Cycle through tactical formations.
        
        Shows different defensive configurations.
        """
        action = np.zeros(18)
        formation = (self.step // 100) % 5
        
        formations = {
            0: ("DIAMOND", [(0.7, 0.0), (0.0, 0.7), (-0.7, 0.0), (0.0, -0.7)]),
            1: ("SQUARE", [(0.5, 0.5), (0.5, -0.5), (-0.5, -0.5), (-0.5, 0.5)]),
            2: ("LINE ABREAST", [(0.8, 0.0), (0.4, 0.0), (-0.4, 0.0), (-0.8, 0.0)]),
            3: ("STACK", [(0.0, 0.8), (0.0, 0.4), (0.0, -0.4), (0.0, -0.8)]),
            4: ("SPREAD", [(0.9, 0.3), (0.9, -0.3), (-0.9, 0.3), (-0.9, -0.3)]),
        }
        
        name, positions = formations[formation]
        self.phase = f"FORMATION: {name}"
        
        for i, (yaw, pitch) in enumerate(positions):
            action[i*4 + 0] = 0.7  # Consistent extension
            action[i*4 + 1] = yaw
            action[i*4 + 2] = pitch
        
        return action
    
    def _throttle_test(self) -> np.ndarray:
        """
        THROTTLE TEST: 0 to max speed demonstration.
        
        Smooth acceleration showing system response.
        """
        action = np.zeros(18)
        cycle = self.step % 400
        
        if cycle < 200:
            # Accelerate
            self.phase = f"THROTTLE: {int(cycle/2)}%"
            throttle = cycle / 200
        else:
            # Decelerate
            self.phase = f"THROTTLE: {int(100 - (cycle-200)/2)}%"
            throttle = 1.0 - (cycle - 200) / 200
        
        # Corkscrew controls thrust
        action[16] = throttle
        
        # TABs stream back during acceleration
        for i in range(4):
            action[i*4 + 0] = 0.5 + throttle * 0.3
            action[i*4 + 1] = 0.0
            action[i*4 + 2] = -throttle * 0.3  # Stream back
        
        return action
    
    def _evasive_demo(self) -> np.ndarray:
        """
        EVASIVE DEMO: Aggressive evasive maneuvers.
        
        Shows maximum agility of the system.
        """
        action = np.zeros(18)
        self.phase = "EVASIVE MANEUVERS"
        
        # Rapid, unpredictable movements
        t = self.step * 0.3
        
        for i in range(4):
            # Random-ish but smooth movement
            action[i*4 + 0] = 0.6 + 0.3 * np.sin(t + i * 1.7)
            action[i*4 + 1] = np.sin(t * 1.3 + i * 2.1) * 0.8
            action[i*4 + 2] = np.cos(t * 0.9 + i * 1.3) * 0.6
            action[i*4 + 3] = np.sin(t * 2.1 + i * 0.7) * 0.5
        
        # Corkscrew for evasive thrust
        action[16] = np.sin(t * 0.7) * 0.6
        
        return action
    
    def _sacrifice_drill(self) -> np.ndarray:
        """
        SACRIFICE DRILL: Practice TAB sacrifice for Buzzard protection.
        
        Demonstrates intentional TAB release to intercept threats.
        Cycle through releasing each TAB to show the system understands
        they are EXPENDABLE - the Buzzard is what matters.
        """
        action = np.zeros(18)
        cycle = self.step % 400
        tab_to_sacrifice = (cycle // 100) % 4
        
        self.phase = f"SACRIFICE: TAB {['UP','DOWN','LEFT','RIGHT'][tab_to_sacrifice]}"
        
        for i in range(4):
            if i == tab_to_sacrifice and (cycle % 100) > 70:
                # This TAB releases (sacrifice)
                action[17] = 1.0  # Release signal
                action[i*4 + 0] = 1.0  # Full extension before release
            else:
                # Other TABs maintain defensive posture
                action[i*4 + 0] = 0.7
                action[i*4 + 1] = np.sin(self.step * 0.1 + i) * 0.3
        
        return action
    
    def _slingshot_demo(self) -> np.ndarray:
        """
        SLINGSHOT DEMO: Nunchaku/bola mode demonstration.
        
        Demonstrates the consolidated TAB mode where all airfoils
        merge into a single mass that the Buzzard swings like nunchucks.
        
        Phases:
        1. CONSOLIDATE - TABs converge to form bola
        2. CHARGE - Build angular momentum (spin up)
        3. SWING - Full rotation with max tension
        4. RELEASE - Cut cable at optimal angle
        5. SAIL - Grid fins deploy, steer to target
        6. DISPERSE - Return to standard mode
        """
        action = np.zeros(18)
        
        # Extended action space for slingshot mode:
        # action[18] = consolidate command (0-1)
        # action[19] = swing torque (-1 to 1)
        # action[20] = release command (0-1)
        # action[21] = grid fin deploy (0-1)
        # action[22:26] = fin deflections
        
        # For now, we use the existing action space creatively
        # All TABs move together to simulate consolidation
        
        cycle_length = 600
        phase_time = self.step % cycle_length
        
        if phase_time < 100:
            # PHASE 1: CONSOLIDATE - All TABs converge to center-back
            self.phase = "SLINGSHOT: CONSOLIDATE"
            t = phase_time / 100.0
            
            # All TABs retract and move toward same point
            for i in range(4):
                action[i*4 + 0] = 0.3 - 0.2 * t  # Retract cables
                action[i*4 + 1] = -0.5  # All pitch backward
                action[i*4 + 2] = 0     # Center yaw
            
            # Buzzard slows down for consolidation
            action[16] = 0.2
            
        elif phase_time < 200:
            # PHASE 2: CHARGE - Start spinning up (build angular momentum)
            self.phase = "SLINGSHOT: CHARGE SPIN"
            t = (phase_time - 100) / 100.0
            
            # Coordinated rotation - all TABs act as one mass
            angle = t * np.pi * 2  # One full rotation buildup
            
            for i in range(4):
                action[i*4 + 0] = 0.1  # Tight formation
                action[i*4 + 1] = np.cos(angle) * 0.5
                action[i*4 + 2] = np.sin(angle) * 0.5
            
            # Corkscrew provides the spin torque
            action[16] = 0.8  # High thrust
            
        elif phase_time < 400:
            # PHASE 3: SWING - Full nunchaku rotation, high speed
            self.phase = "SLINGSHOT: SWING (Bruce Lee mode!)"
            t = (phase_time - 200) / 200.0
            
            # Rapid rotation - multiple revolutions
            angle = t * np.pi * 8  # 4 full rotations
            
            # Extend cables for max angular momentum
            extension = 0.1 + t * 0.9  # Gradually extend
            
            for i in range(4):
                action[i*4 + 0] = extension
                # All rotate together as bola head
                action[i*4 + 1] = np.cos(angle + i * 0.01) * 0.8  # Slight offset for visual
                action[i*4 + 2] = np.sin(angle + i * 0.01) * 0.8
            
            # Buzzard provides centripetal force
            action[16] = 0.5 + 0.3 * np.cos(angle * 0.5)
            
        elif phase_time < 450:
            # PHASE 4: RELEASE - Cut at optimal angle
            self.phase = "SLINGSHOT: RELEASE!"
            
            # Max extension at release
            for i in range(4):
                action[i*4 + 0] = 1.0  # Full extension
                action[i*4 + 1] = 0.8  # Forward pitch (release direction)
                action[i*4 + 2] = 0.0  # Straight
            
            # Release command
            action[17] = 1.0
            
        elif phase_time < 550:
            # PHASE 5: SAIL - Grid fins deployed, steering
            self.phase = "SLINGSHOT: SAILING (Grid fins active)"
            t = (phase_time - 450) / 100.0
            
            # Simulate fin steering with airfoil controls
            for i in range(4):
                action[i*4 + 1] = np.sin(t * 2) * 0.3  # Steering adjustments
                action[i*4 + 2] = np.cos(t * 2) * 0.2
                action[i*4 + 3] = 0.5 + np.sin(t * 4) * 0.3  # Fin flutter
            
        else:
            # PHASE 6: DISPERSE - Return to standard mode
            self.phase = "SLINGSHOT: DISPERSE"
            t = (phase_time - 550) / 50.0
            
            # Spread back out to defensive formation
            offsets = [(0, 0.5), (0, -0.5), (-0.5, 0), (0.5, 0)]
            for i, (pitch_off, yaw_off) in enumerate(offsets):
                action[i*4 + 0] = 0.3 + 0.3 * t  # Extend
                action[i*4 + 1] = pitch_off * t
                action[i*4 + 2] = yaw_off * t
        
        return action


class KAPSHoldController:
    """
    Main HOLD controller for KAPS training.
    
    Manages:
    - Pause/resume
    - Showcase modes
    - Speed control
    - AI freeze
    """
    
    def __init__(self):
        self.state = HoldState()
        self.showcase = ShowcaseController()
        self._lock = threading.Lock()
        
        # Callbacks for UI updates
        self._mode_callbacks: List[Callable[[HoldMode], None]] = []
        self._showcase_callbacks: List[Callable[[ShowcaseMode, str], None]] = []
    
    # =========================================================================
    # MODE CONTROL
    # =========================================================================
    
    def set_mode(self, mode: HoldMode):
        """Change the hold mode."""
        with self._lock:
            old_mode = self.state.mode
            self.state.mode = mode
            
            if mode == HoldMode.FULL_PAUSE:
                self.state.paused_at = time.time()
            elif mode == HoldMode.AI_PAUSE:
                self.state.ai_frozen = True
            else:
                self.state.ai_frozen = False
                self.state.paused_at = None
            
            print(f"[HOLD] Mode: {old_mode.name} → {mode.name}")
        
        for cb in self._mode_callbacks:
            cb(mode)
    
    def toggle_pause(self):
        """Toggle between CONTINUOUS and FULL_PAUSE."""
        if self.state.mode == HoldMode.FULL_PAUSE:
            self.set_mode(HoldMode.CONTINUOUS)
        else:
            self.set_mode(HoldMode.FULL_PAUSE)
    
    def step_once(self):
        """Advance one step when in STEP mode."""
        with self._lock:
            self.state.steps_remaining = 1
            if self.state.mode != HoldMode.STEP:
                self.state.mode = HoldMode.STEP
    
    # =========================================================================
    # SHOWCASE CONTROL
    # =========================================================================
    
    def set_showcase(self, mode: ShowcaseMode):
        """Start a showcase demonstration."""
        with self._lock:
            self.state.showcase = mode
            self.showcase.reset()
            print(f"[SHOWCASE] Starting: {mode.name}")
        
        for cb in self._showcase_callbacks:
            cb(mode, "")
    
    def stop_showcase(self):
        """Stop current showcase, return to normal."""
        self.set_showcase(ShowcaseMode.NONE)
    
    # =========================================================================
    # SPEED CONTROL
    # =========================================================================
    
    def set_speed(self, scale: float):
        """Set time scale (0.1 = 10% speed, 2.0 = 2x speed)."""
        with self._lock:
            self.state.time_scale = max(0.1, min(5.0, scale))
            print(f"[SPEED] {self.state.time_scale:.1f}x")
    
    def speed_up(self):
        """Increase speed."""
        self.set_speed(self.state.time_scale * 1.5)
    
    def slow_down(self):
        """Decrease speed."""
        self.set_speed(self.state.time_scale / 1.5)
    
    # =========================================================================
    # ACTION GENERATION
    # =========================================================================
    
    def should_step(self) -> bool:
        """Check if simulation should advance."""
        if self.state.mode == HoldMode.FULL_PAUSE:
            return False
        if self.state.mode == HoldMode.STEP:
            if self.state.steps_remaining > 0:
                self.state.steps_remaining -= 1
                return True
            return False
        return True
    
    def should_ai_act(self) -> bool:
        """Check if AI should generate new action."""
        if self.state.mode == HoldMode.AI_PAUSE:
            return False  # Environment runs, AI frozen
        return self.should_step()
    
    def get_action(self, obs: np.ndarray, ai_action: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Get the action to use.
        
        Priority:
        1. Showcase action (if showcase active)
        2. AI action (if AI not frozen)
        3. Last AI action (if AI frozen)
        4. Zero action
        """
        with self._lock:
            if self.state.showcase != ShowcaseMode.NONE:
                # Showcase takes priority
                action = self.showcase.get_action(self.state.showcase, obs)
                self.state.showcase_phase = self.showcase.phase
            elif ai_action is not None and not self.state.ai_frozen:
                # Normal AI action
                action = ai_action
                self.state.last_ai_action = ai_action.copy()
            elif self.state.last_ai_action is not None:
                # AI frozen, use last action
                action = self.state.last_ai_action
            else:
                action = np.zeros(18)
            
            return action
    
    # =========================================================================
    # UI REGISTRATION
    # =========================================================================
    
    def on_mode_change(self, callback: Callable[[HoldMode], None]):
        """Register callback for mode changes."""
        self._mode_callbacks.append(callback)
    
    def on_showcase_update(self, callback: Callable[[ShowcaseMode, str], None]):
        """Register callback for showcase updates."""
        self._showcase_callbacks.append(callback)


# =============================================================================
# GUI KEY BINDINGS
# =============================================================================

# These will be used by the Panda3D visual trainer
HOLD_KEYBINDS = {
    # Mode control
    'space': 'toggle_pause',      # Pause/Resume
    'f': 'toggle_ai_freeze',      # Freeze AI only
    '.': 'step_once',             # Step one frame
    
    # Speed control
    '=': 'speed_up',              # Faster
    '-': 'slow_down',             # Slower
    '0': 'reset_speed',           # 1x speed
    
    # Showcases (number keys)
    '1': ShowcaseMode.SYSTEMS_CHECK,
    '2': ShowcaseMode.DEFENSIVE_SPHERE,
    '3': ShowcaseMode.INTERCEPTION_DRILL,
    '4': ShowcaseMode.MAX_EXTENSION,
    '5': ShowcaseMode.CORKSCREW_SPIN,
    '6': ShowcaseMode.FORMATION_PARADE,
    '7': ShowcaseMode.THROTTLE_TEST,
    '8': ShowcaseMode.EVASIVE_DEMO,
    '9': ShowcaseMode.SACRIFICE_DRILL,
    
    # Cancel
    'escape': 'stop_showcase',
    'backspace': 'stop_showcase',
}
