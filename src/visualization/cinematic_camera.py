"""
Cinematic Camera System for KAPS Visual Trainer
================================================

Dynamic camera perspectives that follow the Buzzard from multiple angles:

CAMERA MODES:
- CHASE: Behind and above, following velocity vector
- ORBIT: Circular orbit around the Buzzard
- CINEMATIC: Smooth transitions between dramatic angles
- COCKPIT: First-person from Buzzard's perspective
- TACTICAL: Top-down overview
- SIDE: Traditional side view
- DYNAMIC: Automatic scene transitions based on action

The camera should always give contextual awareness of:
1. Where the Buzzard is going (velocity direction)
2. Where threats are coming from
3. Where the TAB airfoils are positioned
"""

import numpy as np
from typing import Optional, Tuple, List
from enum import Enum, auto
from dataclasses import dataclass
import time


class CameraMode(Enum):
    """Available camera perspectives."""
    CHASE = auto()       # Behind and above, following velocity
    ORBIT = auto()       # Circular orbit
    CINEMATIC = auto()   # Smooth dramatic transitions
    COCKPIT = auto()     # First-person from Buzzard
    TACTICAL = auto()    # Top-down overview
    SIDE = auto()        # Side view
    DYNAMIC = auto()     # Automatic scene changes
    FREE = auto()        # User-controlled


@dataclass
class CameraState:
    """Current camera state."""
    position: np.ndarray      # World position
    look_at: np.ndarray       # Target point
    up_vector: np.ndarray     # Up direction
    fov: float = 60.0         # Field of view
    
    # For smooth transitions
    target_position: np.ndarray = None
    target_look_at: np.ndarray = None
    transition_speed: float = 3.0
    
    def __post_init__(self):
        if self.target_position is None:
            self.target_position = self.position.copy()
        if self.target_look_at is None:
            self.target_look_at = self.look_at.copy()


class CinematicSequence:
    """A sequence of camera angles for dramatic effect."""
    
    def __init__(self):
        self.shots: List[dict] = []
        self.current_shot = 0
        self.shot_start_time = 0
        
    def add_shot(self, 
                 offset: np.ndarray, 
                 duration: float,
                 fov: float = 60,
                 look_offset: np.ndarray = None):
        """Add a camera shot to the sequence."""
        self.shots.append({
            'offset': offset,
            'duration': duration,
            'fov': fov,
            'look_offset': look_offset or np.zeros(3)
        })
    
    def get_current_shot(self, elapsed: float) -> Optional[dict]:
        """Get current shot based on elapsed time."""
        if not self.shots:
            return None
        
        total = 0
        for i, shot in enumerate(self.shots):
            total += shot['duration']
            if elapsed < total:
                return shot
        
        # Loop back
        return self.shots[0]


class CinematicCamera:
    """
    Dynamic cinematic camera system.
    
    Provides multiple perspectives and smooth transitions.
    """
    
    def __init__(self):
        # Current state
        self.mode = CameraMode.CHASE
        self.state = CameraState(
            position=np.array([0.0, -100.0, 50.0]),
            look_at=np.array([0.0, 0.0, 0.0]),
            up_vector=np.array([0.0, 0.0, 1.0])
        )
        
        # Target tracking
        self.target_position = np.array([0.0, 0.0, 1000.0])  # Buzzard position
        self.target_velocity = np.array([0.0, 50.0, 0.0])     # Buzzard velocity
        
        # Mode-specific settings
        self.chase_distance = 80.0
        self.chase_height = 30.0
        self.chase_lag = 0.1  # Lag behind velocity for dramatic effect
        
        self.orbit_radius = 100.0
        self.orbit_speed = 0.3  # Radians per second
        self.orbit_angle = 0.0
        self.orbit_height = 40.0
        
        self.tactical_height = 300.0
        
        # Cinematic sequences
        self.sequences = self._create_default_sequences()
        self.active_sequence: Optional[CinematicSequence] = None
        self.sequence_start_time = 0.0
        
        # Transition state
        self.in_transition = False
        self.transition_progress = 0.0
        self.transition_duration = 2.0
        self.prev_state: Optional[CameraState] = None
        
        # Dynamic mode timing
        self.dynamic_last_change = time.time()
        self.dynamic_interval = 8.0  # Change every 8 seconds
        self.dynamic_mode_sequence = [
            CameraMode.CHASE,
            CameraMode.ORBIT,
            CameraMode.SIDE,
            CameraMode.TACTICAL,
            CameraMode.CHASE,
        ]
        self.dynamic_mode_index = 0
    
    def _create_default_sequences(self) -> dict:
        """Create default cinematic sequences."""
        sequences = {}
        
        # Dramatic reveal sequence
        reveal = CinematicSequence()
        reveal.add_shot(np.array([0, -150, 20]), 3.0, fov=40)   # Low far approach
        reveal.add_shot(np.array([100, -50, 80]), 2.0, fov=50)  # Sweep right high
        reveal.add_shot(np.array([0, -60, 40]), 2.0, fov=60)    # Settle into chase
        sequences['reveal'] = reveal
        
        # Action sequence - quick cuts
        action = CinematicSequence()
        action.add_shot(np.array([-50, -30, 20]), 1.5, fov=70)  # Close left
        action.add_shot(np.array([50, -30, 20]), 1.5, fov=70)   # Close right
        action.add_shot(np.array([0, 50, 100]), 1.0, fov=50)    # High front
        action.add_shot(np.array([0, -80, 30]), 2.0, fov=60)    # Pull back
        sequences['action'] = action
        
        return sequences
    
    def set_mode(self, mode: CameraMode, transition: bool = True):
        """Change camera mode with optional transition."""
        if mode == self.mode:
            return
        
        if transition:
            self.prev_state = CameraState(
                position=self.state.position.copy(),
                look_at=self.state.look_at.copy(),
                up_vector=self.state.up_vector.copy(),
                fov=self.state.fov
            )
            self.in_transition = True
            self.transition_progress = 0.0
        
        self.mode = mode
        print(f"[CAMERA] Mode: {mode.name}")
    
    def start_sequence(self, name: str):
        """Start a cinematic sequence."""
        if name in self.sequences:
            self.active_sequence = self.sequences[name]
            self.sequence_start_time = time.time()
            print(f"[CAMERA] Starting sequence: {name}")
    
    def stop_sequence(self):
        """Stop active sequence."""
        self.active_sequence = None
    
    def update(self, 
               target_pos: np.ndarray,
               target_vel: np.ndarray,
               dt: float,
               threats: List = None) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Update camera position.
        
        Returns: (position, look_at, fov)
        """
        self.target_position = target_pos
        self.target_velocity = target_vel
        
        # Handle cinematic sequence
        if self.active_sequence:
            elapsed = time.time() - self.sequence_start_time
            shot = self.active_sequence.get_current_shot(elapsed)
            if shot:
                new_pos = target_pos + shot['offset']
                new_look = target_pos + shot['look_offset']
                self.state.target_position = new_pos
                self.state.target_look_at = new_look
                self.state.fov = shot['fov']
        
        # Handle dynamic mode changes
        elif self.mode == CameraMode.DYNAMIC:
            now = time.time()
            if now - self.dynamic_last_change > self.dynamic_interval:
                self.dynamic_last_change = now
                self.dynamic_mode_index = (self.dynamic_mode_index + 1) % len(self.dynamic_mode_sequence)
                next_mode = self.dynamic_mode_sequence[self.dynamic_mode_index]
                # Temporarily switch to compute target, but keep dynamic
                self._compute_mode_position(next_mode, threats)
        
        else:
            # Compute position for current mode
            self._compute_mode_position(self.mode, threats)
        
        # Smooth transition
        if self.in_transition:
            self.transition_progress += dt / self.transition_duration
            if self.transition_progress >= 1.0:
                self.transition_progress = 1.0
                self.in_transition = False
                self.prev_state = None
            
            t = self._ease_in_out(self.transition_progress)
            if self.prev_state:
                self.state.position = self._lerp(self.prev_state.position, 
                                                  self.state.target_position, t)
                self.state.look_at = self._lerp(self.prev_state.look_at,
                                                 self.state.target_look_at, t)
        else:
            # Smooth follow
            alpha = min(1.0, dt * self.state.transition_speed)
            self.state.position = self._lerp(self.state.position,
                                              self.state.target_position, alpha)
            self.state.look_at = self._lerp(self.state.look_at,
                                             self.state.target_look_at, alpha)
        
        return self.state.position, self.state.look_at, self.state.fov
    
    def _compute_mode_position(self, mode: CameraMode, threats: List = None):
        """Compute target camera position for a mode."""
        pos = self.target_position
        vel = self.target_velocity
        
        # Normalize velocity for direction
        speed = np.linalg.norm(vel)
        if speed > 0.1:
            vel_dir = vel / speed
        else:
            vel_dir = np.array([0, 1, 0])  # Default forward
        
        if mode == CameraMode.CHASE:
            # Behind and above, along velocity vector
            behind = -vel_dir * self.chase_distance
            up = np.array([0, 0, self.chase_height])
            self.state.target_position = pos + behind + up
            self.state.target_look_at = pos + vel_dir * 20  # Look ahead
        
        elif mode == CameraMode.ORBIT:
            # Circular orbit around target
            self.orbit_angle += 0.016 * self.orbit_speed  # Assume ~60fps
            x = self.orbit_radius * np.cos(self.orbit_angle)
            y = self.orbit_radius * np.sin(self.orbit_angle)
            self.state.target_position = pos + np.array([x, y, self.orbit_height])
            self.state.target_look_at = pos
        
        elif mode == CameraMode.COCKPIT:
            # First person from Buzzard
            forward = vel_dir * 5  # Just ahead
            self.state.target_position = pos + np.array([0, 0, 2])
            self.state.target_look_at = pos + forward * 50
        
        elif mode == CameraMode.TACTICAL:
            # Top-down
            self.state.target_position = pos + np.array([0, 0, self.tactical_height])
            self.state.target_look_at = pos
        
        elif mode == CameraMode.SIDE:
            # Side view
            right = np.cross(vel_dir, np.array([0, 0, 1]))
            if np.linalg.norm(right) < 0.1:
                right = np.array([1, 0, 0])
            else:
                right = right / np.linalg.norm(right)
            self.state.target_position = pos + right * 80 + np.array([0, 0, 20])
            self.state.target_look_at = pos
        
        elif mode == CameraMode.CINEMATIC:
            # Dramatic low angle
            behind = -vel_dir * 40
            self.state.target_position = pos + behind + np.array([20, 0, -10])
            self.state.target_look_at = pos + vel_dir * 30
        
        elif mode == CameraMode.FREE:
            # Don't update automatically
            pass
    
    def _lerp(self, a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
        """Linear interpolation."""
        return a + (b - a) * t
    
    def _ease_in_out(self, t: float) -> float:
        """Smooth easing function."""
        return t * t * (3 - 2 * t)
    
    # =========================================================================
    # USER INPUT HANDLERS
    # =========================================================================
    
    def handle_orbit_drag(self, dx: float, dy: float):
        """Handle mouse drag for orbit control."""
        if self.mode == CameraMode.FREE:
            # Adjust orbit angle and height
            self.orbit_angle += dx * 0.01
            self.orbit_height = np.clip(self.orbit_height + dy * 2, 10, 200)
    
    def handle_zoom(self, direction: float):
        """Handle zoom in/out."""
        factor = 0.9 if direction > 0 else 1.1
        self.chase_distance = np.clip(self.chase_distance * factor, 30, 300)
        self.orbit_radius = np.clip(self.orbit_radius * factor, 40, 400)


# =============================================================================
# CAMERA MODE HOTKEYS
# =============================================================================

CAMERA_KEYBINDS = {
    'c': 'cycle_mode',        # Cycle through modes
    'v': CameraMode.CHASE,    # Chase cam
    'b': CameraMode.ORBIT,    # Orbit cam
    'n': CameraMode.TACTICAL, # Tactical (top-down)
    'm': CameraMode.DYNAMIC,  # Dynamic cinematic
    ',': CameraMode.COCKPIT,  # First person
}
