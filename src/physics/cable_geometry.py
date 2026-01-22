"""
Cable Geometry & Intersection Detection
========================================
Handles the REAL physics of tether geometry:
- Each TAB has a defined operational SECTOR
- Cables cannot cross without consequences  
- Intersection detection using line-line distance
- Tangling physics when cables touch
- Forced release or system drag when tangled

The 4 TABs are in cross formation:
    
           UP (z+)
            |
            |
    LEFT ---+--- RIGHT (y axis)
    (y-)    |
            |
          DOWN (z-)
            
    ^ Forward (x axis - direction of flight)

Each TAB operates in a WEDGE-SHAPED SECTOR:
- UP: angles 45° to 135° from vertical (upper hemisphere)
- DOWN: angles -45° to -135° (lower hemisphere)  
- LEFT: angles 135° to 225° (left hemisphere)
- RIGHT: angles -45° to 45° (right hemisphere)

Cable intersection occurs when line segments cross in 3D space.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Set
from enum import Enum
import itertools


class TangleState(Enum):
    """Cable tangling states"""
    CLEAR = "clear"           # No issues
    PROXIMITY = "proximity"   # Getting close - warning
    CROSSED = "crossed"       # Cables have crossed
    TANGLED = "tangled"       # Wrapped around each other
    LOCKED = "locked"         # Cannot be separated without release


@dataclass
class CableGeometry:
    """
    Represents a single cable from mother drone to TAB.
    
    The cable is modeled as a line segment from anchor to TAB position,
    with some sag based on tension.
    """
    tab_id: str
    anchor_point: np.ndarray      # Point on mother drone
    tab_position: np.ndarray      # Current TAB position
    cable_length: float           # Maximum cable length
    current_tension: float = 0.0  # Current tension (N)
    
    @property
    def direction(self) -> np.ndarray:
        """Unit vector from anchor to TAB"""
        vec = self.tab_position - self.anchor_point
        norm = np.linalg.norm(vec)
        if norm < 1e-6:
            return np.array([1.0, 0.0, 0.0])
        return vec / norm
    
    @property
    def extension(self) -> float:
        """Current cable extension (may be less than max)"""
        return np.linalg.norm(self.tab_position - self.anchor_point)
    
    @property
    def slack(self) -> float:
        """How much slack in the cable (length - extension)"""
        return max(0, self.cable_length - self.extension)
    
    def get_midpoint(self) -> np.ndarray:
        """Get cable midpoint (for intersection checks)"""
        return (self.anchor_point + self.tab_position) / 2
    
    def sample_points(self, n: int = 10) -> np.ndarray:
        """
        Sample points along the cable for intersection checking.
        
        Includes simple catenary sag model when cable has slack.
        """
        t = np.linspace(0, 1, n)
        points = np.zeros((n, 3))
        
        for i, ti in enumerate(t):
            # Linear interpolation
            p = self.anchor_point + ti * (self.tab_position - self.anchor_point)
            
            # Add sag based on slack and position
            # Maximum sag at midpoint
            if self.slack > 0.1:
                sag_factor = 4 * ti * (1 - ti)  # Parabolic
                sag_amount = self.slack * 0.3 * sag_factor
                p[2] -= sag_amount  # Sag downward
            
            points[i] = p
        
        return points


@dataclass 
class OperationalSector:
    """
    Defines the allowed operational wedge for a TAB.
    
    The sector is defined by angle ranges in the YZ plane
    (perpendicular to flight direction).
    
    Angle 0 = +Y (right)
    Angle 90 = +Z (up)
    Angle 180/-180 = -Y (left)
    Angle -90 = -Z (down)
    """
    tab_id: str
    angle_min: float  # Radians
    angle_max: float  # Radians
    radial_min: float = 5.0   # Minimum distance from center (m)
    radial_max: float = 35.0  # Maximum (cable length)
    
    def contains_angle(self, angle: float) -> bool:
        """Check if angle is within sector"""
        # Normalize angle to [-pi, pi]
        angle = np.arctan2(np.sin(angle), np.cos(angle))
        
        # Handle wrap-around
        if self.angle_min <= self.angle_max:
            return self.angle_min <= angle <= self.angle_max
        else:
            # Sector crosses -pi/pi boundary
            return angle >= self.angle_min or angle <= self.angle_max
    
    def contains_point(self, point_yz: np.ndarray) -> bool:
        """Check if a YZ point is within the sector"""
        y, z = point_yz[0], point_yz[1]
        r = np.sqrt(y**2 + z**2)
        angle = np.arctan2(z, y)
        
        if r < self.radial_min or r > self.radial_max:
            return False
        
        return self.contains_angle(angle)
    
    def clamp_to_sector(self, point_yz: np.ndarray) -> np.ndarray:
        """Project a point to the nearest valid sector location"""
        y, z = point_yz[0], point_yz[1]
        r = np.sqrt(y**2 + z**2)
        angle = np.arctan2(z, y)
        
        # Clamp radius
        r = np.clip(r, self.radial_min, self.radial_max)
        
        # Clamp angle to sector
        if not self.contains_angle(angle):
            # Find nearest sector boundary
            d_min = self._angle_distance(angle, self.angle_min)
            d_max = self._angle_distance(angle, self.angle_max)
            angle = self.angle_min if d_min < d_max else self.angle_max
        
        return np.array([r * np.cos(angle), r * np.sin(angle)])
    
    def _angle_distance(self, a1: float, a2: float) -> float:
        """Compute minimum angular distance"""
        diff = a2 - a1
        return abs(np.arctan2(np.sin(diff), np.cos(diff)))


# Define the operational sectors for each TAB position
# Sectors have a small gap between them to prevent intersection

OPERATIONAL_SECTORS = {
    "UP": OperationalSector(
        tab_id="UP",
        angle_min=np.radians(50),    # 50° from +Y
        angle_max=np.radians(130),   # 130° from +Y
    ),
    "DOWN": OperationalSector(
        tab_id="DOWN", 
        angle_min=np.radians(-130),  # -130° from +Y
        angle_max=np.radians(-50),   # -50° from +Y
    ),
    "LEFT": OperationalSector(
        tab_id="LEFT",
        angle_min=np.radians(140),   # 140° from +Y (into left side)
        angle_max=np.radians(-140),  # Wraps around -180
    ),
    "RIGHT": OperationalSector(
        tab_id="RIGHT",
        angle_min=np.radians(-40),   # -40° from +Y
        angle_max=np.radians(40),    # 40° from +Y  
    ),
}


class CableIntersectionDetector:
    """
    Detects and manages cable intersection geometry.
    
    Key responsibilities:
    1. Track all cable geometries
    2. Detect when cables cross
    3. Compute intersection severity
    4. Determine consequences (warning, drag, forced release)
    """
    
    # Minimum distance between cables before intersection warning
    PROXIMITY_THRESHOLD = 2.0  # meters
    INTERSECTION_THRESHOLD = 0.5  # meters - cables touching
    
    def __init__(self, cable_length: float = 30.0):
        self.cable_length = cable_length
        self.cables: Dict[str, CableGeometry] = {}
        self.tangle_states: Dict[Tuple[str, str], TangleState] = {}
        self.sectors = OPERATIONAL_SECTORS.copy()
        
        # Track intersection history for tangling
        self.intersection_history: Dict[Tuple[str, str], List[float]] = {}
    
    def update_cable(self, 
                     tab_id: str,
                     mother_pos: np.ndarray,
                     tab_pos: np.ndarray,
                     tension: float = 0.0):
        """Update cable geometry for a TAB"""
        # Anchor points on mother drone (offset from center)
        anchor_offsets = {
            "UP": np.array([0, 0, 2]),
            "DOWN": np.array([0, 0, -2]),
            "LEFT": np.array([0, -3, 0]),
            "RIGHT": np.array([0, 3, 0]),
        }
        
        anchor = mother_pos + anchor_offsets.get(tab_id, np.zeros(3))
        
        self.cables[tab_id] = CableGeometry(
            tab_id=tab_id,
            anchor_point=anchor,
            tab_position=tab_pos,
            cable_length=self.cable_length,
            current_tension=tension
        )
    
    def check_all_intersections(self) -> Dict[Tuple[str, str], float]:
        """
        Check for intersections between all cable pairs.
        
        Returns dict of (tab1, tab2) -> minimum distance
        """
        distances = {}
        
        cable_ids = list(self.cables.keys())
        for i, id1 in enumerate(cable_ids):
            for id2 in cable_ids[i+1:]:
                dist = self._compute_cable_distance(id1, id2)
                distances[(id1, id2)] = dist
                
                # Update tangle state
                self._update_tangle_state(id1, id2, dist)
        
        return distances
    
    def _compute_cable_distance(self, id1: str, id2: str) -> float:
        """
        Compute minimum distance between two cables.
        
        Uses line segment to line segment distance.
        """
        cable1 = self.cables[id1]
        cable2 = self.cables[id2]
        
        # Get endpoints
        p1, p2 = cable1.anchor_point, cable1.tab_position
        p3, p4 = cable2.anchor_point, cable2.tab_position
        
        return self._segment_segment_distance(p1, p2, p3, p4)
    
    def _segment_segment_distance(self,
                                   p1: np.ndarray, p2: np.ndarray,
                                   p3: np.ndarray, p4: np.ndarray) -> float:
        """
        Compute minimum distance between two line segments.
        
        Uses the algorithm from:
        "Distance between Lines and Segments with their Closest Point of Approach"
        """
        d1 = p2 - p1  # Direction of segment 1
        d2 = p4 - p3  # Direction of segment 2
        r = p1 - p3
        
        a = np.dot(d1, d1)
        e = np.dot(d2, d2)
        f = np.dot(d2, r)
        
        # Check if both segments are points
        if a < 1e-8 and e < 1e-8:
            return np.linalg.norm(p1 - p3)
        
        if a < 1e-8:
            # Segment 1 is a point
            s = 0.0
            t = np.clip(f / e, 0, 1)
        else:
            c = np.dot(d1, r)
            if e < 1e-8:
                # Segment 2 is a point
                t = 0.0
                s = np.clip(-c / a, 0, 1)
            else:
                b = np.dot(d1, d2)
                denom = a * e - b * b
                
                if abs(denom) > 1e-8:
                    s = np.clip((b * f - c * e) / denom, 0, 1)
                else:
                    s = 0.0
                
                t = (b * s + f) / e
                
                if t < 0:
                    t = 0
                    s = np.clip(-c / a, 0, 1)
                elif t > 1:
                    t = 1
                    s = np.clip((b - c) / a, 0, 1)
        
        closest1 = p1 + s * d1
        closest2 = p3 + t * d2
        
        return np.linalg.norm(closest1 - closest2)
    
    def _update_tangle_state(self, id1: str, id2: str, distance: float):
        """Update the tangle state between two cables"""
        key = tuple(sorted([id1, id2]))
        
        current = self.tangle_states.get(key, TangleState.CLEAR)
        
        if distance > self.PROXIMITY_THRESHOLD:
            new_state = TangleState.CLEAR
        elif distance > self.INTERSECTION_THRESHOLD:
            new_state = TangleState.PROXIMITY
        else:
            # Cables are intersecting
            if current in (TangleState.CLEAR, TangleState.PROXIMITY):
                new_state = TangleState.CROSSED
            elif current == TangleState.CROSSED:
                # Track how long they've been crossed
                history = self.intersection_history.get(key, [])
                history.append(distance)
                self.intersection_history[key] = history
                
                if len(history) > 10:  # Crossed for multiple updates
                    new_state = TangleState.TANGLED
                else:
                    new_state = TangleState.CROSSED
            elif current == TangleState.TANGLED:
                new_state = TangleState.LOCKED
            else:
                new_state = current
        
        # Clear history if cables separate
        if new_state == TangleState.CLEAR:
            if key in self.intersection_history:
                del self.intersection_history[key]
        
        self.tangle_states[key] = new_state
    
    def get_tangle_state(self, id1: str, id2: str) -> TangleState:
        """Get current tangle state between two cables"""
        key = tuple(sorted([id1, id2]))
        return self.tangle_states.get(key, TangleState.CLEAR)
    
    def get_tangled_pairs(self) -> List[Tuple[str, str]]:
        """Get list of cable pairs that are tangled or locked"""
        return [
            pair for pair, state in self.tangle_states.items()
            if state in (TangleState.TANGLED, TangleState.LOCKED)
        ]
    
    def get_crossed_pairs(self) -> List[Tuple[str, str]]:
        """Get list of cable pairs that are currently crossed"""
        return [
            pair for pair, state in self.tangle_states.items()
            if state in (TangleState.CROSSED, TangleState.TANGLED, TangleState.LOCKED)
        ]
    
    def is_in_sector(self, tab_id: str, position_yz: np.ndarray) -> bool:
        """Check if TAB position is within its allowed sector"""
        if tab_id not in self.sectors:
            return True
        return self.sectors[tab_id].contains_point(position_yz)
    
    def clamp_to_sector(self, tab_id: str, position_yz: np.ndarray) -> np.ndarray:
        """Clamp position to valid sector"""
        if tab_id not in self.sectors:
            return position_yz
        return self.sectors[tab_id].clamp_to_sector(position_yz)
    
    def compute_drag_penalty(self) -> float:
        """
        Compute additional drag from cable tangling.
        
        Tangled cables create turbulence and drag.
        """
        penalty = 0.0
        
        for pair, state in self.tangle_states.items():
            if state == TangleState.CROSSED:
                penalty += 0.1  # Minor drag increase
            elif state == TangleState.TANGLED:
                penalty += 0.5  # Significant drag
            elif state == TangleState.LOCKED:
                penalty += 1.0  # Major drag, system unstable
        
        return penalty
    
    def get_forced_releases(self) -> List[str]:
        """
        Get TABs that must be released due to locked cables.
        
        When cables are LOCKED, one must be released to prevent
        catastrophic drag on the mother drone.
        """
        locked_pairs = [
            pair for pair, state in self.tangle_states.items()
            if state == TangleState.LOCKED
        ]
        
        if not locked_pairs:
            return []
        
        # Release the TABs involved in locks
        # Strategy: release the one with lower tension (less load)
        releases = set()
        for id1, id2 in locked_pairs:
            c1 = self.cables.get(id1)
            c2 = self.cables.get(id2)
            
            if c1 and c2:
                if c1.current_tension <= c2.current_tension:
                    releases.add(id1)
                else:
                    releases.add(id2)
            elif c1:
                releases.add(id1)
            elif c2:
                releases.add(id2)
        
        return list(releases)


class SectorConstrainedActionSpace:
    """
    Constrains TAB actions to valid sectors.
    
    This ensures the agent can NEVER request an action that would
    cause cable intersection - the impossible maneuvers are removed
    from the action space entirely.
    """
    
    def __init__(self, cable_length: float = 30.0):
        self.cable_length = cable_length
        self.sectors = OPERATIONAL_SECTORS.copy()
        
        # Build action masks per TAB
        self.action_masks = self._compute_action_masks()
    
    def _compute_action_masks(self) -> Dict[str, np.ndarray]:
        """
        Precompute valid action regions for each TAB.
        
        Actions are (elevator, rudder) which translate to
        (pitch, yaw) which determine YZ position.
        """
        masks = {}
        
        # Discretize action space for masking
        # Elevator affects Z, Rudder affects Y
        for tab_id, sector in self.sectors.items():
            # Create mask over discretized YZ positions
            resolution = 20
            y_range = np.linspace(-self.cable_length, self.cable_length, resolution)
            z_range = np.linspace(-self.cable_length, self.cable_length, resolution)
            
            mask = np.zeros((resolution, resolution), dtype=bool)
            
            for i, y in enumerate(y_range):
                for j, z in enumerate(z_range):
                    if sector.contains_point(np.array([y, z])):
                        mask[i, j] = True
            
            masks[tab_id] = mask
        
        return masks
    
    def constrain_action(self, 
                         tab_id: str, 
                         action_yz: np.ndarray,
                         current_pos_yz: np.ndarray) -> np.ndarray:
        """
        Constrain an action to stay within the TAB's valid sector.
        
        Args:
            tab_id: Which TAB
            action_yz: Requested action in YZ plane (relative motion)
            current_pos_yz: Current YZ position
            
        Returns:
            Constrained action that won't leave sector
        """
        if tab_id not in self.sectors:
            return action_yz
        
        sector = self.sectors[tab_id]
        
        # Compute target position
        target_yz = current_pos_yz + action_yz
        
        # If target is valid, allow it
        if sector.contains_point(target_yz):
            return action_yz
        
        # Otherwise, clamp to sector boundary
        clamped = sector.clamp_to_sector(target_yz)
        return clamped - current_pos_yz
    
    def get_valid_action_range(self, 
                                tab_id: str,
                                current_pos_yz: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the valid action range for a TAB at current position.
        
        Returns (min_action, max_action) bounds.
        """
        if tab_id not in self.sectors:
            return (np.array([-10, -10]), np.array([10, 10]))
        
        sector = self.sectors[tab_id]
        
        # Sample boundary and find limits
        angles = np.linspace(sector.angle_min, sector.angle_max, 50)
        
        boundary_points = []
        for angle in angles:
            for r in [sector.radial_min, sector.radial_max]:
                y = r * np.cos(angle)
                z = r * np.sin(angle)
                boundary_points.append(np.array([y, z]))
        
        boundary = np.array(boundary_points)
        
        # Compute action bounds (target - current)
        deltas = boundary - current_pos_yz
        
        return deltas.min(axis=0), deltas.max(axis=0)


def get_sector_boundaries_for_viz() -> Dict[str, List[np.ndarray]]:
    """
    Get sector boundary lines for visualization.
    
    Returns dict of tab_id -> list of line points
    """
    viz_data = {}
    
    for tab_id, sector in OPERATIONAL_SECTORS.items():
        points = []
        
        # Inner arc
        angles = np.linspace(sector.angle_min, sector.angle_max, 30)
        for angle in angles:
            y = sector.radial_min * np.cos(angle)
            z = sector.radial_min * np.sin(angle)
            points.append(np.array([0, y, z]))  # At x=0 (cross-section)
        
        # Outer arc (reverse for closed polygon)
        for angle in reversed(angles):
            y = sector.radial_max * np.cos(angle)
            z = sector.radial_max * np.sin(angle)
            points.append(np.array([0, y, z]))
        
        # Close the shape
        if points:
            points.append(points[0])
        
        viz_data[tab_id] = points
    
    return viz_data
