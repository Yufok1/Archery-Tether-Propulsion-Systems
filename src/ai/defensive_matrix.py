"""
Defensive Matrix AI Controller
==============================
The 360° threat detection and intercept coordination system.

This is the "brain" of the KAPS defensive system:
- Threat detection and tracking
- Intercept trajectory calculation
- TAB selection for optimal intercept
- Sacrifice decision making
- Evasion coordination
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from enum import Enum
import heapq


class ThreatType(Enum):
    """Classification of incoming threats"""
    MISSILE_IR = "ir_missile"           # Heat-seeking missile
    MISSILE_RADAR = "radar_missile"     # Radar-guided missile
    DRONE_SUICIDE = "suicide_drone"     # FPV/Suicide drone
    PROJECTILE = "projectile"           # Unguided projectile
    UNKNOWN = "unknown"


class ThreatPriority(Enum):
    """Threat priority levels"""
    CRITICAL = 1      # Immediate intercept required
    HIGH = 2          # Intercept within seconds
    MEDIUM = 3        # Track and prepare
    LOW = 4           # Monitor only


@dataclass
class TrackedThreat:
    """A tracked threat in the defensive matrix"""
    threat_id: str
    threat_type: ThreatType
    priority: ThreatPriority
    position: np.ndarray           # Current position (m)
    velocity: np.ndarray           # Current velocity (m/s)
    acceleration: np.ndarray       # Estimated acceleration (m/s²)
    time_to_impact: float          # Estimated seconds to impact
    confidence: float              # Tracking confidence (0-1)
    assigned_tab: Optional[str]    # TAB assigned for intercept
    
    def predict_position(self, t: float) -> np.ndarray:
        """Predict threat position at time t seconds from now"""
        return (self.position + 
                self.velocity * t + 
                0.5 * self.acceleration * t**2)


@dataclass  
class DefenseConfig:
    """Configuration for defensive matrix"""
    detection_radius: float = 500.0      # m
    tracking_update_rate: float = 100.0  # Hz
    intercept_lead_time: float = 0.5     # s
    sacrifice_threshold: float = 0.85    # confidence
    min_intercept_angle: float = 15.0    # degrees


class DefensiveMatrixAI:
    """
    The AI controller for the 360° defensive bubble.
    
    Responsibilities:
    - Track multiple incoming threats
    - Calculate geometric intercept solutions
    - Assign TABs to threats based on position/capability
    - Execute intercept maneuvers (slingshot or direct release)
    - Coordinate evasive maneuvers for mother drone
    """
    
    def __init__(self, config: Optional[DefenseConfig] = None):
        self.config = config or DefenseConfig()
        
        # Threat tracking
        self.tracked_threats: Dict[str, TrackedThreat] = {}
        self._threat_counter = 0
        
        # TAB assignments
        self.tab_assignments: Dict[str, str] = {}  # threat_id -> tab_id
        
        # State
        self.alert_level = "GREEN"
        self.intercepts_executed = 0
        self.threats_neutralized = 0
        
    def detect_threat(self,
                      position: np.ndarray,
                      velocity: np.ndarray,
                      threat_type: ThreatType = ThreatType.UNKNOWN,
                      mother_position: np.ndarray = None) -> Optional[str]:
        """
        Register a new threat detection.
        
        Args:
            position: Threat position in world coords
            velocity: Threat velocity vector
            threat_type: Classification of threat
            mother_position: Current mother drone position
            
        Returns:
            threat_id if registered, None if outside detection range
        """
        if mother_position is None:
            mother_position = np.zeros(3)
        
        # Check if within detection radius
        distance = np.linalg.norm(position - mother_position)
        if distance > self.config.detection_radius:
            return None
        
        # Generate threat ID
        self._threat_counter += 1
        threat_id = f"THREAT-{self._threat_counter:04d}"
        
        # Calculate time to impact (assuming straight-line approach)
        closing_velocity = -np.dot(velocity, (position - mother_position) / distance)
        if closing_velocity > 0:
            tti = distance / closing_velocity
        else:
            tti = float('inf')  # Moving away
        
        # Assign priority based on TTI
        if tti < 2.0:
            priority = ThreatPriority.CRITICAL
            self.alert_level = "RED"
        elif tti < 5.0:
            priority = ThreatPriority.HIGH
            if self.alert_level != "RED":
                self.alert_level = "ORANGE"
        elif tti < 15.0:
            priority = ThreatPriority.MEDIUM
            if self.alert_level == "GREEN":
                self.alert_level = "YELLOW"
        else:
            priority = ThreatPriority.LOW
        
        # Create threat record
        threat = TrackedThreat(
            threat_id=threat_id,
            threat_type=threat_type,
            priority=priority,
            position=position.copy(),
            velocity=velocity.copy(),
            acceleration=np.zeros(3),
            time_to_impact=tti,
            confidence=0.8,
            assigned_tab=None
        )
        
        self.tracked_threats[threat_id] = threat
        
        return threat_id
    
    def update_threat(self, 
                      threat_id: str, 
                      new_position: np.ndarray,
                      new_velocity: np.ndarray,
                      dt: float):
        """Update threat tracking with new observation"""
        if threat_id not in self.tracked_threats:
            return
        
        threat = self.tracked_threats[threat_id]
        
        # Estimate acceleration from velocity change
        if dt > 0:
            threat.acceleration = (new_velocity - threat.velocity) / dt
        
        threat.position = new_position.copy()
        threat.velocity = new_velocity.copy()
        
        # Increase confidence with consistent tracking
        threat.confidence = min(1.0, threat.confidence + 0.05)
    
    def calculate_intercept_solution(self,
                                      threat: TrackedThreat,
                                      tab_position: np.ndarray,
                                      tab_velocity: np.ndarray,
                                      cable_length: float) -> Dict:
        """
        Calculate the geometric intercept solution for a TAB.
        
        This is the core "Archer" calculation - finding where the
        TAB's trajectory will intersect the threat's trajectory.
        
        Returns:
            Solution dict with intercept point, time, and required release angle
        """
        # Relative position and velocity
        rel_pos = threat.position - tab_position
        rel_vel = threat.velocity - tab_velocity
        
        # Quadratic intercept equation
        # |P + V*t|² = (v_intercept * t)²
        # where v_intercept is TAB's speed after release
        
        tab_speed = np.linalg.norm(tab_velocity)
        intercept_speed = tab_speed * 1.3  # Assume 30% boost from slingshot
        
        a = np.dot(rel_vel, rel_vel) - intercept_speed**2
        b = 2 * np.dot(rel_pos, rel_vel)
        c = np.dot(rel_pos, rel_pos)
        
        discriminant = b**2 - 4*a*c
        
        if discriminant < 0:
            return {'solution_exists': False, 'reason': 'No intercept trajectory'}
        
        # Take the positive, smaller root
        t1 = (-b - np.sqrt(discriminant)) / (2*a)
        t2 = (-b + np.sqrt(discriminant)) / (2*a)
        
        t_intercept = min(t for t in [t1, t2] if t > 0) if any(t > 0 for t in [t1, t2]) else None
        
        if t_intercept is None:
            return {'solution_exists': False, 'reason': 'Intercept time negative'}
        
        # Intercept point
        intercept_point = threat.predict_position(t_intercept)
        
        # Required launch direction
        launch_vector = intercept_point - tab_position
        launch_direction = launch_vector / (np.linalg.norm(launch_vector) + 1e-9)
        
        # Calculate required slingshot release angle
        # The TAB needs to be tangent to this direction at release
        current_direction = tab_velocity / (tab_speed + 1e-9)
        release_angle = np.arccos(np.clip(np.dot(current_direction, launch_direction), -1, 1))
        
        return {
            'solution_exists': True,
            'intercept_time': t_intercept,
            'intercept_point': intercept_point,
            'launch_direction': launch_direction,
            'release_angle': np.degrees(release_angle),
            'intercept_speed': intercept_speed,
            'miss_margin': cable_length * 0.1  # 10% of cable length tolerance
        }
    
    def assign_tab_to_threat(self,
                             threat_id: str,
                             available_tabs: Dict[str, Dict],
                             cable_length: float) -> Optional[str]:
        """
        Assign the best available TAB to intercept a threat.
        
        Selection criteria:
        1. Best geometric intercept solution
        2. TAB not already assigned
        3. Formation position alignment with threat direction
        """
        if threat_id not in self.tracked_threats:
            return None
        
        threat = self.tracked_threats[threat_id]
        
        best_tab = None
        best_score = float('inf')
        best_solution = None
        
        for tab_id, tab_info in available_tabs.items():
            # Skip already assigned TABs
            if tab_id in self.tab_assignments.values():
                continue
            
            # Skip detached TABs
            if not tab_info.get('attached', True):
                continue
            
            # Calculate intercept solution
            solution = self.calculate_intercept_solution(
                threat,
                tab_info['position'],
                tab_info['velocity'],
                cable_length
            )
            
            if not solution['solution_exists']:
                continue
            
            # Score: lower is better (prefer quick intercepts with good angles)
            score = (solution['intercept_time'] * 
                    (1 + solution['release_angle'] / 90))
            
            if score < best_score:
                best_score = score
                best_tab = tab_id
                best_solution = solution
        
        if best_tab:
            self.tab_assignments[threat_id] = best_tab
            threat.assigned_tab = best_tab
            
            return best_tab
        
        return None
    
    def calculate_defensive_response(self,
                                      mother_position: np.ndarray,
                                      mother_velocity: np.ndarray,
                                      tab_states: Dict) -> Dict:
        """
        Calculate the overall defensive response for current threats.
        
        Returns commands for:
        - Mother drone evasion
        - TAB release sequences
        - Formation adjustments
        """
        response = {
            'evasion_vector': np.zeros(3),
            'release_commands': [],
            'formation_adjustments': {},
            'threat_summary': [],
            'alert_level': self.alert_level
        }
        
        if not self.tracked_threats:
            self.alert_level = "GREEN"
            response['alert_level'] = "GREEN"
            return response
        
        # Sort threats by priority
        sorted_threats = sorted(
            self.tracked_threats.values(),
            key=lambda t: (t.priority.value, t.time_to_impact)
        )
        
        for threat in sorted_threats:
            # Update TTI
            distance = np.linalg.norm(threat.position - mother_position)
            closing_vel = -np.dot(threat.velocity - mother_velocity,
                                   (threat.position - mother_position) / (distance + 1e-9))
            threat.time_to_impact = distance / (closing_vel + 1e-9) if closing_vel > 0 else float('inf')
            
            # Threat summary
            response['threat_summary'].append({
                'id': threat.threat_id,
                'type': threat.threat_type.value,
                'priority': threat.priority.value,
                'tti': threat.time_to_impact,
                'assigned_tab': threat.assigned_tab
            })
            
            # Handle critical threats
            if threat.priority == ThreatPriority.CRITICAL:
                # Assign TAB if not already assigned
                if threat.assigned_tab is None:
                    self.assign_tab_to_threat(threat.threat_id, tab_states, 30.0)
                
                # If assigned, prepare release command
                if threat.assigned_tab and threat.confidence >= self.config.sacrifice_threshold:
                    solution = self.calculate_intercept_solution(
                        threat,
                        tab_states[threat.assigned_tab]['position'],
                        tab_states[threat.assigned_tab]['velocity'],
                        30.0
                    )
                    
                    if solution['solution_exists']:
                        response['release_commands'].append({
                            'tab_id': threat.assigned_tab,
                            'threat_id': threat.threat_id,
                            'mode': 'slingshot' if solution['release_angle'] > 20 else 'instant',
                            'launch_direction': solution['launch_direction'],
                            'intercept_time': solution['intercept_time']
                        })
                
                # Calculate evasion vector (opposite of threat approach)
                threat_direction = (threat.position - mother_position)
                threat_direction = threat_direction / (np.linalg.norm(threat_direction) + 1e-9)
                response['evasion_vector'] -= threat_direction * (1.0 / threat.priority.value)
        
        # Normalize evasion vector
        evasion_mag = np.linalg.norm(response['evasion_vector'])
        if evasion_mag > 0:
            response['evasion_vector'] /= evasion_mag
        
        return response
    
    def execute_intercept(self, threat_id: str) -> Dict:
        """Mark an intercept as executed"""
        if threat_id in self.tracked_threats:
            self.intercepts_executed += 1
            
            # Remove assignment
            if threat_id in self.tab_assignments:
                del self.tab_assignments[threat_id]
            
            return {'success': True, 'threat_id': threat_id}
        
        return {'success': False, 'error': 'Threat not found'}
    
    def confirm_neutralization(self, threat_id: str) -> Dict:
        """Confirm a threat has been neutralized"""
        if threat_id in self.tracked_threats:
            self.threats_neutralized += 1
            del self.tracked_threats[threat_id]
            
            # Update alert level
            if not self.tracked_threats:
                self.alert_level = "GREEN"
            
            return {'success': True, 'neutralized': threat_id}
        
        return {'success': False, 'error': 'Threat not found'}
    
    def get_defensive_bubble_status(self) -> Dict:
        """Get overall status of the defensive matrix"""
        return {
            'alert_level': self.alert_level,
            'active_threats': len(self.tracked_threats),
            'assigned_tabs': len(self.tab_assignments),
            'intercepts_executed': self.intercepts_executed,
            'threats_neutralized': self.threats_neutralized,
            'threats': [
                {
                    'id': t.threat_id,
                    'type': t.threat_type.value,
                    'priority': t.priority.name,
                    'tti': t.time_to_impact,
                    'confidence': t.confidence
                }
                for t in self.tracked_threats.values()
            ]
        }
