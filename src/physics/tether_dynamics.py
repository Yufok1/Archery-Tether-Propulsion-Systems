"""
Tether Dynamics Module
======================
Physics engine for cable tension, constraints, and snap mechanics.

This module handles:
- Spring-damper cable model (Kelvin-Voigt viscoelastic)
- Tension calculations under aerodynamic load
- Cable release/snap mechanics
- Whip prevention algorithms
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional
from enum import Enum


class CableState(Enum):
    """Cable operational states"""
    ATTACHED = "attached"      # Normal towing operation
    RELEASING = "releasing"    # In process of mechanical release
    SEVERED = "severed"        # Fully disconnected
    SLACK = "slack"            # Tension below minimum (dangerous)


@dataclass
class CableProperties:
    """Physical properties of tether cable"""
    length: float              # Nominal length (m)
    diameter: float            # Cable diameter (m)
    breaking_strength: float   # Maximum tension before snap (N)
    stiffness: float          # Young's modulus (N/m²)
    damping: float            # Damping ratio (dimensionless)
    linear_density: float     # Mass per unit length (kg/m)
    max_tension: float        # Operating limit (N)
    min_tension: float        # Slack threshold (N)


class TetherConstraint:
    """
    Models a single tether cable between mother drone and TAB.
    
    Uses spring-damper (Kelvin-Voigt) model for realistic stretch behavior:
    
        F_tension = k * (L - L0) + c * (dL/dt)
    
    Where:
        k = stiffness coefficient
        c = damping coefficient
        L = current length
        L0 = nominal length
        dL/dt = rate of length change
    """
    
    def __init__(self, 
                 cable_id: str,
                 properties: CableProperties,
                 anchor_point_mother: np.ndarray,
                 anchor_point_tab: np.ndarray):
        
        self.cable_id = cable_id
        self.props = properties
        self.state = CableState.ATTACHED
        
        # Anchor points in local coordinates
        self.anchor_mother = np.array(anchor_point_mother, dtype=np.float64)
        self.anchor_tab = np.array(anchor_point_tab, dtype=np.float64)
        
        # State variables
        self.current_tension = 0.0
        self.current_length = properties.length
        self.stretch_rate = 0.0
        
        # Calculate spring/damper coefficients from material properties
        cross_section = np.pi * (properties.diameter / 2) ** 2
        self.spring_k = (properties.stiffness * cross_section) / properties.length
        self.damper_c = properties.damping * 2 * np.sqrt(self.spring_k * properties.linear_density * properties.length)
        
        # History for rate calculation
        self._prev_length = properties.length
        self._prev_time = 0.0
        
    def compute_tension_force(self,
                               pos_mother: np.ndarray,
                               pos_tab: np.ndarray,
                               vel_mother: np.ndarray,
                               vel_tab: np.ndarray,
                               dt: float) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Calculate tension forces on both ends of the cable.
        
        Returns:
            force_on_mother: Force vector applied to mother drone (N)
            force_on_tab: Force vector applied to TAB (N)
            tension_magnitude: Scalar tension value (N)
        """
        if self.state == CableState.SEVERED:
            return np.zeros(3), np.zeros(3), 0.0
        
        # Vector from mother to TAB
        displacement = pos_tab - pos_mother
        self.current_length = np.linalg.norm(displacement)
        
        if self.current_length < 1e-6:
            return np.zeros(3), np.zeros(3), 0.0
        
        # Unit vector along cable
        cable_direction = displacement / self.current_length
        
        # Calculate stretch (extension beyond nominal length)
        stretch = self.current_length - self.props.length
        
        # Calculate stretch rate (for damping)
        if dt > 0:
            self.stretch_rate = (self.current_length - self._prev_length) / dt
        self._prev_length = self.current_length
        
        # Spring-damper force (Kelvin-Voigt model)
        # Only apply tension when stretched (cables can't push)
        if stretch > 0:
            spring_force = self.spring_k * stretch
            damper_force = self.damper_c * self.stretch_rate
            tension_magnitude = max(0, spring_force + damper_force)
        else:
            # Cable is slack
            tension_magnitude = 0.0
            self.state = CableState.SLACK
        
        # Clamp to operating limits
        if tension_magnitude > self.props.breaking_strength:
            # SNAP! Cable breaks under load
            self._trigger_snap()
            return np.zeros(3), np.zeros(3), 0.0
        
        tension_magnitude = min(tension_magnitude, self.props.max_tension)
        self.current_tension = tension_magnitude
        
        # Check for slack state
        if tension_magnitude < self.props.min_tension:
            self.state = CableState.SLACK
        elif self.state == CableState.SLACK:
            self.state = CableState.ATTACHED
        
        # Force vectors (equal and opposite)
        force_on_mother = cable_direction * tension_magnitude   # Pulls mother backward
        force_on_tab = -cable_direction * tension_magnitude     # Pulls TAB forward
        
        return force_on_mother, force_on_tab, tension_magnitude
    
    def release(self, whip_prevention: bool = True) -> np.ndarray:
        """
        Execute cable release sequence.
        
        Args:
            whip_prevention: If True, applies damping to prevent cable striking mother
            
        Returns:
            final_momentum: Momentum vector transferred to TAB at release
        """
        if self.state == CableState.SEVERED:
            return np.zeros(3)
        
        self.state = CableState.RELEASING
        
        # Store the tension energy at moment of release
        # This converts to kinetic energy in the TAB
        stored_energy = 0.5 * self.spring_k * max(0, self.current_length - self.props.length) ** 2
        
        if whip_prevention:
            # Apply braking to cable retraction
            # Mother drone retracts cable with controlled damping
            pass  # Would trigger servo/winch control
        
        self.state = CableState.SEVERED
        self.current_tension = 0.0
        
        return stored_energy
    
    def _trigger_snap(self):
        """Handle catastrophic cable failure"""
        self.state = CableState.SEVERED
        self.current_tension = 0.0
        # In real sim, would spawn cable debris particles
        
    def get_cable_strain(self) -> float:
        """Get current strain (stretch / original length)"""
        return (self.current_length - self.props.length) / self.props.length
    
    def get_cable_stress(self) -> float:
        """Get current stress (tension / cross-sectional area)"""
        cross_section = np.pi * (self.props.diameter / 2) ** 2
        return self.current_tension / cross_section


class TetherArray:
    """
    Manages the complete array of tether cables for the KAPS system.
    
    Handles:
    - Multi-cable coordination
    - Tangle detection/prevention
    - Synchronized release sequences
    - Formation constraint forces
    """
    
    def __init__(self, cable_properties: CableProperties, num_cables: int = 4):
        self.cables: dict[str, TetherConstraint] = {}
        self.cable_props = cable_properties
        
        # Default cross-formation anchor points on mother drone
        # UP, DOWN, LEFT, RIGHT relative to drone body
        anchor_offsets = {
            "UP": np.array([0.0, 0.0, 0.5]),
            "DOWN": np.array([0.0, 0.0, -0.5]),
            "LEFT": np.array([0.0, -0.5, 0.0]),
            "RIGHT": np.array([0.0, 0.5, 0.0]),
        }
        
        for i, (name, offset) in enumerate(anchor_offsets.items()):
            if i >= num_cables:
                break
            self.cables[name] = TetherConstraint(
                cable_id=name,
                properties=cable_properties,
                anchor_point_mother=offset,
                anchor_point_tab=np.array([0.0, 0.0, 0.0])  # Center of TAB
            )
    
    def compute_all_forces(self,
                           mother_pos: np.ndarray,
                           mother_vel: np.ndarray,
                           tab_positions: dict[str, np.ndarray],
                           tab_velocities: dict[str, np.ndarray],
                           dt: float) -> dict:
        """
        Compute tension forces for all cables simultaneously.
        
        Returns dict with force data for each cable.
        """
        results = {}
        total_force_on_mother = np.zeros(3)
        
        for cable_id, cable in self.cables.items():
            if cable_id not in tab_positions:
                continue
                
            f_mother, f_tab, tension = cable.compute_tension_force(
                mother_pos, 
                tab_positions[cable_id],
                mother_vel,
                tab_velocities.get(cable_id, np.zeros(3)),
                dt
            )
            
            total_force_on_mother += f_mother
            
            results[cable_id] = {
                "force_on_mother": f_mother,
                "force_on_tab": f_tab,
                "tension": tension,
                "state": cable.state,
                "length": cable.current_length,
                "strain": cable.get_cable_strain()
            }
        
        results["total_mother_force"] = total_force_on_mother
        return results
    
    def release_cable(self, cable_id: str) -> bool:
        """Release a specific cable by ID"""
        if cable_id in self.cables:
            self.cables[cable_id].release()
            return True
        return False
    
    def release_all(self) -> dict[str, float]:
        """
        Emergency release of all cables.
        Used for the "speed burst" maneuver.
        
        Returns energy released per cable.
        """
        energies = {}
        for cable_id, cable in self.cables.items():
            energies[cable_id] = cable.release()
        return energies
    
    def check_tangle_risk(self, tab_positions: dict[str, np.ndarray]) -> list[Tuple[str, str]]:
        """
        Detect if any cables are at risk of tangling.
        
        Returns list of (cable1, cable2) pairs that are too close.
        """
        tangle_pairs = []
        cable_ids = list(tab_positions.keys())
        
        for i, id1 in enumerate(cable_ids):
            for id2 in cable_ids[i+1:]:
                pos1 = tab_positions[id1]
                pos2 = tab_positions[id2]
                
                # Check if TABs are closer than safe distance
                distance = np.linalg.norm(pos1 - pos2)
                safe_distance = self.cable_props.length * 0.1  # 10% of cable length
                
                if distance < safe_distance:
                    tangle_pairs.append((id1, id2))
        
        return tangle_pairs
    
    def get_total_drag_contribution(self) -> float:
        """
        Estimate additional drag on mother drone from cable windage.
        
        Returns approximate drag force in Newtons.
        """
        total_drag = 0.0
        cable_drag_coeff = 1.2  # Cylinder in crossflow
        
        for cable in self.cables.values():
            if cable.state == CableState.ATTACHED:
                # Approximate cable as cylinder in airflow
                # D = 0.5 * rho * V^2 * Cd * A
                # Assuming 50 m/s airspeed, sea level density
                airspeed = 50.0
                rho = 1.225
                area = cable.props.diameter * cable.current_length
                drag = 0.5 * rho * airspeed**2 * cable_drag_coeff * area
                total_drag += drag
        
        return total_drag
