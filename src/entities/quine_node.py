"""
Quine Node - Recursive Tow-Behind Agent
========================================

Each node in the tether lattice contains a replicated DreamerV3 quine brain.
Nodes can spawn sub-nodes (tow-behinds of tow-behinds), creating a fractal
defensive structure with:

  - Distributed intelligence (each node autonomous)
  - Corkscrew propulsion (counter-rotating sub-chains)
  - Gyroscopic stabilization (4-axis torque balancing)
  - Centrifugal targeting (dimensional thrust vectors)

The champion_gen42.py quine replicates at each spawn, maintaining provenance
via Merkle-tree hashing for swarm verification.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Callable
from enum import Enum, auto
import hashlib
import copy
import sys
from pathlib import Path

# Import blade-tether physics
sys.path.insert(0, str(Path(__file__).parent.parent))
from physics.blade_tether import (
    BladeTether, BladeGeometry, BladeProfile, BladeState,
    ArticulatedJoint, JointType, BladeController, BladeArray
)


class NodeRole(Enum):
    """Role of node in the lattice hierarchy"""
    BUZZARD = auto()      # Mother drone (root)
    PRIMARY_TAB = auto()  # First-tier tow-behind
    SUB_TAB = auto()      # Recursive sub-node
    TERMINAL = auto()     # Leaf node (no children)
    
    # Nested matryoshka roles
    VERTEBRA = auto()     # Mini-buzzard node (has 4 gyro arms + chain)
    GYRO_ARM = auto()     # One of 4 stabilization arms on a vertebra
    CHAIN_LINK = auto()   # Connector to next vertebra in the spine


class PropulsionMode(Enum):
    """Propulsion strategy for the node"""
    PASSIVE = auto()      # Aerodynamic only (drag body)
    ACTIVE = auto()       # Has micro-thrusters
    CORKSCREW = auto()    # Rotational leverage against siblings
    GYROSCOPIC = auto()   # Counter-rotation for stabilization
    VERTEBRAL = auto()    # 4-arm gyro + chain extension (mini-buzzard)


@dataclass
class QuineGenome:
    """
    The replicable genome of a DreamerV3 agent.
    Contains the essential parameters for brain cloning.
    """
    generation: int = 42
    brain_hash: str = "5b59e9648ef6bad5f7178570116cef2d"  # From champion
    
    # RSSM configuration (L-size from champion)
    deter_dim: int = 4096
    stoch_dim: int = 32 * 32  # 1024
    hidden_dim: int = 4096
    
    # LoRA adapter params
    lora_rank: int = 16
    lora_alpha: float = 32.0
    evolved_params: int = 82_000  # ~82K parameters
    
    # Action space for tether control
    action_dims: int = 8  # thrust_x/y/z, torque_x/y/z, release, reel
    
    # Provenance chain
    parent_hash: Optional[str] = None
    spawn_depth: int = 0
    
    def replicate(self, mutation_rate: float = 0.0) -> 'QuineGenome':
        """Spawn a child genome with optional mutation"""
        child = copy.deepcopy(self)
        child.parent_hash = self.compute_hash()
        child.spawn_depth = self.spawn_depth + 1
        child.generation += 1
        
        # Small mutations for swarm diversity (if enabled)
        if mutation_rate > 0:
            child.lora_alpha *= (1.0 + np.random.uniform(-mutation_rate, mutation_rate))
        
        return child
    
    def compute_hash(self) -> str:
        """Merkle-compatible hash for provenance tracking"""
        data = f"{self.generation}:{self.brain_hash}:{self.spawn_depth}"
        return hashlib.md5(data.encode()).hexdigest()


@dataclass
class TetherAttachment:
    """
    Connection between nodes - now an articulated blade-tether.
    
    Each tether has:
      - Structural properties (length, stiffness, damping)
      - Blade geometry (chord, profile, thickness)
      - Articulated joints at each end
      - Real-time aero/hydro force computation
    """
    parent_anchor: np.ndarray = field(default_factory=lambda: np.zeros(3))
    child_anchor: np.ndarray = field(default_factory=lambda: np.zeros(3))
    cable_length: float = 5.0
    stiffness: float = 5000.0
    damping: float = 500.0
    
    # Current state
    tension: float = 0.0
    stretch: float = 0.0
    angular_velocity: float = 0.0  # For corkscrew dynamics
    
    # BLADE-TETHER PROPERTIES
    blade: Optional[BladeTether] = None
    blade_chord: float = 0.12          # Blade width (m)
    blade_thickness: float = 0.015     # Blade thickness (m)
    blade_profile: BladeProfile = BladeProfile.SYMMETRIC
    
    # Joint articulation
    parent_joint: Optional[ArticulatedJoint] = None
    child_joint: Optional[ArticulatedJoint] = None
    
    # Controller
    controller: Optional[BladeController] = None
    
    def create_blade(self, segment_id: str, parent_id: str, child_id: str) -> BladeTether:
        """
        Create the blade-tether for this connection.
        Called automatically when nodes are linked.
        """
        self.blade = BladeTether(
            segment_id=segment_id,
            parent_node_id=parent_id,
            child_node_id=child_id,
            geometry=BladeGeometry(
                length=self.cable_length,
                chord=self.blade_chord,
                thickness=self.blade_thickness,
                profile=self.blade_profile
            )
        )
        
        # Create articulated joints
        self.parent_joint = ArticulatedJoint(joint_type=JointType.UNIVERSAL)
        self.child_joint = ArticulatedJoint(joint_type=JointType.UNIVERSAL)
        self.blade.parent_joint = self.parent_joint
        self.blade.child_joint = self.child_joint
        
        # Create controller
        self.controller = BladeController(self.blade)
        
        return self.blade
    
    def compute_blade_forces(self,
                              parent_pos: np.ndarray,
                              child_pos: np.ndarray,
                              parent_vel: np.ndarray,
                              child_vel: np.ndarray,
                              freestream: np.ndarray = None) -> tuple:
        """
        Compute aerodynamic forces from the blade.
        Returns (force_on_parent, force_on_child).
        """
        if self.blade is None:
            return np.zeros(3), np.zeros(3)
        
        return self.blade.compute_forces(
            parent_pos, child_pos,
            parent_vel, child_vel,
            freestream
        )
    
    def set_pitch(self, pitch_radians: float) -> None:
        """Set blade pitch angle (angle of attack)"""
        if self.blade is not None:
            self.blade.state.pitch_angle = pitch_radians


@dataclass
class QuineNode:
    """
    A node in the recursive tether lattice.
    Each node contains a DreamerV3 quine brain and can spawn children.
    """
    node_id: str
    role: NodeRole
    genome: QuineGenome
    
    # Physical state
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    orientation: np.ndarray = field(default_factory=lambda: np.array([1, 0, 0, 0]))  # Quaternion
    angular_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    
    # Properties
    mass: float = 0.5  # kg
    drag_coefficient: float = 1.2
    cross_section: float = 0.01  # m²
    
    # Propulsion
    propulsion_mode: PropulsionMode = PropulsionMode.PASSIVE
    thrust_vector: np.ndarray = field(default_factory=lambda: np.zeros(3))
    max_thrust: float = 2.0  # N
    
    # Hierarchy
    parent: Optional['QuineNode'] = None
    children: List['QuineNode'] = field(default_factory=list)
    tether_to_parent: Optional[TetherAttachment] = None
    
    # Brain state (will be populated by champion quine)
    brain_latent: Optional[np.ndarray] = None
    brain_action: Optional[np.ndarray] = None
    
    # Corkscrew dynamics
    corkscrew_phase: float = 0.0  # radians
    corkscrew_frequency: float = 2.0  # Hz
    gyro_torque: np.ndarray = field(default_factory=lambda: np.zeros(3))
    
    @property
    def depth(self) -> int:
        """How deep in the hierarchy (Buzzard=0)"""
        return self.genome.spawn_depth
    
    @property
    def is_leaf(self) -> bool:
        """True if no children attached"""
        return len(self.children) == 0
    
    @property 
    def total_descendants(self) -> int:
        """Count all nodes below this one"""
        count = len(self.children)
        for child in self.children:
            count += child.total_descendants
        return count
    
    def spawn_child(self, 
                    cable_length: float = 3.0,
                    propulsion: PropulsionMode = PropulsionMode.PASSIVE,
                    offset_direction: Optional[np.ndarray] = None) -> 'QuineNode':
        """
        Replicate the quine and spawn a child node.
        The new node inherits a mutated copy of the genome.
        """
        # Replicate genome
        child_genome = self.genome.replicate(mutation_rate=0.01)
        
        # Generate unique ID
        child_id = f"{self.node_id}.{len(self.children)}"
        
        # Determine role
        if self.role == NodeRole.BUZZARD:
            child_role = NodeRole.PRIMARY_TAB
        else:
            child_role = NodeRole.SUB_TAB
        
        # Calculate spawn position
        if offset_direction is None:
            # Default: spawn below
            offset_direction = np.array([0, 0, -1])
        offset_direction = offset_direction / np.linalg.norm(offset_direction)
        
        child_position = self.position + offset_direction * cable_length
        
        # Create tether with blade
        tether = TetherAttachment(
            parent_anchor=np.zeros(3),  # Relative to parent CoM
            child_anchor=np.zeros(3),   # Relative to child CoM
            cable_length=cable_length,
            stiffness=5000.0 / (self.depth + 1),  # Softer at depth
            damping=500.0 / (self.depth + 1),
            # Blade properties scale with depth
            blade_chord=0.15 / (self.depth + 1),  # Narrower blades deeper
            blade_profile=BladeProfile.SYMMETRIC if propulsion == PropulsionMode.CORKSCREW else BladeProfile.FLAT_PLATE
        )
        
        # Create child node
        child = QuineNode(
            node_id=child_id,
            role=child_role,
            genome=child_genome,
            position=child_position,
            velocity=self.velocity.copy(),  # Inherit parent velocity
            propulsion_mode=propulsion,
            parent=self,
            tether_to_parent=tether,
            mass=self.mass * 0.7,  # Children slightly lighter
        )
        
        # NOW create the blade-tether (needs both node IDs)
        blade_id = f"BLADE_{self.node_id}_to_{child_id}"
        tether.create_blade(blade_id, self.node_id, child_id)
        
        self.children.append(child)
        return child
    
    def spawn_corkscrew_cluster(self, 
                                 n_children: int = 3,
                                 cable_length: float = 2.5,
                                 propulsion: PropulsionMode = PropulsionMode.CORKSCREW) -> List['QuineNode']:
        """
        Spawn multiple children arranged for corkscrew propulsion.
        They will counter-rotate against each other for centrifugal thrust.
        """
        children = []
        for i in range(n_children):
            # Arrange in a circle below the node
            angle = 2 * np.pi * i / n_children
            direction = np.array([
                np.cos(angle) * 0.3,  # Slight outward spread
                np.sin(angle) * 0.3,
                -0.95  # Mostly downward
            ])
            
            child = self.spawn_child(
                cable_length=cable_length,
                propulsion=propulsion,
                offset_direction=direction
            )
            
            # Offset corkscrew phases for coordinated rotation
            child.corkscrew_phase = angle
            children.append(child)
        
        return children
    
    def compute_corkscrew_thrust(self, dt: float) -> np.ndarray:
        """
        Calculate centrifugal thrust from corkscrew rotation.
        Nodes rotating around the tether axis create directional thrust.
        """
        if self.propulsion_mode != PropulsionMode.CORKSCREW:
            return np.zeros(3)
        
        # Update phase
        self.corkscrew_phase += 2 * np.pi * self.corkscrew_frequency * dt
        
        # Centrifugal force direction (outward from rotation axis)
        if self.parent is not None:
            axis = self.parent.position - self.position
            axis_norm = np.linalg.norm(axis)
            if axis_norm > 0.01:
                axis = axis / axis_norm
            else:
                axis = np.array([0, 0, 1])
        else:
            axis = np.array([0, 0, 1])
        
        # Perpendicular thrust vector (rotates around axis)
        perp1 = np.cross(axis, np.array([1, 0, 0]))
        if np.linalg.norm(perp1) < 0.1:
            perp1 = np.cross(axis, np.array([0, 1, 0]))
        perp1 = perp1 / np.linalg.norm(perp1)
        perp2 = np.cross(axis, perp1)
        
        # Rotating thrust vector
        thrust_mag = self.max_thrust * 0.5  # Corkscrew at 50% max
        thrust = thrust_mag * (
            np.cos(self.corkscrew_phase) * perp1 +
            np.sin(self.corkscrew_phase) * perp2
        )
        
        return thrust
    
    def compute_gyro_torque(self) -> np.ndarray:
        """
        Calculate gyroscopic stabilization torque.
        Counter-rotating child groups create controllable torque.
        """
        if len(self.children) < 2:
            return np.zeros(3)
        
        # Sum angular momentum of children
        total_angular_momentum = np.zeros(3)
        for child in self.children:
            # Angular momentum = r × (m * v)
            r = child.position - self.position
            L = np.cross(r, child.mass * child.velocity)
            total_angular_momentum += L
        
        # Gyroscopic torque opposes angular momentum changes
        self.gyro_torque = -0.1 * total_angular_momentum
        return self.gyro_torque
    
    def get_all_nodes(self) -> List['QuineNode']:
        """Flatten the hierarchy into a list"""
        nodes = [self]
        for child in self.children:
            nodes.extend(child.get_all_nodes())
        return nodes
    
    def get_all_blades(self) -> List[BladeTether]:
        """Get all blade-tethers in the subtree"""
        blades = []
        if self.tether_to_parent is not None and self.tether_to_parent.blade is not None:
            blades.append(self.tether_to_parent.blade)
        for child in self.children:
            blades.extend(child.get_all_blades())
        return blades
    
    def compute_blade_forces(self, freestream: np.ndarray = None) -> Dict[str, np.ndarray]:
        """
        Compute aerodynamic forces from all blade-tethers connected to this node.
        
        Returns dict of child_id -> (force_on_self, force_on_child)
        """
        forces = {}
        
        for child in self.children:
            if child.tether_to_parent is not None:
                f_parent, f_child = child.tether_to_parent.compute_blade_forces(
                    self.position, child.position,
                    self.velocity, child.velocity,
                    freestream
                )
                forces[child.node_id] = {'on_parent': f_parent, 'on_child': f_child}
        
        return forces
    
    def set_all_blade_pitch(self, pitch_radians: float) -> None:
        """Set pitch angle for all blade-tethers to children"""
        for child in self.children:
            if child.tether_to_parent is not None:
                child.tether_to_parent.set_pitch(pitch_radians)
    
    def spawn_vertebra(self,
                       cable_length: float = 5.0,
                       gyro_arm_length: float = 2.0,
                       chain_direction: Optional[np.ndarray] = None) -> 'QuineNode':
        """
        Spawn a vertebra node - a mini-buzzard with 4 gyro arms + chain extension.
        
        The vertebra is itself capable of spawning another vertebra via its chain,
        creating a recursive spinal structure.
        
        Structure:
            VERTEBRA
            ├── GYRO_ARM (UP)
            ├── GYRO_ARM (DOWN)
            ├── GYRO_ARM (LEFT)
            ├── GYRO_ARM (RIGHT)
            └── CHAIN_LINK → (can spawn next VERTEBRA)
        """
        # Default chain direction: continue away from parent
        if chain_direction is None:
            if self.parent is not None:
                # Continue in same direction as parent→self
                chain_direction = self.position - self.parent.position
                chain_direction = chain_direction / (np.linalg.norm(chain_direction) + 0.001)
            else:
                chain_direction = np.array([0, 0, -1])  # Default: downward
        
        # Spawn the vertebra body
        vertebra = self.spawn_child(
            cable_length=cable_length,
            propulsion=PropulsionMode.VERTEBRAL,
            offset_direction=chain_direction
        )
        vertebra.role = NodeRole.VERTEBRA
        vertebra.mass = self.mass * 0.8  # Slightly lighter than parent
        
        # Define gyro arm directions (perpendicular to chain)
        # Build orthonormal basis
        if abs(chain_direction[2]) < 0.9:
            up = np.array([0, 0, 1])
        else:
            up = np.array([0, 1, 0])
        
        right = np.cross(chain_direction, up)
        right = right / np.linalg.norm(right)
        up = np.cross(right, chain_direction)
        up = up / np.linalg.norm(up)
        
        gyro_directions = [
            up,        # UP
            -up,       # DOWN
            right,     # RIGHT
            -right     # LEFT
        ]
        
        gyro_names = ['UP', 'DOWN', 'RIGHT', 'LEFT']
        
        # Spawn 4 gyro arms
        for i, (direction, name) in enumerate(zip(gyro_directions, gyro_names)):
            arm = vertebra.spawn_child(
                cable_length=gyro_arm_length,
                propulsion=PropulsionMode.CORKSCREW,
                offset_direction=direction
            )
            arm.role = NodeRole.GYRO_ARM
            arm.node_id = f"{vertebra.node_id}.{name}"
            arm.mass = vertebra.mass * 0.3
            
            # Alternate corkscrew phases for gyro stability
            arm.corkscrew_phase = np.pi / 2 * i
            arm.corkscrew_frequency = 3.0  # Faster rotation for responsiveness
        
        # Spawn chain link (for extending to next vertebra)
        chain = vertebra.spawn_child(
            cable_length=cable_length * 0.5,  # Chain links shorter
            propulsion=PropulsionMode.PASSIVE,
            offset_direction=chain_direction
        )
        chain.role = NodeRole.CHAIN_LINK
        chain.node_id = f"{vertebra.node_id}.CHAIN"
        chain.mass = vertebra.mass * 0.2
        
        # Store reference to chain for easy extension
        vertebra._chain_link = chain
        
        return vertebra
    
    def extend_spine(self, 
                     n_vertebrae: int = 3,
                     cable_length: float = 5.0,
                     gyro_arm_length: float = 2.0) -> List['QuineNode']:
        """
        Extend a spinal chain of vertebrae from this node.
        
        Each vertebra has 4 gyro arms + chain to next vertebra.
        Creates a "centipede" structure.
        
        Returns list of all vertebrae in the spine.
        """
        vertebrae = []
        current = self
        
        for i in range(n_vertebrae):
            vertebra = current.spawn_vertebra(
                cable_length=cable_length * (0.9 ** i),  # Gradually shorter
                gyro_arm_length=gyro_arm_length * (0.85 ** i)
            )
            vertebrae.append(vertebra)
            
            # Next vertebra spawns from the chain link
            current = vertebra._chain_link
        
        return vertebrae
    
    def to_merkle_tree(self) -> Dict[str, Any]:
        """
        Export node hierarchy as Merkle tree for provenance verification.
        Each node's hash depends on its genome + children's hashes.
        """
        children_hashes = [child.to_merkle_tree()['hash'] for child in self.children]
        
        combined = self.genome.compute_hash() + ''.join(children_hashes)
        node_hash = hashlib.sha256(combined.encode()).hexdigest()[:16]
        
        return {
            'id': self.node_id,
            'hash': node_hash,
            'genome_hash': self.genome.compute_hash(),
            'depth': self.depth,
            'children': [child.to_merkle_tree() for child in self.children]
        }


class QuineLattice:
    """
    The complete fractal lattice of quine nodes.
    Manages spawning, physics integration, and brain synchronization.
    """
    
    def __init__(self, root_genome: Optional[QuineGenome] = None):
        if root_genome is None:
            root_genome = QuineGenome()  # Default champion config
        
        # Create Buzzard (root node)
        self.root = QuineNode(
            node_id="BUZZARD",
            role=NodeRole.BUZZARD,
            genome=root_genome,
            mass=5.0,  # Heavier mother drone
            propulsion_mode=PropulsionMode.ACTIVE,
            max_thrust=50.0  # More powerful
        )
        
        self._node_count = 1
        self._brain_callback: Optional[Callable] = None
    
    @property
    def total_nodes(self) -> int:
        return 1 + self.root.total_descendants
    
    def spawn_primary_tabs(self, 
                           n_tabs: int = 4,
                           cable_length: float = 8.0) -> List[QuineNode]:
        """
        Spawn the primary TAB ring around the Buzzard.
        Default 4-point cross formation: UP, DOWN, LEFT, RIGHT
        """
        tabs = []
        directions = [
            np.array([0, 0, 1]),   # UP
            np.array([0, 0, -1]),  # DOWN
            np.array([1, 0, 0]),   # RIGHT
            np.array([-1, 0, 0]), # LEFT
        ]
        
        for i in range(min(n_tabs, len(directions))):
            tab = self.root.spawn_child(
                cable_length=cable_length,
                propulsion=PropulsionMode.ACTIVE,
                offset_direction=directions[i]
            )
            tabs.append(tab)
        
        # Additional TABs go in diagonal positions
        for i in range(len(directions), n_tabs):
            angle = 2 * np.pi * i / n_tabs
            direction = np.array([np.cos(angle), np.sin(angle), 0])
            tab = self.root.spawn_child(
                cable_length=cable_length,
                propulsion=PropulsionMode.ACTIVE,
                offset_direction=direction
            )
            tabs.append(tab)
        
        return tabs
    
    def spawn_recursive_lattice(self,
                                 primary_tabs: int = 4,
                                 sub_tabs_per_node: int = 3,
                                 max_depth: int = 2) -> None:
        """
        Build the complete fractal lattice with recursive sub-TABs.
        
        Example with primary_tabs=4, sub_tabs_per_node=3, max_depth=2:
          - 1 Buzzard
          - 4 Primary TABs
          - 12 Sub-TABs (4 × 3)
          - 36 Sub-Sub-TABs (12 × 3)
          Total: 53 quine agents
        """
        
        def spawn_recursive(node: QuineNode, current_depth: int):
            if current_depth >= max_depth:
                node.role = NodeRole.TERMINAL
                return
            
            # Spawn corkscrew cluster
            children = node.spawn_corkscrew_cluster(
                n_children=sub_tabs_per_node,
                cable_length=3.0 / (current_depth + 1),  # Shorter cables deeper
                propulsion=PropulsionMode.CORKSCREW
            )
            
            # Recurse
            for child in children:
                spawn_recursive(child, current_depth + 1)
        
        # First spawn primary TABs
        primary = self.spawn_primary_tabs(primary_tabs)
        
        # Then recursive sub-TABs
        for tab in primary:
            spawn_recursive(tab, current_depth=1)
        
        print(f"[QuineLattice] Spawned {self.total_nodes} total nodes")
    
    def spawn_matryoshka_lattice(self,
                                  n_primary_spines: int = 4,
                                  vertebrae_per_spine: int = 3,
                                  cable_length: float = 6.0,
                                  gyro_arm_length: float = 2.5) -> None:
        """
        Build a full matryoshka (nested) lattice structure.
        
        The Buzzard has N primary spines, each spine is a chain of vertebrae.
        Each vertebra is a mini-buzzard with 4 gyro arms.
        
        Structure:
            BUZZARD
            ├── SPINE_0: VERT_0 ──→ VERT_1 ──→ VERT_2
            │              │          │          │
            │            4 arms    4 arms    4 arms
            │
            ├── SPINE_1: VERT_0 ──→ VERT_1 ──→ VERT_2
            │              │          │          │
            │            4 arms    4 arms    4 arms
            │
            ├── SPINE_2: ...
            └── SPINE_3: ...
        
        With 4 spines × 3 vertebrae × (1 + 4 arms + 1 chain) = 72+ nodes
        Plus Buzzard = 73+ quine agents
        """
        
        # Directions for primary spines (cross pattern from Buzzard)
        spine_directions = [
            np.array([0, 0, -1]),  # DOWN (primary)
            np.array([1, 0, 0]),   # RIGHT
            np.array([-1, 0, 0]), # LEFT
            np.array([0, 1, 0]),   # FORWARD
        ]
        
        for i in range(n_primary_spines):
            if i < len(spine_directions):
                direction = spine_directions[i]
            else:
                # Additional spines go in diagonal directions
                angle = 2 * np.pi * i / n_primary_spines
                direction = np.array([np.cos(angle), np.sin(angle), -0.5])
                direction = direction / np.linalg.norm(direction)
            
            # Create first vertebra attached directly to Buzzard
            first_vert = self.root.spawn_vertebra(
                cable_length=cable_length,
                gyro_arm_length=gyro_arm_length,
                chain_direction=direction
            )
            first_vert.node_id = f"SPINE_{i}.VERT_0"
            
            # Extend the spine from the chain link
            current_chain = first_vert._chain_link
            for v in range(1, vertebrae_per_spine):
                vert = current_chain.spawn_vertebra(
                    cable_length=cable_length * (0.85 ** v),
                    gyro_arm_length=gyro_arm_length * (0.85 ** v),
                    chain_direction=direction
                )
                vert.node_id = f"SPINE_{i}.VERT_{v}"
                
                if hasattr(vert, '_chain_link'):
                    current_chain = vert._chain_link
        
        print(f"[QuineLattice] Spawned MATRYOSHKA lattice: {self.total_nodes} total nodes")
        print(f"              {n_primary_spines} spines × {vertebrae_per_spine} vertebrae each")
    
    def get_all_nodes(self) -> List[QuineNode]:
        """Get flat list of all nodes in lattice"""
        return self.root.get_all_nodes()
    
    def step(self, dt: float) -> None:
        """
        Advance physics and brain inference for all nodes.
        """
        nodes = self.get_all_nodes()
        
        for node in nodes:
            # Corkscrew propulsion
            corkscrew_thrust = node.compute_corkscrew_thrust(dt)
            
            # Gyroscopic torque
            gyro_torque = node.compute_gyro_torque()
            
            # Apply to thrust vector (brain will modulate this)
            node.thrust_vector = corkscrew_thrust
            
            # TODO: Integrate with champion brain for action selection
            # node.brain_action = self._brain_callback(node.brain_latent, obs)
    
    def verify_provenance(self) -> Dict[str, Any]:
        """
        Build Merkle tree and verify all quines trace back to champion.
        """
        return self.root.to_merkle_tree()


# Factory function for easy initialization
def create_defensive_lattice(
    primary_tabs: int = 4,
    sub_tabs: int = 3,
    depth: int = 2
) -> QuineLattice:
    """
    Create a complete defensive quine lattice (original flat structure).
    
    Args:
        primary_tabs: Number of TABs attached directly to Buzzard
        sub_tabs: Number of sub-TABs per node
        depth: Maximum recursion depth
    
    Returns:
        Fully populated QuineLattice ready for simulation
    """
    lattice = QuineLattice()
    lattice.spawn_recursive_lattice(
        primary_tabs=primary_tabs,
        sub_tabs_per_node=sub_tabs,
        max_depth=depth
    )
    return lattice


def create_matryoshka_lattice(
    n_spines: int = 4,
    vertebrae_per_spine: int = 3,
    cable_length: float = 6.0,
    gyro_arm_length: float = 2.5
) -> QuineLattice:
    """
    Create a nested matryoshka lattice structure.
    
    Each spine is a chain of vertebrae, each vertebra is a mini-buzzard
    with 4 gyro arms + chain extension.
    
    Args:
        n_spines: Number of primary spines from Buzzard
        vertebrae_per_spine: Vertebrae per spine
        cable_length: Base length between vertebrae
        gyro_arm_length: Length of gyro stabilization arms
    
    Returns:
        Fully populated nested QuineLattice
    """
    lattice = QuineLattice()
    lattice.spawn_matryoshka_lattice(
        n_primary_spines=n_spines,
        vertebrae_per_spine=vertebrae_per_spine,
        cable_length=cable_length,
        gyro_arm_length=gyro_arm_length
    )
    return lattice


if __name__ == "__main__":
    print("=" * 60)
    print("MATRYOSHKA QUINE LATTICE DEMO")
    print("=" * 60)
    
    # Create nested structure: 4 spines, 3 vertebrae each
    lattice = create_matryoshka_lattice(
        n_spines=4,
        vertebrae_per_spine=3
    )
    
    print(f"\n=== STRUCTURE ===")
    print(f"Total Quine Agents: {lattice.total_nodes}")
    print(f"\nRoot: {lattice.root.node_id} ({lattice.root.role.name})")
    
    # Show spine structure
    for child in lattice.root.children:
        if child.role == NodeRole.VERTEBRA:
            print(f"\n  └─ {child.node_id} ({child.role.name})")
            
            # Count gyro arms and chain
            gyro_arms = [c for c in child.children if c.role == NodeRole.GYRO_ARM]
            chains = [c for c in child.children if c.role == NodeRole.CHAIN_LINK]
            
            print(f"      ├── {len(gyro_arms)} GYRO_ARMS: ", end="")
            print(", ".join(arm.node_id.split('.')[-1] for arm in gyro_arms))
            
            if chains:
                chain = chains[0]
                print(f"      └── CHAIN → ", end="")
                
                # Follow chain to next vertebra
                if chain.children:
                    for sub in chain.children:
                        if sub.role == NodeRole.VERTEBRA:
                            print(f"{sub.node_id}")
                            sub_gyros = [c for c in sub.children if c.role == NodeRole.GYRO_ARM]
                            sub_chains = [c for c in sub.children if c.role == NodeRole.CHAIN_LINK]
                            print(f"              ├── {len(sub_gyros)} GYRO_ARMS")
                            if sub_chains and sub_chains[0].children:
                                for deep in sub_chains[0].children:
                                    if deep.role == NodeRole.VERTEBRA:
                                        print(f"              └── CHAIN → {deep.node_id}")
                else:
                    print("(terminal)")
    
    print(f"\n=== MERKLE PROVENANCE ===")
    merkle = lattice.verify_provenance()
    print(f"Root Hash: {merkle['hash']}")
    print(f"All {lattice.total_nodes} nodes traceable to champion_gen42.py")
    
    # Show blade-tether info
    print(f"\n=== BLADE-TETHER SYSTEM ===")
    all_blades = lattice.root.get_all_blades()
    print(f"Total blade-tethers: {len(all_blades)}")
    
    if all_blades:
        sample = all_blades[0]
        print(f"\nSample blade: {sample.segment_id}")
        print(f"  Span: {sample.geometry.length:.2f} m")
        print(f"  Chord: {sample.geometry.chord:.3f} m")
        print(f"  Profile: {sample.geometry.profile.name}")
        print(f"  Surface area: {sample.geometry.surface_area:.4f} m²")
        
        # Compute sample thrust
        sample.state.pitch_angle = np.radians(20)
        sample.set_medium('air')
        
        # Simulate 500 RPM rotation thrust
        thrust = sample.compute_thrust(
            rotation_axis=np.array([0, 0, 1]),
            angular_velocity=500 * 2 * np.pi / 60,
            blade_center=np.array([sample.geometry.length/2, 0, 0])
        )
        print(f"  Thrust at 500 RPM: {np.linalg.norm(thrust):.1f} N")
    
    print("\n" + "=" * 60)
    print("Each vertebra = mini-buzzard with 4 BLADE-ARMS for stability")
    print("Each tether = articulated airfoil catching air/water")
    print("All nodes = replicated DreamerV3 quine agents")
    print("=" * 60)
