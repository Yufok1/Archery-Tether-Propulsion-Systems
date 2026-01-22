"""
Visual Training Mode with Interactive HOLD Control
===================================================

Watch DreamerV3 train in real-time with the Panda3D visualizer.

INTERACTIVE CONTROLS:
---------------------
SPACE   - Pause/Resume (Full HOLD)
F       - Freeze AI only (environment continues)
.       - Step one frame

Speed:
+/-     - Speed up/down

SHOWCASE MODES (demonstrate capabilities):
1 - Systems Check (test all actuators)
2 - Defensive Sphere (max coverage)
3 - Interception Drill (threat pursuit)
4 - Max Extension (cable limits)
5 - Corkscrew Spin (propulsion)
6 - Formation Parade (tactical patterns)
7 - Throttle Test (0 to max)
8 - Evasive Demo (agility)
9 - Sacrifice Drill (TAB expendability)
0 - SLINGSHOT / NUNCHAKU mode (Bruce Lee bola swing!)

CAMERA MODES:
V - Chase cam (behind Buzzard)
B - Orbit cam (circles around)
N - Tactical (top-down)
M - Dynamic (automatic transitions)
, - Cockpit (first person)
C - Cycle through modes

ESC     - Stop showcase, return to training
R       - Reset episode
"""

import numpy as np
import time
import threading
from typing import Dict, Optional, List
from collections import deque

import sys
import os
# Fix imports for both module and direct execution
_parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_root = os.path.dirname(_parent)
if _parent not in sys.path:
    sys.path.insert(0, _parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

try:
    from src.training.exploration_env import ExplorationKAPSEnv
    from src.training.threat_environment import Threat, ThreatType
    from src.training.hold_control import (
        KAPSHoldController, HoldMode, ShowcaseMode, HOLD_KEYBINDS
    )
    from src.visualization.cinematic_camera import (
        CinematicCamera, CameraMode, CAMERA_KEYBINDS
    )
    from src.visualization.simple_geometry import (
        create_visible_buzzard, create_visible_airfoil,
        create_visible_cable, create_visible_bola, create_visible_grid_fin,
        VISIBLE_TAB_COLORS
    )
    # Alias for compatibility
    create_cable_geometry = create_visible_cable
    create_airfoil_geometry = create_visible_airfoil
    create_buzzard_geometry = create_visible_buzzard
    TAB_COLORS = VISIBLE_TAB_COLORS
    
    from src.physics.slingshot_dynamics import (
        SlingshotController, SlingshotState, AirfoiledBuzzard,
        DeployableGridFin, AirfoilDeployState
    )
except ImportError:
    from training.exploration_env import ExplorationKAPSEnv
    from training.threat_environment import Threat, ThreatType
    from training.hold_control import (
        KAPSHoldController, HoldMode, ShowcaseMode, HOLD_KEYBINDS
    )
    try:
        from visualization.cinematic_camera import (
            CinematicCamera, CameraMode, CAMERA_KEYBINDS
        )
        from visualization.simple_geometry import (
            create_visible_buzzard, create_visible_airfoil,
            create_visible_cable, create_visible_bola, create_visible_grid_fin,
            VISIBLE_TAB_COLORS
        )
        create_cable_geometry = create_visible_cable
        create_airfoil_geometry = create_visible_airfoil
        create_buzzard_geometry = create_visible_buzzard
        TAB_COLORS = VISIBLE_TAB_COLORS
    except ImportError:
        # Fallback - will define inline
        CinematicCamera = None
        CameraMode = None
        create_cable_geometry = None
        create_buzzard_geometry = None
        create_airfoil_geometry = None
        TAB_COLORS = None
    
    try:
        from physics.slingshot_dynamics import (
            SlingshotController, SlingshotState, AirfoiledBuzzard,
            DeployableGridFin, AirfoilDeployState
        )
    except ImportError:
        SlingshotController = None
        SlingshotState = None


# =============================================================================
# PANDA3D VISUALIZATION
# =============================================================================

try:
    from direct.showbase.ShowBase import ShowBase
    from panda3d.core import (
        Vec3, Vec4, Point3,
        LineSegs, NodePath, GeomNode,
        AmbientLight, DirectionalLight, PointLight,
        TextNode, ClockObject,
        GeomVertexFormat, GeomVertexData, GeomVertexWriter,
        Geom, GeomTriangles, Fog
    )
    from direct.task import Task
    PANDA3D_AVAILABLE = True
    globalClock = ClockObject.getGlobalClock()
except ImportError:
    PANDA3D_AVAILABLE = False
    print("Panda3D not available for visual training")


def create_sphere_geom(radius: float = 1.0, color: tuple = (1, 1, 1, 1), segments: int = 12) -> GeomNode:
    """Create sphere geometry"""
    format = GeomVertexFormat.getV3n3c4()
    vdata = GeomVertexData("sphere", format, Geom.UHStatic)
    
    vertex = GeomVertexWriter(vdata, "vertex")
    normal = GeomVertexWriter(vdata, "normal")
    col = GeomVertexWriter(vdata, "color")
    
    for i in range(segments + 1):
        lat = np.pi * (-0.5 + float(i) / segments)
        for j in range(segments + 1):
            lon = 2 * np.pi * float(j) / segments
            
            x = radius * np.cos(lat) * np.cos(lon)
            y = radius * np.cos(lat) * np.sin(lon)
            z = radius * np.sin(lat)
            
            vertex.addData3f(x, y, z)
            normal.addData3f(x/radius, y/radius, z/radius)
            col.addData4f(*color)
    
    prim = GeomTriangles(Geom.UHStatic)
    for i in range(segments):
        for j in range(segments):
            v0 = i * (segments + 1) + j
            v1 = v0 + 1
            v2 = v0 + segments + 1
            v3 = v2 + 1
            prim.addVertices(v0, v2, v1)
            prim.addVertices(v1, v2, v3)
    
    geom = Geom(vdata)
    geom.addPrimitive(prim)
    node = GeomNode("sphere")
    node.addGeom(geom)
    return node


def create_airfoil_geom(wingspan: float = 1.5, chord: float = 0.3, color: tuple = (1, 1, 1, 1)) -> GeomNode:
    """
    Create an airfoil/wing geometry for TABs.
    
    TABs are AIRFOILS - flying wings with visible thickness!
    """
    format = GeomVertexFormat.getV3n3c4()
    vdata = GeomVertexData("airfoil", format, Geom.UHStatic)
    
    vertex = GeomVertexWriter(vdata, "vertex")
    normal = GeomVertexWriter(vdata, "normal")
    col = GeomVertexWriter(vdata, "color")
    
    # More visible wing - thicker, proper delta shape
    half_span = wingspan / 2
    thickness = 0.15  # Thicker for visibility
    
    # Delta wing vertices - 6 vertices for a proper 3D wing
    #
    #     NOSE (front)
    #      /\
    #     /  \
    #    /    \
    #   /______\
    # LEFT    RIGHT (wingtips at back)
    #
    verts = [
        # Top surface
        (0, half_span, 0),              # 0: nose
        (-half_span, -half_span * 0.5, thickness),   # 1: left wingtip top
        (half_span, -half_span * 0.5, thickness),    # 2: right wingtip top
        # Bottom surface  
        (-half_span, -half_span * 0.5, -thickness),  # 3: left wingtip bottom
        (half_span, -half_span * 0.5, -thickness),   # 4: right wingtip bottom
        (0, -half_span * 0.3, 0),       # 5: tail center
    ]
    
    normals = [
        (0, 1, 0),   # nose
        (0, 0, 1),   # top left
        (0, 0, 1),   # top right
        (0, 0, -1),  # bottom left
        (0, 0, -1),  # bottom right
        (0, -1, 0),  # tail
    ]
    
    for i, v in enumerate(verts):
        vertex.addData3f(*v)
        normal.addData3f(*normals[i])
        col.addData4f(*color)
    
    prim = GeomTriangles(Geom.UHStatic)
    # Top surface (2 triangles)
    prim.addVertices(0, 1, 2)  # nose to wingtips
    prim.addVertices(1, 5, 2)  # wingtips to tail
    # Bottom surface (2 triangles)
    prim.addVertices(0, 4, 3)  # nose to wingtips (reversed winding)
    prim.addVertices(3, 4, 5)  # wingtips to tail
    # Sides (close the shape)
    prim.addVertices(1, 3, 5)  # left side
    prim.addVertices(2, 5, 4)  # right side
    prim.addVertices(0, 3, 1)  # left leading edge
    prim.addVertices(0, 2, 4)  # right leading edge
    
    geom = Geom(vdata)
    geom.addPrimitive(prim)
    node = GeomNode("airfoil")
    node.addGeom(geom)
    return node


if PANDA3D_AVAILABLE:
    
    class VisualTrainer(ShowBase):
        """
        Panda3D-based visual training interface with HOLD control.
        
        Features:
        - Cinematic camera system with multiple perspectives
        - Enhanced 3D geometry for cables and airfoils
        - Interactive HOLD control for pause/step/speed
        - Showcase modes for capability demonstration
        """
        
        def __init__(self, 
                     dreamer_agent=None,
                     episode_steps: int = 3000,
                     threat_interval: int = 100,
                     speed_multiplier: float = 1.0):
            ShowBase.__init__(self)
            
            self.dreamer_agent = dreamer_agent
            self.speed_multiplier = speed_multiplier
            
            # HOLD controller for interactive control
            self.hold = KAPSHoldController()
            
            # Cinematic camera system
            if CinematicCamera is not None:
                self.cinema_cam = CinematicCamera()
            else:
                self.cinema_cam = None
            
            # Create environment
            self.env = ExplorationKAPSEnv(
                episode_steps=episode_steps,
                threat_spawn_interval=threat_interval
            )
            
            # Episode state
            self.obs = None
            self.episode_reward = 0
            self.episode_count = 0
            self.total_steps = 0
            self.recent_rewards = deque(maxlen=100)
            self.recent_intercepts = deque(maxlen=100)
            
            # Legacy camera (fallback)
            self.disableMouse()
            self.camera_distance = 150
            self.camera_heading = 45
            self.camera_pitch = 25
            self.camera_target = np.array([0.0, 0.0, 1000.0])
            self.camera_mode_index = 0
            
            # Sky gradient background
            self.setBackgroundColor(0.4, 0.6, 0.9)
            
            # Enhanced lighting
            self._setup_lighting()
            
            # Visual nodes
            self.mother_node = None
            self.tab_nodes: Dict[str, NodePath] = {}
            self.cable_nodes: Dict[str, NodePath] = {}
            self.threat_nodes: Dict[str, NodePath] = {}
            self.hud_texts = {}
            
            # SLINGSHOT / BOLA MODE
            self.slingshot = SlingshotController(cable_length=50.0) if SlingshotController else None
            self.airfoiled_buzzard = AirfoiledBuzzard() if SlingshotController else None
            self.bola_node = None  # Visual for consolidated TAB mass
            self.grid_fin_nodes: Dict[str, NodePath] = {}  # Visual for grid fins
            self.buzzard_wing_nodes = []  # Visual for Buzzard's deployable wings
            self.slingshot_cable_node = None  # Braided cable in bola mode
            
            # Create scene
            self._create_ground()
            self._create_entities()
            self._create_hud()
            self._create_control_panel()
            
            # Input - camera controls
            self.accept("wheel_up", self._zoom_in)
            self.accept("wheel_down", self._zoom_out)
            self.accept("mouse1", self._start_drag)
            self.accept("mouse1-up", self._stop_drag)
            
            # CAMERA MODE controls
            self.accept("c", self._cycle_camera_mode)
            self.accept("v", lambda: self._set_camera_mode(CameraMode.CHASE))
            self.accept("b", lambda: self._set_camera_mode(CameraMode.ORBIT))
            self.accept("n", lambda: self._set_camera_mode(CameraMode.TACTICAL))
            self.accept("m", lambda: self._set_camera_mode(CameraMode.DYNAMIC))
            self.accept(",", lambda: self._set_camera_mode(CameraMode.COCKPIT))
            
            # HOLD controls
            self.accept("space", self._toggle_pause)
            self.accept("f", self._toggle_ai_freeze)
            self.accept(".", self._step_once)
            
            # Speed controls
            self.accept("=", self._speed_up)
            self.accept("+", self._speed_up)
            self.accept("-", self._slow_down)
            
            # SLINGSHOT MODE controls (direct, not showcase)
            self.accept("s", self._toggle_slingshot_mode)  # Toggle bola mode
            self.accept("x", self._slingshot_release)       # Release bola
            self.accept("z", self._deploy_grid_fins)        # Deploy grid fins
            self.accept("a", self._extend_buzzard_wings)    # Extend Buzzard airfoils
            
            # Showcase modes (0-9)
            self.accept("1", lambda: self._set_showcase(ShowcaseMode.SYSTEMS_CHECK))
            self.accept("2", lambda: self._set_showcase(ShowcaseMode.DEFENSIVE_SPHERE))
            self.accept("3", lambda: self._set_showcase(ShowcaseMode.INTERCEPTION_DRILL))
            self.accept("4", lambda: self._set_showcase(ShowcaseMode.MAX_EXTENSION))
            self.accept("5", lambda: self._set_showcase(ShowcaseMode.CORKSCREW_SPIN))
            self.accept("6", lambda: self._set_showcase(ShowcaseMode.FORMATION_PARADE))
            self.accept("7", lambda: self._set_showcase(ShowcaseMode.THROTTLE_TEST))
            self.accept("8", lambda: self._set_showcase(ShowcaseMode.EVASIVE_DEMO))
            self.accept("9", lambda: self._set_showcase(ShowcaseMode.SACRIFICE_DRILL))
            self.accept("0", lambda: self._set_showcase(ShowcaseMode.SLINGSHOT_DEMO))
            
            # Cancel/Reset
            self.accept("escape", self._stop_showcase)
            self.accept("backspace", self._stop_showcase)
            self.accept("r", self._reset_episode)
            self.accept("q", self.userExit)
            
            self.mouse_dragging = False
            self.last_mouse_x = 0
            self.last_mouse_y = 0
            
            # Track last frame time for dt calculation
            self.last_frame_time = time.time()
            
            # Update task
            self.taskMgr.add(self._update_task, "training_update")
            self.taskMgr.add(self._camera_task, "camera_update")
            
            # Start first episode
            self._reset_episode()
            
            self._print_controls()
        
        def _print_controls(self):
            """Print control instructions to terminal."""
            print("=" * 70)
            print("  KAPS VISUAL TRAINER - Cinematic Mode")
            print("=" * 70)
            print("")
            print("  CAMERA MODES:")
            print("    V - Chase (behind Buzzard, follows velocity)")
            print("    B - Orbit (circles around)")
            print("    N - Tactical (top-down overview)")
            print("    M - Dynamic (automatic cinematic transitions)")
            print("    , - Cockpit (first person)")
            print("    C - Cycle through modes")
            print("")
            print("  HOLD CONTROLS:")
            print("    SPACE     - Pause/Resume (Full HOLD)")
            print("    F         - Freeze AI only (environment continues)")
            print("    .         - Step one frame")
            print("")
            print("  SPEED:")
            print("    +/-       - Speed up/down")
            print("")
            print("  SHOWCASE DEMOS:")
            print("    1 - Systems Check    2 - Defensive Sphere")
            print("    3 - Intercept Drill  4 - Max Extension")
            print("    5 - Corkscrew Spin   6 - Formation Parade")
            print("    7 - Throttle Test    8 - Evasive Demo")
            print("    9 - Sacrifice Drill  0 - Slingshot/Bola Demo")
            print("")
            print("  SLINGSHOT / BOLA MODE (Bruce Lee nunchaku!):")
            print("    S - Toggle Slingshot mode (consolidate TABs)")
            print("    X - Release bola at max velocity")
            print("    Z - Deploy grid fins (post-release steering)")
            print("    A - Extend Buzzard wings (airfoiled mode)")
            print("")
            print("  OTHER:")
            print("    ESC/BKSP  - Stop showcase")
            print("    R - Reset episode    Q - Quit")
            print("=" * 70)
        
        # =====================================================================
        # CAMERA CONTROL METHODS
        # =====================================================================
        
        def _cycle_camera_mode(self):
            """Cycle through camera modes."""
            if self.cinema_cam is None:
                return
            modes = [CameraMode.CHASE, CameraMode.ORBIT, CameraMode.TACTICAL, 
                     CameraMode.DYNAMIC, CameraMode.COCKPIT, CameraMode.SIDE]
            self.camera_mode_index = (self.camera_mode_index + 1) % len(modes)
            self.cinema_cam.set_mode(modes[self.camera_mode_index])
            self._update_camera_display()
        
        def _set_camera_mode(self, mode):
            """Set specific camera mode."""
            if self.cinema_cam is None or mode is None:
                return
            self.cinema_cam.set_mode(mode)
            self._update_camera_display()
        
        def _update_camera_display(self):
            """Update camera mode in HUD."""
            if "camera" in self.hud_texts and self.cinema_cam:
                self.hud_texts["camera"].setText(f"CAM: {self.cinema_cam.mode.name}")
        
        # =====================================================================
        # HOLD CONTROL METHODS
        # =====================================================================
        
        def _toggle_pause(self):
            """Toggle full pause."""
            self.hold.toggle_pause()
            self._update_mode_display()
        
        def _toggle_ai_freeze(self):
            """Toggle AI freeze (environment continues)."""
            if self.hold.state.mode == HoldMode.AI_PAUSE:
                self.hold.set_mode(HoldMode.CONTINUOUS)
            else:
                self.hold.set_mode(HoldMode.AI_PAUSE)
            self._update_mode_display()
        
        def _step_once(self):
            """Step one frame."""
            self.hold.step_once()
        
        def _speed_up(self):
            """Increase simulation speed."""
            self.hold.speed_up()
            self._update_mode_display()
        
        def _slow_down(self):
            """Decrease simulation speed."""
            self.hold.slow_down()
            self._update_mode_display()
        
        def _reset_speed(self):
            """Reset to 1x speed."""
            self.hold.set_speed(1.0)
            self._update_mode_display()
        
        # =====================================================================
        # SLINGSHOT / BOLA MODE CONTROLS
        # =====================================================================
        
        def _toggle_slingshot_mode(self):
            """Toggle between dispersed and bola mode."""
            if self.slingshot is None:
                print("[SLINGSHOT] Not available")
                return
            
            if self.slingshot.state == SlingshotState.DISPERSED:
                # Consolidate TABs into bola
                print("[SLINGSHOT] Consolidating TABs into BOLA mode...")
                
                # Get TAB states from simulation
                if self.env and self.env.sim:
                    tab_states = {}
                    for tab_id, tab in self.env.sim.tab_array.tabs.items():
                        tab_states[tab_id] = {
                            'position': tab.position,
                            'velocity': tab.velocity,
                            'mass': getattr(tab, 'mass', 5.0),
                            'is_attached': tab.is_attached
                        }
                    
                    self.slingshot.consolidate_tabs(tab_states)
                    self._create_bola_visual()
                    print(f"[SLINGSHOT] BOLA READY! Mass: {self.slingshot.bola.mass} kg")
                    
                    # Start swing automatically
                    buzzard_pos = self.env.sim.mother_drone.position
                    self.slingshot.begin_swing(buzzard_pos, initial_angular_rate=1.0)
                    print("[SLINGSHOT] SWINGING - Use X to release!")
            else:
                # Disperse back to normal
                print("[SLINGSHOT] Dispersing to standard TAB mode...")
                self.slingshot.disperse_tabs()
                self._remove_bola_visual()
            
            self._update_mode_display()
        
        def _slingshot_release(self):
            """Release the bola at current angle."""
            if self.slingshot is None:
                return
            
            if self.slingshot.state == SlingshotState.SWINGING:
                pos, vel, speed = self.slingshot.release()
                print(f"[SLINGSHOT] RELEASED at {speed:.1f} m/s!")
                self._update_mode_display()
        
        def _deploy_grid_fins(self):
            """Deploy grid fins for post-release steering."""
            if self.slingshot is None:
                return
            
            if self.slingshot.state in [SlingshotState.BALLISTIC, SlingshotState.SAILING]:
                self.slingshot.deploy_grid_fins()
                self._update_grid_fin_visuals()
                print("[SLINGSHOT] Grid fins DEPLOYED!")
            
            # Also deploy Buzzard grid fins
            if self.airfoiled_buzzard:
                self.airfoiled_buzzard.deploy_all_fins()
                print("[BUZZARD] Grid fins deployed!")
        
        def _extend_buzzard_wings(self):
            """Extend Buzzard's lifting surfaces."""
            if self.airfoiled_buzzard is None:
                return
            
            # Toggle wing extension
            if self.airfoiled_buzzard.wing_extension < 0.5:
                target = 1.0
                print("[BUZZARD] Extending wings to full span!")
            else:
                target = 0.0
                print("[BUZZARD] Retracting wings!")
            
            # Will be animated in update loop
            self._buzzard_wing_target = target
        
        def _create_bola_visual(self):
            """Create visual representation of the consolidated bola."""
            if self.bola_node:
                self.bola_node.removeNode()
            
            # Bola is a larger sphere (combined mass)
            bola_geom = create_sphere_geom(radius=6, color=(0.9, 0.6, 0.2, 1))
            self.bola_node = self.render.attachNewNode(bola_geom)
            self.bola_node.setScale(1.5)
            
            # Create grid fin visuals on the bola
            if self.slingshot and self.slingshot.bola.grid_fins:
                for fin_id, fin in self.slingshot.bola.grid_fins.items():
                    fin_geom = create_airfoil_geom(wingspan=1.5, chord=0.4, 
                                                    color=(0.6, 0.6, 0.7, 1))
                    fin_node = self.bola_node.attachNewNode(fin_geom)
                    fin_node.setPos(*fin.position_local * 5)
                    fin_node.setScale(0.5)
                    self.grid_fin_nodes[fin_id] = fin_node
            
            print("[VIZ] Created BOLA visual with grid fins")
        
        def _remove_bola_visual(self):
            """Remove bola visual, show individual TABs again."""
            if self.bola_node:
                self.bola_node.removeNode()
                self.bola_node = None
            
            for node in self.grid_fin_nodes.values():
                node.removeNode()
            self.grid_fin_nodes.clear()
            
            # Show TABs again
            for node in self.tab_nodes.values():
                node.show()
        
        def _update_grid_fin_visuals(self):
            """Update grid fin orientations based on deployment/deflection."""
            if not self.slingshot or not self.slingshot.bola.grid_fins:
                return
            
            for fin_id, fin_node in self.grid_fin_nodes.items():
                fin = self.slingshot.bola.grid_fins.get(fin_id)
                if fin:
                    # Scale based on deployment
                    scale = 0.3 + 0.7 * fin.deploy_fraction
                    fin_node.setScale(scale)
                    
                    # Rotate based on deflection
                    fin_node.setR(fin.deflection)
        
        def _update_slingshot_visuals(self, dt: float):
            """Update all slingshot-related visuals each frame."""
            if self.slingshot is None:
                return
            
            state = self.slingshot.state
            
            if state == SlingshotState.DISPERSED:
                # Normal mode - show individual TABs
                if self.bola_node:
                    self.bola_node.hide()
                for node in self.tab_nodes.values():
                    node.show()
                return
            
            # Bola/Slingshot modes - hide individual TABs, show bola
            for node in self.tab_nodes.values():
                node.hide()
            
            if self.bola_node:
                self.bola_node.show()
            
            # Get Buzzard state
            if self.env and self.env.sim:
                buzzard_pos = self.env.sim.mother_drone.position
                buzzard_vel = self.env.sim.mother_drone.velocity
                
                if state == SlingshotState.SWINGING:
                    # Update swing physics
                    self.slingshot.accelerate_swing(torque=3000.0, dt=dt)
                    bola_pos, bola_vel, tension = self.slingshot.update_swing_physics(
                        buzzard_pos, buzzard_vel, dt
                    )
                    
                    # Position bola visual
                    if self.bola_node:
                        self.bola_node.setPos(*bola_pos)
                        
                        # Rotate bola based on swing
                        angle_deg = np.degrees(self.slingshot.swing_angle)
                        self.bola_node.setH(angle_deg)
                    
                    # Draw braided cable from Buzzard to bola
                    self._update_slingshot_cable(buzzard_pos, bola_pos, tension)
                    
                elif state in [SlingshotState.BALLISTIC, SlingshotState.SAILING]:
                    # Update ballistic flight
                    bola_pos = self.slingshot.update_ballistic(dt)
                    
                    if self.bola_node:
                        self.bola_node.setPos(*bola_pos)
                    
                    # Update grid fin visuals
                    self._update_grid_fin_visuals()
                    
                    # No cable in ballistic mode
                    if self.slingshot_cable_node:
                        self.slingshot_cable_node.removeNode()
                        self.slingshot_cable_node = None
                    
                    # Steer toward any active threat
                    if self.env.threat_spawner:
                        threats = self.env.threat_spawner.get_active_threats()
                        if threats:
                            target = threats[0].position
                            self.slingshot.steer_to_target(target)
            
            # Update Buzzard wing extension animation
            if hasattr(self, '_buzzard_wing_target') and self.airfoiled_buzzard:
                self.airfoiled_buzzard.extend_wings(
                    self._buzzard_wing_target, dt, rate=2.0
                )
                self._update_buzzard_wing_visuals()
        
        def _update_slingshot_cable(self, start: np.ndarray, end: np.ndarray, tension: float):
            """Draw the braided cable in slingshot mode."""
            if self.slingshot_cable_node:
                self.slingshot_cable_node.removeNode()
            
            # Color based on tension (more red = higher tension)
            max_tension = 10000  # N
            tension_ratio = min(1.0, tension / max_tension)
            cable_color = (
                0.4 + tension_ratio * 0.5,
                0.4 * (1 - tension_ratio),
                0.2,
                1.0
            )
            
            # Use 3D cable geometry for thick braided cable
            if create_cable_geometry:
                cable_geom = create_cable_geometry(
                    start=start,
                    end=end,
                    radius=0.8,  # Thicker braided cable
                    segments=8,
                    color=cable_color
                )
                if cable_geom:
                    self.slingshot_cable_node = self.render.attachNewNode(cable_geom)
                    return
            
            # Fallback to lines
            lines = LineSegs()
            lines.setThickness(5.0)
            lines.setColor(*cable_color)
            lines.moveTo(Point3(*start))
            lines.drawTo(Point3(*end))
            self.slingshot_cable_node = self.render.attachNewNode(lines.create())
        
        def _update_buzzard_wing_visuals(self):
            """Update Buzzard wing geometry based on extension."""
            # TODO: Create actual wing geometry that scales with extension
            # For now, we'll update mother node scale to hint at wing extension
            if self.mother_node and self.airfoiled_buzzard:
                ext = self.airfoiled_buzzard.wing_extension
                # Scale X (wingspan) based on extension
                self.mother_node.setScale(1.0 + ext * 0.5, 1.0, 1.0)
        
        def _set_showcase(self, mode: ShowcaseMode):
            """Start a showcase demonstration."""
            self.hold.set_showcase(mode)
            self._update_mode_display()
        
        def _stop_showcase(self):
            """Stop showcase, return to training."""
            self.hold.stop_showcase()
            self._update_mode_display()
        
        def _update_mode_display(self):
            """Update the mode indicator in HUD."""
            if "mode" not in self.hud_texts:
                return
            
            state = self.hold.state
            
            # Check slingshot state first
            slingshot_active = (self.slingshot and 
                               self.slingshot.state != SlingshotState.DISPERSED)
            
            if slingshot_active:
                # Slingshot mode - show state
                ss = self.slingshot
                if ss.state == SlingshotState.SWINGING:
                    speed = ss.get_release_velocity()
                    text = f"[SLINGSHOT] SWINGING - {speed:.0f} m/s - Press X to release!"
                    color = (1.0, 0.6, 0.0, 1)  # Orange
                elif ss.state == SlingshotState.BOLA_READY:
                    text = "[SLINGSHOT] BOLA READY - Building momentum..."
                    color = (0.8, 0.8, 0.0, 1)  # Yellow
                elif ss.state == SlingshotState.BALLISTIC:
                    text = "[SLINGSHOT] BALLISTIC - Press Z for grid fins!"
                    color = (0.0, 0.8, 1.0, 1)  # Cyan
                elif ss.state == SlingshotState.SAILING:
                    text = "[SLINGSHOT] SAILING - Grid fins active!"
                    color = (0.0, 1.0, 0.5, 1)  # Green-cyan
                else:
                    text = f"[SLINGSHOT] {ss.state.name}"
                    color = (0.8, 0.6, 0.2, 1)
            elif state.showcase != ShowcaseMode.NONE:
                # Showcase mode
                phase = state.showcase_phase or state.showcase.name
                text = f"[SHOWCASE] {phase}"
                color = (1.0, 0.5, 0.0, 1)  # Orange
            elif state.mode == HoldMode.FULL_PAUSE:
                text = "[PAUSED] Press SPACE to resume"
                color = (1.0, 0.2, 0.2, 1)  # Red
            elif state.mode == HoldMode.AI_PAUSE:
                text = "[AI FROZEN] Environment running - Press F to resume"
                color = (0.2, 0.5, 1.0, 1)  # Blue
            elif state.mode == HoldMode.STEP:
                text = "[STEP] Press . to advance"
                color = (1.0, 1.0, 0.2, 1)  # Yellow
            else:
                # Continuous mode
                speed = state.time_scale
                if self.dreamer_agent:
                    text = f"[TRAINING] DreamerV3 | {speed:.1f}x"
                else:
                    text = f"[TRAINING] Random | {speed:.1f}x"
                color = (0.2, 1.0, 0.2, 1)  # Green
            
            self.hud_texts["mode"].setText(text)
        
        def _setup_lighting(self):
            ambient = AmbientLight("ambient")
            ambient.setColor(Vec4(0.4, 0.4, 0.5, 1))
            self.render.setLight(self.render.attachNewNode(ambient))
            
            sun = DirectionalLight("sun")
            sun.setColor(Vec4(1.0, 0.95, 0.9, 1))
            sun_np = self.render.attachNewNode(sun)
            sun_np.setHpr(45, -45, 0)
            self.render.setLight(sun_np)
        
        def _create_ground(self):
            """Ocean surface"""
            lines = LineSegs()
            lines.setThickness(1.0)
            lines.setColor(0.1, 0.3, 0.5, 0.5)
            
            for i in range(-1000, 1001, 100):
                lines.moveTo(Point3(i, -1000, 0))
                lines.drawTo(Point3(i, 1000, 0))
                lines.moveTo(Point3(-1000, i, 0))
                lines.drawTo(Point3(1000, i, 0))
            
            self.render.attachNewNode(lines.create())
        
        def _create_entities(self):
            """Create Buzzard (mother drone) and TAB airfoils with VISIBLE geometry"""
            
            # ===== BUZZARD - The protected asset =====
            # Use SIMPLE VISIBLE geometry - SCALED UP to be obvious
            buzzard_created = False
            if create_buzzard_geometry:
                try:
                    buzzard_geom = create_buzzard_geometry(
                        length=15.0,   # BIGGER
                        width=6.0,     # WIDER
                        height=4.0,    # TALLER
                        color=(0.2, 0.4, 0.9, 1.0)
                    )
                    if buzzard_geom:
                        self.mother_node = self.render.attachNewNode(buzzard_geom)
                        self.mother_node.setScale(3.0)  # TRIPLE the size!
                        self.mother_node.setTwoSided(True)
                        # Orient so nose points forward (+Y)
                        self.mother_node.setH(90)  # Rotate to face forward
                        buzzard_created = True
                        print(f"[VIZ] Created VISIBLE Buzzard - BIG elongated fuselage, scale 3.0")
                except Exception as e:
                    print(f"[VIZ] Error creating buzzard geometry: {e}")
            
            if not buzzard_created:
                # Use a CardMaker to create a visible BOX instead of sphere
                # This is guaranteed to work
                from panda3d.core import CardMaker
                cm = CardMaker("buzzard_card")
                cm.setFrame(-10, 10, -5, 5)  # Wide card
                cm.setColor(0.2, 0.4, 0.9, 1.0)
                
                self.mother_node = self.render.attachNewNode("buzzard_fallback")
                
                # Add cards on all 6 sides to make a visible BOX
                # Top
                top = self.mother_node.attachNewNode(cm.generate())
                top.setPos(0, 0, 4)
                top.setP(-90)
                # Bottom
                bottom = self.mother_node.attachNewNode(cm.generate())
                bottom.setPos(0, 0, -4)
                bottom.setP(90)
                # Front
                front = self.mother_node.attachNewNode(cm.generate())
                front.setPos(0, 10, 0)
                # Back
                back = self.mother_node.attachNewNode(cm.generate())
                back.setPos(0, -10, 0)
                back.setH(180)
                # Left
                left = self.mother_node.attachNewNode(cm.generate())
                left.setPos(-5, 0, 0)
                left.setH(-90)
                # Right
                right = self.mother_node.attachNewNode(cm.generate())
                right.setPos(5, 0, 0)
                right.setH(90)
                
                self.mother_node.setTwoSided(True)
                print("[VIZ] Created fallback BOX Buzzard (cards)")
            
            # ===== TABs - Flying Wing Airfoils =====
            tab_colors = TAB_COLORS if TAB_COLORS else {
                "UP": (0.2, 1.0, 0.2, 1.0),     # Bright green
                "DOWN": (1.0, 0.2, 0.2, 1.0),   # Bright red
                "LEFT": (1.0, 1.0, 0.2, 1.0),   # Bright yellow
                "RIGHT": (1.0, 0.2, 1.0, 1.0),  # Bright magenta
            }
            
            for tab_id, color in tab_colors.items():
                # Use VISIBLE delta wing geometry - BIG and OBVIOUS
                if create_airfoil_geometry:
                    geom = create_airfoil_geometry(
                        wingspan=12.0,     # BIG wingspan
                        chord=4.0,         # BIG chord
                        thickness=1.2,     # THICK
                        color=color
                    )
                    if geom:
                        node = self.render.attachNewNode(geom)
                        node.setScale(2.5)  # BIGGER!
                        node.setTwoSided(True)
                        # Orient nose forward
                        node.setH(90)
                        self.tab_nodes[tab_id] = node
                        self.cable_nodes[tab_id] = None
                        print(f"[VIZ] Created BIG delta wing for {tab_id}")
                        continue
                
                # Fallback to simple geometry
                geom = create_airfoil_geom(wingspan=8.0, chord=2.5, color=color)
                node = self.render.attachNewNode(geom)
                node.setScale(3.0)
                node.setTwoSided(True)
                self.tab_nodes[tab_id] = node
                self.cable_nodes[tab_id] = None
                print(f"[VIZ] Created fallback airfoil for {tab_id}")
        
        def _create_hud(self):
            """Create training statistics HUD - BUZZARD HEALTH is PRIMARY"""
            
            # ===== BUZZARD HEALTH (CENTER TOP - MOST IMPORTANT) =====
            health_txt = TextNode("buzzard_health")
            health_txt.setText("BUZZARD: 100%")
            health_txt.setAlign(TextNode.ACenter)
            health_np = self.aspect2d.attachNewNode(health_txt)
            health_np.setScale(0.08)
            health_np.setPos(0, 0, 0.85)
            health_np.setColor(0.2, 1.0, 0.2, 1)  # Green when healthy
            self.hud_texts["buzzard_health"] = health_txt
            
            # Episode info (top left)
            texts = [
                ("episode", "Episode: 0", -1.3, 0.9),
                ("step", "Step: 0", -1.3, 0.85),
                ("reward", "Reward: 0.0", -1.3, 0.8),
                ("avg_reward", "Avg Reward: 0.0", -1.3, 0.75),
            ]
            
            for name, text, x, y in texts:
                txt = TextNode(name)
                txt.setText(text)
                txt.setAlign(TextNode.ALeft)
                txt_np = self.aspect2d.attachNewNode(txt)
                txt_np.setScale(0.05)
                txt_np.setPos(x, 0, y)
                txt_np.setColor(1, 1, 1, 1)
                self.hud_texts[name] = txt
            
            # Defense stats (top right)
            explore_texts = [
                ("intercepts", "Kills: 0", 0.7, 0.9),
                ("sacrificed", "Sacrificed: 0", 0.7, 0.85),
                ("threats", "Threats: 0", 0.7, 0.8),
                ("impacts", "Hits Taken: 0", 0.7, 0.75),
            ]
            
            for name, text, x, y in explore_texts:
                txt = TextNode(name)
                txt.setText(text)
                txt.setAlign(TextNode.ALeft)
                txt_np = self.aspect2d.attachNewNode(txt)
                txt_np.setScale(0.05)
                txt_np.setPos(x, 0, y)
                txt_np.setColor(0.2, 1.0, 0.5, 1)
                self.hud_texts[name] = txt
            
            # TAB status (bottom) - AIRFOIL DEFENDERS
            for i, tab_id in enumerate(["UP", "DOWN", "LEFT", "RIGHT"]):
                txt = TextNode(f"tab_{tab_id}")
                txt.setText(f"{tab_id}: READY")
                txt.setAlign(TextNode.ACenter)
                txt_np = self.aspect2d.attachNewNode(txt)
                txt_np.setScale(0.04)
                txt_np.setPos(-0.6 + i * 0.4, 0, -0.85)
                self.hud_texts[f"tab_{tab_id}"] = txt
            
            # Mode indicator (updated by _update_mode_display)
            mode_txt = TextNode("mode")
            mode_txt.setText("[TRAINING] Initializing...")
            mode_txt.setAlign(TextNode.ACenter)
            mode_np = self.aspect2d.attachNewNode(mode_txt)
            mode_np.setScale(0.06)
            mode_np.setPos(0, 0, 0.95)
            mode_np.setColor(1, 0.8, 0.2, 1)
            self.hud_texts["mode"] = mode_txt
            
            # Camera mode indicator (bottom right)
            cam_txt = TextNode("camera")
            cam_txt.setText("CAM: CHASE")
            cam_txt.setAlign(TextNode.ARight)
            cam_np = self.aspect2d.attachNewNode(cam_txt)
            cam_np.setScale(0.05)
            cam_np.setPos(1.3, 0, -0.9)
            cam_np.setColor(0.7, 0.9, 1.0, 1)
            self.hud_texts["camera"] = cam_txt
        
        def _create_control_panel(self):
            """Create showcase selection help text (bottom right)."""
            help_lines = [
                "SHOWCASES:",
                "1-Systems 2-Defense 3-Intercept",
                "4-Extend  5-Corkscrew 6-Formation",
                "7-Throttle 8-Evasive 9-Sacrifice",
                "",
                "SPACE-Pause F-AI Freeze .-Step",
            ]
            
            for i, line in enumerate(help_lines):
                txt = TextNode(f"help_{i}")
                txt.setText(line)
                txt.setAlign(TextNode.ARight)
                txt_np = self.aspect2d.attachNewNode(txt)
                txt_np.setScale(0.035)
                txt_np.setPos(1.3, 0, -0.5 - i * 0.05)
                txt_np.setColor(0.7, 0.7, 0.7, 0.8)
        
        def _reset_episode(self):
            """Reset to new episode"""
            self.obs, info = self.env.reset()
            self.episode_reward = 0
            self.episode_count += 1
            print(f"\n[EPISODE {self.episode_count}] Starting...")
        
        def _update_task(self, task):
            """Main training update loop"""
            if self.obs is None:
                return Task.cont
            
            # Check HOLD state - should we step?
            if not self.hold.should_step():
                # Paused - still update visuals but don't advance simulation
                return Task.cont
            
            # Get AI action (if not frozen)
            ai_action = None
            if self.hold.should_ai_act():
                if self.dreamer_agent is not None:
                    # DreamerV3 inference
                    obs_dict = self._obs_to_dict(self.obs)
                    ai_action = self.dreamer_agent.infer(obs_dict)
                    ai_action = np.clip(ai_action, -1.0, 1.0)
                    if len(ai_action) < 18:
                        ai_action = np.concatenate([ai_action, np.zeros(18 - len(ai_action))])
                    elif len(ai_action) > 18:
                        ai_action = ai_action[:18]
                else:
                    # Random baseline when no DreamerV3
                    ai_action = self.env.action_space.sample()
            
            # Get final action (showcase takes priority, or AI, or last action)
            action = self.hold.get_action(self.obs, ai_action)
            
            # Apply time scaling
            # (Note: this affects how many steps we skip, not physics dt)
            
            # Step the environment
            self.obs, reward, terminated, truncated, info = self.env.step(action)
            self.episode_reward += reward
            self.total_steps += 1
            
            # Update the showcase phase display
            if self.hold.state.showcase != ShowcaseMode.NONE:
                self.hold.state.showcase_phase = self.hold.showcase.phase
                self._update_mode_display()
            
            # Update visuals
            self._update_entities()
            self._update_threats()
            self._update_hud(info)
            
            # Update slingshot mode visuals
            dt = globalClock.getDt()
            self._update_slingshot_visuals(dt)
            
            # Episode end
            if terminated or truncated:
                self.recent_rewards.append(self.episode_reward)
                self.recent_intercepts.append(info.get('intercepts', 0))
                print(f"[EPISODE {self.episode_count}] "
                      f"Reward: {self.episode_reward:.1f} | "
                      f"Intercepts: {info.get('intercepts', 0)} | "
                      f"Novel: {info.get('novel_states', 0)}")
                self._reset_episode()
            
            return Task.cont
        
        def _update_entities(self):
            """Update visual positions"""
            sim = self.env.sim
            if sim is None:
                return
            
            # Mother drone
            md_pos = sim.mother_drone.position
            self.mother_node.setPos(md_pos[0], md_pos[1], md_pos[2])
            
            # Camera follows
            self.camera_target = md_pos.copy()
            
            # TABs - these are AIRFOILS that FLY!
            for tab_id, node in self.tab_nodes.items():
                tab = sim.tab_array.tabs[tab_id]
                pos = tab.position
                vel = tab.velocity
                node.setPos(pos[0], pos[1], pos[2])
                
                # Orient airfoil to face direction of travel
                # This makes them look like they're actually FLYING
                if np.linalg.norm(vel) > 0.1:
                    # Calculate heading from velocity
                    heading = np.degrees(np.arctan2(vel[0], vel[1]))
                    pitch = np.degrees(np.arctan2(vel[2], np.sqrt(vel[0]**2 + vel[1]**2)))
                    node.setHpr(heading, pitch, 0)
                
                # Gray out if detached
                if not tab.is_attached:
                    node.setColorScale(0.5, 0.5, 0.5, 0.5)
                else:
                    node.setColorScale(1, 1, 1, 1)
                
                # Update cable with VISIBLE thick lines
                if self.cable_nodes[tab_id]:
                    self.cable_nodes[tab_id].removeNode()
                
                if tab.is_attached:
                    # Calculate cable tension for color
                    cable_length = np.linalg.norm(pos - md_pos)
                    max_length = 60.0  # Approximate max cable length
                    tension_ratio = min(1.0, cable_length / max_length)
                    
                    # Color transitions from green (slack) to red (taut)
                    cable_color = (
                        0.3 + tension_ratio * 0.6,   # More red when taut
                        0.6 * (1 - tension_ratio),   # Less green when taut
                        0.2,
                        1.0
                    )
                    
                    # Draw THICK visible cable lines
                    lines = LineSegs("cable")
                    lines.setThickness(6.0)  # VERY THICK
                    lines.setColor(*cable_color)
                    lines.moveTo(Point3(float(md_pos[0]), float(md_pos[1]), float(md_pos[2])))
                    lines.drawTo(Point3(float(pos[0]), float(pos[1]), float(pos[2])))
                    self.cable_nodes[tab_id] = self.render.attachNewNode(lines.create())
        
        def _update_threats(self):
            """Update threat visuals"""
            # Remove old
            for node in self.threat_nodes.values():
                node.removeNode()
            self.threat_nodes.clear()
            
            # Add current
            if self.env.threat_spawner is None:
                return
            
            for threat in self.env.threat_spawner.get_active_threats():
                # Color by type
                if threat.profile.type == ThreatType.IR_MISSILE:
                    color = (1.0, 0.3, 0.1, 1)
                    size = 2
                elif threat.profile.type == ThreatType.RADAR_MISSILE:
                    color = (1.0, 0.1, 0.1, 1)
                    size = 2.5
                elif threat.profile.type == ThreatType.ATTACK_DRONE:
                    color = (0.8, 0.4, 0.8, 1)
                    size = 3
                elif threat.profile.type == ThreatType.SWARM_ELEMENT:
                    color = (1.0, 0.6, 0.0, 1)
                    size = 1
                else:
                    color = (0.7, 0.7, 0.7, 1)
                    size = 2
                
                geom = create_sphere_geom(radius=size, color=color)
                node = self.render.attachNewNode(geom)
                pos = threat.position
                node.setPos(pos[0], pos[1], pos[2])
                self.threat_nodes[threat.threat_id] = node
        
        def _update_hud(self, info: Dict):
            """Update HUD texts - BUZZARD HEALTH is PRIMARY"""
            
            # ===== BUZZARD HEALTH (most important) =====
            health = info.get('buzzard_health', 100.0)
            self.hud_texts["buzzard_health"].setText(f"BUZZARD: {health:.0f}%")
            
            # Color code health: green -> yellow -> red
            if health > 70:
                # Green - healthy
                pass  # Color is set by NodePath, need to update differently
            elif health > 30:
                # Yellow - damaged
                pass
            else:
                # Red - critical
                pass
            
            # Episode stats
            self.hud_texts["episode"].setText(f"Episode: {self.episode_count}")
            self.hud_texts["step"].setText(f"Step: {info.get('step', 0)}")
            self.hud_texts["reward"].setText(f"Reward: {self.episode_reward:.1f}")
            
            if self.recent_rewards:
                avg = np.mean(self.recent_rewards)
                self.hud_texts["avg_reward"].setText(f"Avg Reward (100): {avg:.1f}")
            
            # Defense stats
            self.hud_texts["intercepts"].setText(f"Kills: {info.get('intercepts', 0)}")
            self.hud_texts["sacrificed"].setText(f"Sacrificed: {info.get('tabs_sacrificed', 0)}")
            self.hud_texts["threats"].setText(f"Threats: {info.get('threats_active', 0)}")
            self.hud_texts["impacts"].setText(f"Hits Taken: {info.get('impacts', 0)}")
            
            # TAB status - AIRFOIL DEFENDERS
            sim = self.env.sim
            if sim:
                for tab_id in ["UP", "DOWN", "LEFT", "RIGHT"]:
                    tab = sim.tab_array.tabs[tab_id]
                    if tab.is_attached:
                        status = "READY"
                    else:
                        status = "LAUNCHED"
                    self.hud_texts[f"tab_{tab_id}"].setText(f"{tab_id}: {status}")
        
        def _camera_task(self, task):
            """Camera control with cinematic modes"""
            dt = globalClock.getDt()
            
            # Handle mouse orbit for FREE mode
            if self.cinema_cam.mode == CameraMode.FREE:
                if self.mouse_dragging and self.mouseWatcherNode.hasMouse():
                    mx = self.mouseWatcherNode.getMouseX()
                    my = self.mouseWatcherNode.getMouseY()
                    
                    dx = (mx - self.last_mouse_x) * 200
                    dy = (my - self.last_mouse_y) * 200
                    
                    self.camera_heading += dx
                    self.camera_pitch = np.clip(self.camera_pitch + dy, -85, 85)
                    
                    self.last_mouse_x = mx
                    self.last_mouse_y = my
                
                # Update position using orbit math
                rad_h = np.radians(self.camera_heading)
                rad_p = np.radians(self.camera_pitch)
                
                x = self.camera_distance * np.cos(rad_p) * np.sin(rad_h)
                y = self.camera_distance * np.cos(rad_p) * np.cos(rad_h)
                z = self.camera_distance * np.sin(rad_p)
                
                cam_pos = self.camera_target + np.array([x, -y, z])
                self.camera.setPos(cam_pos[0], cam_pos[1], cam_pos[2])
                self.camera.lookAt(Point3(*self.camera_target))
            else:
                # Use cinematic camera system for all other modes
                # Get Buzzard position and velocity
                buzzard_pos = self.camera_target  # Updated in _update_entities
                buzzard_vel = np.array([0, 50, 0])  # Default forward
                
                # Try to get actual velocity from simulation
                if self.env and self.env.sim:
                    buzzard_vel = self.env.sim.mother_drone.velocity
                
                # Get threat positions for threat-aware camera
                threats = []
                if self.env and self.env.threat_spawner:
                    threats = self.env.threat_spawner.get_active_threats()
                
                # Update cinematic camera
                cam_pos, look_at, fov = self.cinema_cam.update(
                    target_pos=buzzard_pos,
                    target_vel=buzzard_vel,
                    dt=dt,
                    threats=threats
                )
                
                # Apply to Panda3D camera
                self.camera.setPos(cam_pos[0], cam_pos[1], cam_pos[2])
                self.camera.lookAt(Point3(*look_at))
                
                # Update FOV if needed
                if hasattr(self, 'camLens'):
                    self.camLens.setFov(fov)
            
            return Task.cont
        
        def _start_drag(self):
            if self.mouseWatcherNode.hasMouse():
                self.mouse_dragging = True
                self.last_mouse_x = self.mouseWatcherNode.getMouseX()
                self.last_mouse_y = self.mouseWatcherNode.getMouseY()
        
        def _stop_drag(self):
            self.mouse_dragging = False
        
        def _zoom_in(self):
            self.camera_distance = max(30, self.camera_distance * 0.85)
        
        def _zoom_out(self):
            self.camera_distance = min(500, self.camera_distance * 1.15)
        
        def _obs_to_dict(self, obs: np.ndarray) -> Dict:
            """Convert flat observation to dict for DreamerV3"""
            return {
                'mother_drone': {
                    'position': obs[0:3] * 1000,
                    'velocity': obs[3:6] * 100,
                },
                'tabs': {},
                'threats': []
            }


def run_visual_training(use_dreamer: bool = True, explore: bool = True):
    """
    Launch visual training mode.
    
    Args:
        use_dreamer: Load DreamerV3 weights (gen42)
        explore: Use ExplorationController instead of frozen inference
                 NOTE: DreamerV3 gen42 was NOT trained on KAPS physics.
                 Set explore=True to actually see the airfoils demonstrate
                 what they can do!
    """
    if not PANDA3D_AVAILABLE:
        print("Panda3D not available!")
        return
    
    print("=" * 70)
    print("KAPS VISUAL TRAINING with HOLD Control")
    print("=" * 70)
    
    dreamer_agent = None
    if use_dreamer:
        try:
            from src.ai.dreamer_interface import DreamerBrainInterface
            dreamer_agent = DreamerBrainInterface()
            print("[✓] DreamerV3 agent loaded (trainable)")
        except Exception as e:
            print(f"[!] Could not load DreamerV3: {e}")
            print("[!] Running with random baseline")
    
    print("")
    print("Interactive HOLD controls enabled:")
    print("  - SPACE to pause/resume")
    print("  - F to freeze AI (environment continues)")
    print("  - 1-9 for showcase demonstrations")
    print("=" * 70)
    
    trainer = VisualTrainer(dreamer_agent=dreamer_agent)
    trainer._update_mode_display()  # Set initial mode text
    trainer.run()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="KAPS Visual Training with HOLD Control")
    parser.add_argument("--dreamer", action="store_true",
                       help="Load DreamerV3 agent for training")
    parser.add_argument("--random", action="store_true", 
                       help="Use random actions only (no DreamerV3)")
    args = parser.parse_args()
    
    run_visual_training(use_dreamer=args.dreamer and not args.random)

