"""
Panda3D Visualization Renderer
==============================
3D visualization of the KAPS system using Panda3D.

Renders:
- Mother drone with buzzard-wing geometry
- TABs in cross-formation
- Tether cables with tension coloring
- Threat indicators
- Defensive bubble visualization
"""

import numpy as np
from typing import Dict, Optional, List

# Note: Panda3D imports will fail if not installed
# This module is optional for visualization
try:
    from direct.showbase.ShowBase import ShowBase
    from panda3d.core import (
        Vec3, Vec4, Point3, 
        LineSegs, NodePath, GeomNode,
        AmbientLight, DirectionalLight,
        TextNode, CardMaker,
        GeomVertexFormat, GeomVertexData, GeomVertexWriter,
        Geom, GeomTriangles, ClockObject
    )
    from direct.task import Task
    globalClock = ClockObject.getGlobalClock()
    PANDA3D_AVAILABLE = True
except ImportError:
    PANDA3D_AVAILABLE = False
    print("Panda3D not installed. Visualization disabled.")
    print("Install with: pip install panda3d")


def create_sphere_geom(radius: float = 1.0, color: tuple = (1, 1, 1, 1), segments: int = 12) -> GeomNode:
    """Create a sphere geometry procedurally"""
    import numpy as np
    format = GeomVertexFormat.getV3n3c4()
    vdata = GeomVertexData("sphere", format, Geom.UHStatic)
    
    vertex = GeomVertexWriter(vdata, "vertex")
    normal = GeomVertexWriter(vdata, "normal")
    col = GeomVertexWriter(vdata, "color")
    
    # Generate sphere vertices
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


def create_box_geom(sx: float, sy: float, sz: float, color: tuple = (1, 1, 1, 1)) -> GeomNode:
    """Create a box geometry procedurally"""
    format = GeomVertexFormat.getV3n3c4()
    vdata = GeomVertexData("box", format, Geom.UHStatic)
    
    vertex = GeomVertexWriter(vdata, "vertex")
    normal = GeomVertexWriter(vdata, "normal")
    col = GeomVertexWriter(vdata, "color")
    
    # 8 corners
    hx, hy, hz = sx/2, sy/2, sz/2
    corners = [
        (-hx, -hy, -hz), (hx, -hy, -hz), (hx, hy, -hz), (-hx, hy, -hz),
        (-hx, -hy, hz), (hx, -hy, hz), (hx, hy, hz), (-hx, hy, hz)
    ]
    
    # 6 faces (each with 4 verts)
    faces = [
        ([0,1,2,3], (0,0,-1)),  # bottom
        ([4,7,6,5], (0,0,1)),   # top
        ([0,4,5,1], (0,-1,0)),  # front
        ([2,6,7,3], (0,1,0)),   # back
        ([0,3,7,4], (-1,0,0)),  # left
        ([1,5,6,2], (1,0,0)),   # right
    ]
    
    for indices, n in faces:
        for idx in indices:
            vertex.addData3f(*corners[idx])
            normal.addData3f(*n)
            col.addData4f(*color)
    
    prim = GeomTriangles(Geom.UHStatic)
    for i in range(6):
        base = i * 4
        prim.addVertices(base, base+1, base+2)
        prim.addVertices(base, base+2, base+3)
    
    geom = Geom(vdata)
    geom.addPrimitive(prim)
    node = GeomNode("box")
    node.addGeom(geom)
    return node


if PANDA3D_AVAILABLE:
    
    class KAPSVisualizer(ShowBase):
        """
        Panda3D-based 3D visualizer for the KAPS simulation.
        
        Provides real-time rendering of:
        - Drone geometries
        - Tether cables
        - Formation patterns
        - Threat trajectories
        - Defensive intercepts
        """
        
        def __init__(self, simulation=None):
            ShowBase.__init__(self)
            
            self.simulation = simulation
            
            # Camera state for orbital controls
            self.camera_distance = 150
            self.camera_heading = 45  # degrees
            self.camera_pitch = 25    # degrees
            self.camera_target = np.array([0.0, 0.0, 1000.0])  # Follow target
            self.camera_follow = True
            
            # Mouse state
            self.mouse_dragging = False
            self.last_mouse_x = 0
            self.last_mouse_y = 0
            
            # Camera setup
            self.disableMouse()
            self._update_camera_position()
            
            # Background color - sky blue
            self.setBackgroundColor(0.4, 0.6, 0.85)
            
            # Lighting
            self._setup_lighting()
            
            # Create visual nodes
            self.mother_drone_node = None
            self.tab_nodes: Dict[str, NodePath] = {}
            self.cable_nodes: Dict[str, NodePath] = {}
            self.threat_nodes: Dict[str, NodePath] = {}
            self.velocity_arrow = None
            self.grid_node = None
            
            # UI elements
            self.status_text = None
            self.hud_texts = {}
            
            # Initialize geometry
            self._create_ground()
            self._create_grid()
            self._create_mother_drone()
            self._create_tabs()
            self._create_cables()
            self._create_velocity_arrow()
            self._create_ui()
            
            # Add update task
            self.taskMgr.add(self._update_task, "update_simulation")
            self.taskMgr.add(self._camera_control_task, "camera_control")
            
            # Keyboard controls
            self.accept("escape", self.userExit)
            self.accept("space", self._inject_threat)
            self.accept("b", self._trigger_burst)
            self.accept("r", self._reset_camera)
            self.accept("f", self._toggle_follow)
            self.accept("wheel_up", self._zoom_in)
            self.accept("wheel_down", self._zoom_out)
            
            # Mouse controls
            self.accept("mouse1", self._start_drag)
            self.accept("mouse1-up", self._stop_drag)
            
            # WASD camera
            self.key_map = {"w": False, "s": False, "a": False, "d": False, "q": False, "e": False}
            for key in self.key_map:
                self.accept(key, self._set_key, [key, True])
                self.accept(f"{key}-up", self._set_key, [key, False])
            
            print("=" * 50)
            print("KAPS VISUALIZER - CONTROLS")
            print("=" * 50)
            print("  Mouse Drag  - Orbit camera")
            print("  Scroll      - Zoom in/out")
            print("  WASD        - Pan camera")
            print("  Q/E         - Raise/lower view")
            print("  F           - Toggle follow mode")
            print("  R           - Reset camera")
            print("  SPACE       - Inject threat")
            print("  B           - Speed burst")
            print("  ESC         - Exit")
            print("=" * 50)
        
        def _set_key(self, key, value):
            self.key_map[key] = value
        
        def _update_camera_position(self):
            """Update camera based on orbital parameters"""
            rad_h = np.radians(self.camera_heading)
            rad_p = np.radians(self.camera_pitch)
            
            # Spherical to cartesian offset from target
            x = self.camera_distance * np.cos(rad_p) * np.sin(rad_h)
            y = self.camera_distance * np.cos(rad_p) * np.cos(rad_h)
            z = self.camera_distance * np.sin(rad_p)
            
            cam_pos = self.camera_target + np.array([x, -y, z])
            self.camera.setPos(cam_pos[0], cam_pos[1], cam_pos[2])
            self.camera.lookAt(Point3(*self.camera_target))
        
        def _camera_control_task(self, task):
            """Handle camera controls each frame"""
            dt = globalClock.getDt()
            
            # Mouse drag for orbit - NOT INVERTED
            # Drag left = camera orbits left, drag up = look from above
            if self.mouse_dragging and self.mouseWatcherNode.hasMouse():
                mx = self.mouseWatcherNode.getMouseX()
                my = self.mouseWatcherNode.getMouseY()
                
                dx = (mx - self.last_mouse_x) * 200
                dy = (my - self.last_mouse_y) * 200
                
                # Drag left (dx<0) = heading decreases = camera moves left
                # Drag up (dy>0) = pitch increases = camera goes higher
                self.camera_heading += dx
                self.camera_pitch = np.clip(self.camera_pitch + dy, -85, 85)
                
                self.last_mouse_x = mx
                self.last_mouse_y = my
            
            # Scroll for zoom only - no WASD pan since we stay centered
            self._update_camera_position()
            return Task.cont
        
        def _start_drag(self):
            if self.mouseWatcherNode.hasMouse():
                self.mouse_dragging = True
                self.last_mouse_x = self.mouseWatcherNode.getMouseX()
                self.last_mouse_y = self.mouseWatcherNode.getMouseY()
                self.camera_follow = False  # Disable follow when manually moving
        
        def _stop_drag(self):
            self.mouse_dragging = False
        
        def _zoom_in(self):
            self.camera_distance = max(20, self.camera_distance * 0.85)
            self._update_camera_position()
        
        def _zoom_out(self):
            self.camera_distance = min(1000, self.camera_distance * 1.15)
            self._update_camera_position()
        
        def _toggle_follow(self):
            self.camera_follow = not self.camera_follow
            print(f"Follow mode: {'ON' if self.camera_follow else 'OFF'}")
        
        def _create_grid(self):
            """Create reference grid at altitude"""
            lines = LineSegs()
            lines.setThickness(1.0)
            lines.setColor(0.3, 0.4, 0.5, 0.5)
            
            grid_size = 500
            spacing = 50
            z = 950  # Just below normal flight altitude
            
            for i in range(-grid_size, grid_size + 1, spacing):
                lines.moveTo(Point3(i, -grid_size, z))
                lines.drawTo(Point3(i, grid_size, z))
                lines.moveTo(Point3(-grid_size, i, z))
                lines.drawTo(Point3(grid_size, i, z))
            
            self.grid_node = self.render.attachNewNode(lines.create())
        
        def _create_velocity_arrow(self):
            """Create velocity direction indicator"""
            lines = LineSegs()
            lines.setThickness(3.0)
            lines.setColor(1.0, 0.5, 0.0, 1)  # Orange
            lines.moveTo(Point3(0, 0, 0))
            lines.drawTo(Point3(20, 0, 0))  # Will be updated
            self.velocity_arrow = self.render.attachNewNode(lines.create())
        
        def _create_ground(self):
            """Create ocean/ground reference plane"""
            import numpy as np
            # Large ground plane at z=0
            ground_geom = create_box_geom(5000, 5000, 1, color=(0.1, 0.3, 0.5, 1))
            self.ground = self.render.attachNewNode(ground_geom)
            self.ground.setPos(0, 0, -0.5)  # Ocean at z=0
            
        def _setup_lighting(self):
            """Setup scene lighting"""
            # Ambient light
            ambient = AmbientLight("ambient")
            ambient.setColor(Vec4(0.3, 0.3, 0.4, 1))
            ambient_np = self.render.attachNewNode(ambient)
            self.render.setLight(ambient_np)
            
            # Directional light (sun)
            sun = DirectionalLight("sun")
            sun.setColor(Vec4(0.9, 0.9, 0.8, 1))
            sun_np = self.render.attachNewNode(sun)
            sun_np.setHpr(45, -45, 0)
            self.render.setLight(sun_np)
        
        def _create_mother_drone(self):
            """Create mother drone geometry - The Buzzard"""
            # Main body - sphere
            body_geom = create_sphere_geom(radius=3, color=(0.2, 0.3, 0.8, 1))
            self.mother_drone_node = self.render.attachNewNode(body_geom)
            
            # Wing - flat box
            wing_geom = create_box_geom(16, 1, 0.2, color=(0.3, 0.4, 0.9, 1))
            wing = self.mother_drone_node.attachNewNode(wing_geom)
            
            # Fuselage
            fuse_geom = create_box_geom(2, 8, 2, color=(0.25, 0.35, 0.85, 1))
            fuse = self.mother_drone_node.attachNewNode(fuse_geom)
        
        def _create_tabs(self):
            """Create TAB geometry for all four positions"""
            tab_colors = {
                "UP": (0.2, 0.8, 0.2, 1),      # Green
                "DOWN": (0.8, 0.2, 0.2, 1),    # Red
                "LEFT": (0.8, 0.8, 0.2, 1),    # Yellow
                "RIGHT": (0.8, 0.2, 0.8, 1),   # Purple
            }
            
            for tab_id, color in tab_colors.items():
                # TAB body
                body_geom = create_sphere_geom(radius=1.0, color=color)
                node = self.render.attachNewNode(body_geom)
                
                # TAB wing
                wing_geom = create_box_geom(3, 0.4, 0.1, color=color)
                wing = node.attachNewNode(wing_geom)
                
                self.tab_nodes[tab_id] = node
        
        def _create_cables(self):
            """Create cable line geometry"""
            for tab_id in ["UP", "DOWN", "LEFT", "RIGHT"]:
                # Will be updated each frame
                self.cable_nodes[tab_id] = None
        
        def _update_cables(self, mother_pos: np.ndarray, tab_positions: Dict):
            """Update cable line geometry"""
            for tab_id, cable_node in self.cable_nodes.items():
                # Remove old cable
                if cable_node:
                    cable_node.removeNode()
                
                if tab_id not in tab_positions:
                    continue
                
                tab_pos = tab_positions[tab_id]
                
                # Draw line from mother to TAB
                lines = LineSegs()
                lines.setThickness(2.0)
                lines.setColor(0.5, 0.5, 0.5, 1)  # Gray
                
                lines.moveTo(Point3(*mother_pos))
                lines.drawTo(Point3(*tab_pos))
                
                self.cable_nodes[tab_id] = self.render.attachNewNode(lines.create())
        
        def _create_ui(self):
            """Create comprehensive HUD"""
            # Main status bar (top left)
            self.status_text = TextNode("status")
            self.status_text.setText("KAPS Simulation")
            self.status_text.setAlign(TextNode.ALeft)
            text_np = self.aspect2d.attachNewNode(self.status_text)
            text_np.setScale(0.05)
            text_np.setPos(-1.3, 0, 0.9)
            
            # TAB status panel (right side)
            tab_labels = ["UP", "DOWN", "LEFT", "RIGHT"]
            colors = [(0.2, 0.8, 0.2, 1), (0.8, 0.2, 0.2, 1), (0.8, 0.8, 0.2, 1), (0.8, 0.2, 0.8, 1)]
            for i, (label, col) in enumerate(zip(tab_labels, colors)):
                txt = TextNode(f"tab_{label}")
                txt.setText(f"{label}: --")
                txt.setAlign(TextNode.ALeft)
                txt_np = self.aspect2d.attachNewNode(txt)
                txt_np.setScale(0.04)
                txt_np.setPos(0.8, 0, 0.85 - i * 0.08)
                txt_np.setColor(*col)
                self.hud_texts[f"tab_{label}"] = txt
            
            # Physics info (bottom left)
            physics_labels = ["position", "velocity", "formation", "cables"]
            for i, label in enumerate(physics_labels):
                txt = TextNode(f"phys_{label}")
                txt.setText(f"{label}: --")
                txt.setAlign(TextNode.ALeft)
                txt_np = self.aspect2d.attachNewNode(txt)
                txt_np.setScale(0.035)
                txt_np.setPos(-1.3, 0, -0.7 - i * 0.06)
                txt_np.setColor(0.9, 0.9, 0.9, 1)
                self.hud_texts[f"phys_{label}"] = txt
            
            # Camera info (bottom right)
            cam_txt = TextNode("camera_info")
            cam_txt.setText("Camera: --")
            cam_txt.setAlign(TextNode.ARight)
            cam_np = self.aspect2d.attachNewNode(cam_txt)
            cam_np.setScale(0.035)
            cam_np.setPos(1.3, 0, -0.9)
            cam_np.setColor(0.7, 0.7, 0.8, 1)
            self.hud_texts["camera"] = cam_txt
            
            # Controls hint
            hint_txt = TextNode("hint")
            hint_txt.setText("Mouse:Orbit | Scroll:Zoom | WASD:Pan | F:Follow | R:Reset | SPACE:Threat")
            hint_txt.setAlign(TextNode.ACenter)
            hint_np = self.aspect2d.attachNewNode(hint_txt)
            hint_np.setScale(0.03)
            hint_np.setPos(0, 0, -0.95)
            hint_np.setColor(0.6, 0.6, 0.7, 1)
        
        def _update_task(self, task):
            """Main update loop"""
            if self.simulation is None:
                return Task.cont
            
            # Step simulation
            telemetry = self.simulation.step()
            
            # Update mother drone
            md_pos = telemetry['mother_drone']['position']
            md_vel = telemetry['mother_drone'].get('velocity', np.zeros(3))
            self.mother_drone_node.setPos(md_pos[0], md_pos[1], md_pos[2])
            
            # Orient mother drone in direction of travel
            if np.linalg.norm(md_vel) > 0.1:
                heading = np.degrees(np.arctan2(md_vel[1], md_vel[0]))
                self.mother_drone_node.setH(heading)
            
            # Update velocity arrow
            if self.velocity_arrow:
                self.velocity_arrow.removeNode()
            arrow_lines = LineSegs()
            arrow_lines.setThickness(4.0)
            arrow_lines.setColor(1.0, 0.5, 0.0, 1)
            vel_scale = 0.5
            arrow_lines.moveTo(Point3(md_pos[0], md_pos[1], md_pos[2]))
            arrow_lines.drawTo(Point3(
                md_pos[0] + md_vel[0] * vel_scale,
                md_pos[1] + md_vel[1] * vel_scale,
                md_pos[2] + md_vel[2] * vel_scale
            ))
            self.velocity_arrow = self.render.attachNewNode(arrow_lines.create())
            
            # Update TABs
            tab_positions = {}
            for tab_id, tab_data in telemetry['tabs'].items():
                if tab_id in self.tab_nodes:
                    pos = tab_data['position']
                    tab_positions[tab_id] = pos
                    self.tab_nodes[tab_id].setPos(pos[0], pos[1], pos[2])
                    
                    # Update TAB HUD
                    if f"tab_{tab_id}" in self.hud_texts:
                        cable_info = telemetry['cables'].get(tab_id, {})
                        tension = cable_info.get('tension', 0)
                        attached = "●" if tab_data.get('attached', True) else "○"
                        self.hud_texts[f"tab_{tab_id}"].setText(
                            f"{attached} {tab_id}: T={tension:.0f}N"
                        )
            
            # Update cables
            self._update_cables(
                np.array([md_pos[0], md_pos[1], md_pos[2]]),
                tab_positions
            )
            
            # Camera follow mode
            # ALWAYS center on Buzzard
            self.camera_target = np.array([md_pos[0], md_pos[1], md_pos[2]])
            self._update_camera_position()
            
            # Update grid position to follow
            if self.grid_node:
                self.grid_node.setPos(md_pos[0], md_pos[1], 0)
            
            # Update main status
            status = (
                f"T={self.simulation.time:.1f}s | "
                f"Speed: {telemetry['mother_drone']['speed']:.1f} m/s | "
                f"Alt: {telemetry['mother_drone']['altitude']:.0f}m | "
                f"TABs: {self.simulation.tab_array.count_attached()}/4 | "
                f"Alert: {telemetry['defense']['alert_level']}"
            )
            self.status_text.setText(status)
            
            # Update physics HUD
            if "phys_position" in self.hud_texts:
                self.hud_texts["phys_position"].setText(
                    f"Pos: X={md_pos[0]:.0f} Y={md_pos[1]:.0f} Z={md_pos[2]:.0f}m"
                )
            if "phys_velocity" in self.hud_texts:
                self.hud_texts["phys_velocity"].setText(
                    f"Vel: {telemetry['mother_drone']['speed']:.1f} m/s  Heading: {np.degrees(np.arctan2(md_vel[1], md_vel[0])):.0f}°"
                )
            if "phys_formation" in self.hud_texts:
                form_status = telemetry.get('formation', {})
                self.hud_texts["phys_formation"].setText(
                    f"Formation: {form_status.get('mode', 'CROSS')} | Spread: {form_status.get('spread', 30):.0f}m"
                )
            if "phys_cables" in self.hud_texts:
                total_tension = sum(
                    telemetry['cables'].get(tid, {}).get('tension', 0) 
                    for tid in ["UP", "DOWN", "LEFT", "RIGHT"]
                )
                self.hud_texts["phys_cables"].setText(
                    f"Total Cable Tension: {total_tension:.0f}N"
                )
            
            # Camera info
            if "camera" in self.hud_texts:
                mode = "FOLLOW" if self.camera_follow else "FREE"
                self.hud_texts["camera"].setText(
                    f"Cam: {mode} | Dist: {self.camera_distance:.0f}m | H:{self.camera_heading:.0f}° P:{self.camera_pitch:.0f}°"
                )
            
            return Task.cont
        
        def _inject_threat(self):
            """Inject threat via keyboard"""
            if self.simulation:
                self.simulation.inject_threat()
                print("[!] Threat injected!")
        
        def _trigger_burst(self):
            """Trigger speed burst via keyboard"""
            if self.simulation:
                self.simulation.execute_speed_burst()
                print("[!] SPEED BURST - All cables released!")
        
        def _reset_camera(self):
            """Reset camera position"""
            self.camera_distance = 150
            self.camera_heading = 45
            self.camera_pitch = 25
            self.camera_follow = True
            print("[i] Camera reset")


def run_visualization():
    """Run the Panda3D visualization"""
    if not PANDA3D_AVAILABLE:
        print("Cannot run visualization: Panda3D not installed")
        return
    
    # Import simulation
    from ..main import KAPSSimulation
    
    # Create simulation
    sim = KAPSSimulation()
    
    # Create and run visualizer
    viz = KAPSVisualizer(sim)
    viz.run()


if __name__ == "__main__":
    run_visualization()
