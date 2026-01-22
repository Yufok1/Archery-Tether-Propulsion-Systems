"""
COMBAT VISUALIZER - 3D ASTEROID DEFENSE
========================================

DOPE ASS visualization of the quine lattice defending against asteroids.

Features:
- Full 3D Panda3D rendering
- Spinning blade-tethers with visible flat surfaces
- Asteroids/missiles spawning from all directions
- Particle explosions on intercepts
- DreamerV3 brain controlling the lattice
- Score, wave count, damage HUD
- Camera follows the action

This is what happens when a DreamerV3 quine goes to war.
"""

import numpy as np
import sys
import os
from typing import Dict, List, Optional
from dataclasses import dataclass
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Panda3D imports
try:
    from direct.showbase.ShowBase import ShowBase
    from panda3d.core import (
        Vec3, Vec4, Point3, Point2,
        LineSegs, NodePath, GeomNode,
        AmbientLight, DirectionalLight, PointLight, Spotlight,
        TextNode, CardMaker, Fog,
        GeomVertexFormat, GeomVertexData, GeomVertexWriter,
        Geom, GeomTriangles, GeomLines,
        CollisionTraverser, CollisionNode, CollisionSphere,
        CollisionHandlerQueue, BitMask32,
        TransparencyAttrib, ColorBlendAttrib,
        WindowProperties
    )
    from direct.task import Task
    from direct.particles.Particles import Particles
    from direct.particles.ParticleEffect import ParticleEffect
    from direct.gui.OnscreenText import OnscreenText
    from direct.gui.DirectGui import DirectFrame
    from direct.interval.IntervalGlobal import Sequence, Parallel, LerpPosInterval
    PANDA3D_AVAILABLE = True
except ImportError as e:
    PANDA3D_AVAILABLE = False
    print(f"Panda3D not installed: {e}")
    print("Install with: pip install panda3d")


def create_asteroid_geom(radius: float = 1.0, irregularity: float = 0.3) -> GeomNode:
    """
    Create an irregular asteroid geometry (icosphere with noise).
    """
    format = GeomVertexFormat.getV3n3c4()
    vdata = GeomVertexData("asteroid", format, Geom.UHStatic)
    
    vertex = GeomVertexWriter(vdata, "vertex")
    normal = GeomVertexWriter(vdata, "normal")
    color = GeomVertexWriter(vdata, "color")
    
    # Generate icosphere vertices
    phi = (1 + np.sqrt(5)) / 2
    vertices = []
    
    base_verts = [
        (-1, phi, 0), (1, phi, 0), (-1, -phi, 0), (1, -phi, 0),
        (0, -1, phi), (0, 1, phi), (0, -1, -phi), (0, 1, -phi),
        (phi, 0, -1), (phi, 0, 1), (-phi, 0, -1), (-phi, 0, 1)
    ]
    
    for v in base_verts:
        # Normalize and add noise for irregular shape
        n = np.array(v)
        n = n / np.linalg.norm(n)
        
        # Add randomness
        noise = 1.0 + irregularity * (np.random.random() - 0.5) * 2
        v_pos = n * radius * noise
        
        vertex.addData3f(v_pos[0], v_pos[1], v_pos[2])
        normal.addData3f(n[0], n[1], n[2])
        
        # Brownish-gray asteroid color
        r = 0.4 + 0.2 * np.random.random()
        g = 0.35 + 0.15 * np.random.random()
        b = 0.3 + 0.1 * np.random.random()
        color.addData4f(r, g, b, 1.0)
        vertices.append(v_pos)
    
    # Create triangles (icosahedron faces)
    prim = GeomTriangles(Geom.UHStatic)
    faces = [
        (0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
        (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
        (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
        (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)
    ]
    
    for face in faces:
        prim.addVertices(face[0], face[1], face[2])
    
    geom = Geom(vdata)
    geom.addPrimitive(prim)
    
    node = GeomNode("asteroid")
    node.addGeom(geom)
    
    return node


def create_blade_geom(length: float = 5.0, chord: float = 0.5, twist: float = 0.2) -> GeomNode:
    """
    Create a blade-tether geometry with visible flat surface.
    """
    format = GeomVertexFormat.getV3n3c4()
    vdata = GeomVertexData("blade", format, Geom.UHStatic)
    
    vertex = GeomVertexWriter(vdata, "vertex")
    normal = GeomVertexWriter(vdata, "normal")
    color = GeomVertexWriter(vdata, "color")
    
    n_segments = 8
    
    # Generate blade vertices
    vertices = []
    for i in range(n_segments + 1):
        t = i / n_segments
        x = t * length
        
        # Twist angle increases along blade
        twist_angle = t * twist * np.pi
        
        # Leading and trailing edge
        y_lead = chord / 2 * np.cos(twist_angle)
        z_lead = chord / 2 * np.sin(twist_angle)
        y_trail = -chord / 2 * np.cos(twist_angle)
        z_trail = -chord / 2 * np.sin(twist_angle)
        
        # Leading edge vertex
        vertex.addData3f(x, y_lead, z_lead)
        normal.addData3f(0, np.sin(twist_angle), np.cos(twist_angle))
        color.addData4f(0.7, 0.8, 0.9, 0.9)  # Silver-blue
        
        # Trailing edge vertex
        vertex.addData3f(x, y_trail, z_trail)
        normal.addData3f(0, -np.sin(twist_angle), -np.cos(twist_angle))
        color.addData4f(0.6, 0.7, 0.8, 0.9)
    
    # Create triangles
    prim = GeomTriangles(Geom.UHStatic)
    for i in range(n_segments):
        base = i * 2
        # Quad as two triangles
        prim.addVertices(base, base + 1, base + 2)
        prim.addVertices(base + 1, base + 3, base + 2)
    
    geom = Geom(vdata)
    geom.addPrimitive(prim)
    
    node = GeomNode("blade")
    node.addGeom(geom)
    
    return node


def create_explosion_particles() -> List[np.ndarray]:
    """Generate explosion particle positions"""
    n_particles = 50
    particles = []
    for _ in range(n_particles):
        direction = np.random.randn(3)
        direction = direction / np.linalg.norm(direction)
        speed = np.random.uniform(5, 20)
        particles.append(direction * speed)
    return particles


if PANDA3D_AVAILABLE:
    
    class CombatVisualizer(ShowBase):
        """
        Full 3D combat visualization with asteroids and blade-tethers.
        
        Controls:
          WASD - Move camera
          Mouse - Look around
          SPACE - Spawn threat wave
          B - Trigger defensive burst
          R - Reset camera
          1-4 - Camera presets
          ESC - Exit
        """
        
        def __init__(self, arena=None):
            ShowBase.__init__(self)
            
            # Window setup
            props = WindowProperties()
            props.setTitle("KAPS COMBAT ARENA - ASTEROID DEFENSE")
            props.setSize(1920, 1080)
            self.win.requestProperties(props)
            
            self.arena = arena
            
            # Camera setup
            self.disableMouse()
            self.camera_distance = 300
            self.camera_angle = 0
            self.camera_pitch = 30
            self._setup_camera()
            
            # Lighting
            self._setup_lighting()
            
            # Fog for depth
            self._setup_fog()
            
            # Visual node dictionaries
            self.lattice_nodes: Dict[str, NodePath] = {}
            self.blade_nodes: Dict[str, NodePath] = {}
            self.threat_nodes: Dict[str, NodePath] = {}
            self.explosion_nodes: List[NodePath] = []
            
            # HUD
            self.hud_elements = {}
            
            # Create scene
            self._create_skybox()
            self._create_lattice()
            self._create_hud()
            
            # Input handling
            self._setup_input()
            
            # Update task
            self.taskMgr.add(self._update_task, "main_update")
            self.taskMgr.add(self._camera_orbit_task, "camera_orbit")
            
            # Time tracking
            self.last_update_time = time.time()
            self.frame_count = 0
            
            # Camera orbit
            self.orbit_speed = 5  # degrees per second
            self.auto_orbit = True
            
            print("=" * 60)
            print("COMBAT VISUALIZER INITIALIZED")
            print("=" * 60)
            print("Controls:")
            print("  SPACE  - Spawn threat wave")
            print("  B      - Defensive burst")
            print("  R      - Reset camera")
            print("  1-4    - Camera presets")
            print("  O      - Toggle auto-orbit")
            print("  ESC    - Exit")
            print("=" * 60)
        
        def _setup_camera(self):
            """Initialize camera position"""
            rad_angle = np.radians(self.camera_angle)
            rad_pitch = np.radians(self.camera_pitch)
            
            x = self.camera_distance * np.cos(rad_pitch) * np.cos(rad_angle)
            y = self.camera_distance * np.cos(rad_pitch) * np.sin(rad_angle)
            z = self.camera_distance * np.sin(rad_pitch)
            
            self.camera.setPos(x, y, z)
            self.camera.lookAt(0, 0, 0)
        
        def _setup_lighting(self):
            """Create dramatic lighting"""
            # Ambient
            ambient = AmbientLight("ambient")
            ambient.setColor(Vec4(0.15, 0.15, 0.2, 1))
            ambient_np = self.render.attachNewNode(ambient)
            self.render.setLight(ambient_np)
            
            # Key light (sun)
            sun = DirectionalLight("sun")
            sun.setColor(Vec4(1.0, 0.95, 0.9, 1))
            sun_np = self.render.attachNewNode(sun)
            sun_np.setHpr(45, -60, 0)
            self.render.setLight(sun_np)
            
            # Fill light (blue rim)
            fill = DirectionalLight("fill")
            fill.setColor(Vec4(0.3, 0.4, 0.6, 1))
            fill_np = self.render.attachNewNode(fill)
            fill_np.setHpr(-135, -30, 0)
            self.render.setLight(fill_np)
            
            # Red warning light for threats
            self.warning_light = PointLight("warning")
            self.warning_light.setColor(Vec4(0, 0, 0, 1))  # Off by default
            self.warning_light.setAttenuation((1, 0, 0.0001))
            self.warning_light_np = self.render.attachNewNode(self.warning_light)
            self.warning_light_np.setPos(0, 0, 0)
            self.render.setLight(self.warning_light_np)
        
        def _setup_fog(self):
            """Add distance fog"""
            fog = Fog("arena_fog")
            fog.setColor(0.05, 0.05, 0.1)
            fog.setExpDensity(0.001)
            self.render.setFog(fog)
        
        def _create_skybox(self):
            """Create sky background with ocean below"""
            # Sky gradient - light blue horizon
            self.setBackgroundColor(0.4, 0.6, 0.9)
            
            # Create distant stars (visible at altitude)
            lines = LineSegs()
            lines.setThickness(1.0)
            
            for _ in range(200):
                # Random position on upper hemisphere only
                theta = np.random.uniform(0, 2 * np.pi)
                phi = np.random.uniform(0.1, np.pi/2)  # Upper sky only
                r = 2000
                
                x = r * np.cos(phi) * np.cos(theta)
                y = r * np.cos(phi) * np.sin(theta)
                z = r * np.sin(phi)
                
                # Star brightness (faint in daylight)
                brightness = np.random.uniform(0.1, 0.4)
                lines.setColor(brightness, brightness, brightness * 1.1, 1)
                
                lines.moveTo(Point3(x, y, z))
                lines.drawTo(Point3(x + 0.5, y + 0.5, z + 0.5))
            
            stars = self.render.attachNewNode(lines.create())
            
            # Create ocean surface below
            self._create_ocean()
        
        def _create_ocean(self):
            """Create animated ocean surface with waves"""
            ocean_size = 3000  # Large ocean plane
            ocean_depth = -150  # Below the arena
            wave_resolution = 40  # Grid resolution
            
            # Create ocean geometry with wave topology
            format = GeomVertexFormat.getV3n3c4()
            vdata = GeomVertexData("ocean", format, Geom.UHDynamic)
            
            vertex = GeomVertexWriter(vdata, "vertex")
            normal = GeomVertexWriter(vdata, "normal")
            color = GeomVertexWriter(vdata, "color")
            
            # Store vertex data for wave animation
            self.ocean_vertices = []
            
            # Generate grid vertices with wave displacement
            for i in range(wave_resolution + 1):
                for j in range(wave_resolution + 1):
                    # Position on grid
                    x = (i / wave_resolution - 0.5) * ocean_size
                    y = (j / wave_resolution - 0.5) * ocean_size
                    
                    # Wave displacement (static initial, animated in update)
                    wave1 = np.sin(x * 0.02) * 5
                    wave2 = np.sin(y * 0.015) * 4
                    wave3 = np.sin((x + y) * 0.01) * 3
                    z = ocean_depth + wave1 + wave2 + wave3
                    
                    vertex.addData3f(x, y, z)
                    normal.addData3f(0, 0, 1)
                    
                    # Ocean color gradient - deeper blue further from center
                    dist = np.sqrt(x*x + y*y) / ocean_size
                    r = 0.02 + 0.05 * (1 - dist)
                    g = 0.15 + 0.2 * (1 - dist)
                    b = 0.4 + 0.35 * (1 - dist)
                    alpha = 0.85
                    color.addData4f(r, g, b, alpha)
                    
                    self.ocean_vertices.append([x, y, z])
            
            # Create triangles from grid
            prim = GeomTriangles(Geom.UHStatic)
            for i in range(wave_resolution):
                for j in range(wave_resolution):
                    # Indices of quad corners
                    v0 = i * (wave_resolution + 1) + j
                    v1 = v0 + 1
                    v2 = v0 + (wave_resolution + 1)
                    v3 = v2 + 1
                    
                    # Two triangles per quad
                    prim.addVertices(v0, v2, v1)
                    prim.addVertices(v1, v2, v3)
            
            geom = Geom(vdata)
            geom.addPrimitive(prim)
            
            ocean_node = GeomNode("ocean")
            ocean_node.addGeom(geom)
            
            self.ocean = self.render.attachNewNode(ocean_node)
            self.ocean.setTransparency(TransparencyAttrib.MAlpha)
            
            # Add foam/whitecap lines at wave peaks
            self._create_wave_foam()
            
            # Store for animation
            self.ocean_vdata = vdata
            self.wave_time = 0
            self.wave_resolution = wave_resolution
        
        def _create_wave_foam(self):
            """Create foam/whitecap effect on wave peaks"""
            foam_lines = LineSegs()
            foam_lines.setThickness(2.0)
            foam_lines.setColor(0.9, 0.95, 1.0, 0.6)  # White foam
            
            # Random foam patches
            for _ in range(100):
                x = np.random.uniform(-1000, 1000)
                y = np.random.uniform(-1000, 1000)
                z = -145 + np.random.uniform(0, 5)
                length = np.random.uniform(10, 40)
                angle = np.random.uniform(0, 2 * np.pi)
                
                dx = length * np.cos(angle)
                dy = length * np.sin(angle)
                
                foam_lines.moveTo(Point3(x, y, z))
                foam_lines.drawTo(Point3(x + dx, y + dy, z + np.random.uniform(-1, 1)))
            
            self.foam = self.render.attachNewNode(foam_lines.create())
        
        def _create_lattice(self):
            """Create visual representation of the quine lattice"""
            if self.arena is None:
                return
            
            all_nodes = self.arena.lattice.get_all_nodes()
            
            for node in all_nodes:
                # Create node sphere
                sphere = self.loader.loadModel("models/misc/sphere")
                sphere.reparentTo(self.render)
                
                # Size and color based on role
                if node.role.name == 'BUZZARD':
                    sphere.setScale(5)
                    sphere.setColor(0.2, 0.4, 0.9, 1)  # Blue
                elif node.role.name == 'VERTEBRA':
                    sphere.setScale(3)
                    sphere.setColor(0.3, 0.8, 0.3, 1)  # Green
                elif node.role.name == 'GYRO_ARM':
                    sphere.setScale(1.5)
                    sphere.setColor(0.9, 0.7, 0.2, 1)  # Orange
                else:
                    sphere.setScale(1)
                    sphere.setColor(0.6, 0.6, 0.6, 1)  # Gray
                
                pos = node.position
                sphere.setPos(pos[0], pos[1], pos[2])
                
                self.lattice_nodes[node.node_id] = sphere
                
                # Create blade-tether to parent
                if node.parent is not None:
                    blade = self._create_blade_visual(node, node.parent)
                    self.blade_nodes[node.node_id] = blade
        
        def _create_blade_visual(self, child_node, parent_node) -> NodePath:
            """Create a blade tether visual between two nodes"""
            blade_geom = create_blade_geom(length=8.0, chord=0.8, twist=0.3)
            blade = self.render.attachNewNode(blade_geom)
            blade.setTransparency(TransparencyAttrib.MAlpha)
            
            # Position at parent
            pos = parent_node.position
            blade.setPos(pos[0], pos[1], pos[2])
            
            # Orient toward child
            child_pos = child_node.position
            direction = child_pos - pos
            if np.linalg.norm(direction) > 0.01:
                direction = direction / np.linalg.norm(direction)
                # Convert direction to Panda3D rotation
                blade.lookAt(Point3(child_pos[0], child_pos[1], child_pos[2]))
            
            return blade
        
        def _create_threat_visual(self, threat) -> NodePath:
            """Create visual for an incoming threat"""
            # Create asteroid geometry
            asteroid_geom = create_asteroid_geom(
                radius=threat.radius,
                irregularity=0.4
            )
            node = self.render.attachNewNode(asteroid_geom)
            
            # Color based on threat type
            if threat.threat_type.name == 'MISSILE':
                node.setColor(0.9, 0.2, 0.2, 1)  # Red
            elif threat.threat_type.name == 'ENEMY_DRONE':
                node.setColor(0.8, 0.4, 0.9, 1)  # Purple
            else:
                node.setColor(0.6, 0.5, 0.4, 1)  # Brown
            
            # Position
            pos = threat.position
            node.setPos(pos[0], pos[1], pos[2])
            
            return node
        
        def _create_explosion(self, position: np.ndarray, scale: float = 1.0):
            """Create explosion effect at position"""
            # Simple expanding sphere burst
            for _ in range(10):
                lines = LineSegs()
                lines.setThickness(3.0)
                
                # Random outward direction
                direction = np.random.randn(3)
                direction = direction / np.linalg.norm(direction)
                length = scale * np.random.uniform(5, 15)
                
                # Orange-yellow explosion color
                lines.setColor(
                    np.random.uniform(0.8, 1.0),
                    np.random.uniform(0.3, 0.7),
                    np.random.uniform(0.0, 0.2),
                    1.0
                )
                
                start = position
                end = position + direction * length
                
                lines.moveTo(Point3(start[0], start[1], start[2]))
                lines.drawTo(Point3(end[0], end[1], end[2]))
                
                node = self.render.attachNewNode(lines.create())
                self.explosion_nodes.append({
                    'node': node,
                    'birth_time': time.time(),
                    'lifetime': 0.5
                })
        
        def _create_hud(self):
            """Create heads-up display"""
            # Title
            self.hud_elements['title'] = OnscreenText(
                text="KAPS COMBAT ARENA",
                pos=(-1.3, 0.9),
                scale=0.08,
                fg=(0.2, 0.8, 1.0, 1),
                align=TextNode.ALeft,
                font=None
            )
            
            # Wave counter
            self.hud_elements['wave'] = OnscreenText(
                text="WAVE: 0",
                pos=(-1.3, 0.8),
                scale=0.06,
                fg=(1, 1, 1, 1),
                align=TextNode.ALeft
            )
            
            # Score
            self.hud_elements['score'] = OnscreenText(
                text="SCORE: 0",
                pos=(-1.3, 0.7),
                scale=0.06,
                fg=(0.2, 1.0, 0.2, 1),
                align=TextNode.ALeft
            )
            
            # Threat count
            self.hud_elements['threats'] = OnscreenText(
                text="THREATS: 0",
                pos=(1.0, 0.9),
                scale=0.05,
                fg=(1, 0.4, 0.4, 1),
                align=TextNode.ALeft
            )
            
            # Damage meter
            self.hud_elements['damage'] = OnscreenText(
                text="DAMAGE: 0",
                pos=(1.0, 0.8),
                scale=0.05,
                fg=(1, 0.6, 0.2, 1),
                align=TextNode.ALeft
            )
            
            # FPS
            self.hud_elements['fps'] = OnscreenText(
                text="FPS: 60",
                pos=(1.0, -0.9),
                scale=0.04,
                fg=(0.5, 0.5, 0.5, 1),
                align=TextNode.ALeft
            )
            
            # Brain status
            self.hud_elements['brain'] = OnscreenText(
                text="DREAMER: ACTIVE",
                pos=(-1.3, -0.9),
                scale=0.05,
                fg=(0.8, 0.4, 1.0, 1),
                align=TextNode.ALeft
            )
        
        def _update_ocean(self, dt: float):
            """Animate ocean waves"""
            if not hasattr(self, 'ocean_vdata'):
                return
            
            self.wave_time += dt
            
            # Rewrite ocean vertices with time-based wave animation
            vertex = GeomVertexWriter(self.ocean_vdata, "vertex")
            vertex.setRow(0)
            
            res = self.wave_resolution
            ocean_size = 3000
            ocean_depth = -150
            
            for i in range(res + 1):
                for j in range(res + 1):
                    x = (i / res - 0.5) * ocean_size
                    y = (j / res - 0.5) * ocean_size
                    
                    # Animated wave displacement
                    t = self.wave_time
                    wave1 = np.sin(x * 0.02 + t * 0.5) * 6
                    wave2 = np.sin(y * 0.015 - t * 0.3) * 5
                    wave3 = np.sin((x + y) * 0.01 + t * 0.7) * 4
                    wave4 = np.sin(x * 0.008 - y * 0.008 + t * 0.2) * 8  # Large swell
                    z = ocean_depth + wave1 + wave2 + wave3 + wave4
                    
                    vertex.setData3f(x, y, z)
            
            # Animate foam movement
            if hasattr(self, 'foam'):
                self.foam.setH(self.foam.getH() + dt * 2)
                self.foam.setPos(
                    np.sin(self.wave_time * 0.1) * 20,
                    np.cos(self.wave_time * 0.08) * 20,
                    np.sin(self.wave_time * 0.5) * 3
                )
        
        def _update_hud(self):
            """Update HUD with current stats"""
            if self.arena is None:
                return
            
            stats = self.arena.stats
            
            self.hud_elements['wave'].setText(f"WAVE: {stats.wave_survived}")
            self.hud_elements['score'].setText(f"SCORE: {stats.score}")
            self.hud_elements['threats'].setText(f"THREATS: {len(self.arena.spawner.threats)}")
            self.hud_elements['damage'].setText(f"DAMAGE: {stats.damage_taken:.0f}")
            
            # FPS
            current_time = time.time()
            dt = current_time - self.last_update_time
            if dt > 0:
                fps = 1.0 / dt
                self.hud_elements['fps'].setText(f"FPS: {fps:.0f}")
            
            # Brain status flicker
            if self.frame_count % 30 < 15:
                self.hud_elements['brain'].setFg((0.8, 0.4, 1.0, 1.0))
            else:
                self.hud_elements['brain'].setFg((1.0, 0.6, 1.0, 1.0))
        
        def _setup_input(self):
            """Setup keyboard controls"""
            self.accept("escape", self.userExit)
            self.accept("space", self._spawn_wave)
            self.accept("b", self._trigger_burst)
            self.accept("r", self._reset_camera)
            self.accept("o", self._toggle_orbit)
            self.accept("1", lambda: self._camera_preset(1))
            self.accept("2", lambda: self._camera_preset(2))
            self.accept("3", lambda: self._camera_preset(3))
            self.accept("4", lambda: self._camera_preset(4))
        
        def _spawn_wave(self):
            """Manually spawn a threat wave"""
            if self.arena:
                wave = self.arena.spawner.spawn_wave()
                
                # Flash warning light
                self.warning_light.setColor(Vec4(1, 0.2, 0.1, 1))
                self.taskMgr.doMethodLater(
                    0.5, 
                    lambda task: self.warning_light.setColor(Vec4(0, 0, 0, 1)) or Task.done,
                    "warning_flash"
                )
        
        def _trigger_burst(self):
            """Trigger defensive burst"""
            if self.arena:
                # Increase all blade rotations briefly
                for node in self.arena.lattice.get_all_nodes():
                    node.corkscrew_frequency *= 2.0
                
                # Reset after 1 second
                self.taskMgr.doMethodLater(
                    1.0,
                    lambda task: self._reset_burst() or Task.done,
                    "burst_reset"
                )
        
        def _reset_burst(self):
            """Reset burst mode"""
            if self.arena:
                for node in self.arena.lattice.get_all_nodes():
                    node.corkscrew_frequency = max(1.0, node.corkscrew_frequency / 2.0)
        
        def _reset_camera(self):
            """Reset camera to default position"""
            self.camera_distance = 300
            self.camera_angle = 0
            self.camera_pitch = 30
            self._setup_camera()
        
        def _toggle_orbit(self):
            """Toggle auto camera orbit"""
            self.auto_orbit = not self.auto_orbit
            print(f"Auto-orbit: {'ON' if self.auto_orbit else 'OFF'}")
        
        def _camera_preset(self, preset: int):
            """Switch to camera preset"""
            presets = {
                1: (300, 0, 30),     # Front
                2: (150, 90, 10),    # Side close
                3: (500, 45, 60),    # High overview
                4: (100, 180, 0),    # Behind close
            }
            if preset in presets:
                self.camera_distance, self.camera_angle, self.camera_pitch = presets[preset]
                self._setup_camera()
        
        def _camera_orbit_task(self, task):
            """Slowly orbit camera around scene"""
            if self.auto_orbit:
                dt = globalClock.getDt()
                self.camera_angle += self.orbit_speed * dt
                if self.camera_angle > 360:
                    self.camera_angle -= 360
                self._setup_camera()
            return Task.cont
        
        def _update_task(self, task):
            """Main update loop"""
            current_time = time.time()
            dt = current_time - self.last_update_time
            self.last_update_time = current_time
            self.frame_count += 1
            
            if self.arena is None:
                return Task.cont
            
            # Step combat simulation
            obs = self.arena.step()
            
            # Update lattice visuals
            all_nodes = self.arena.lattice.get_all_nodes()
            for node in all_nodes:
                if node.node_id in self.lattice_nodes:
                    visual = self.lattice_nodes[node.node_id]
                    pos = node.position
                    visual.setPos(pos[0], pos[1], pos[2])
                    
                    # Spin gyro arms
                    if node.role.name == 'GYRO_ARM':
                        visual.setH(visual.getH() + node.corkscrew_frequency * 360 * dt)
                
                # Update blade positions
                if node.node_id in self.blade_nodes and node.parent is not None:
                    blade = self.blade_nodes[node.node_id]
                    parent_pos = node.parent.position
                    child_pos = node.position
                    
                    blade.setPos(parent_pos[0], parent_pos[1], parent_pos[2])
                    blade.lookAt(Point3(child_pos[0], child_pos[1], child_pos[2]))
                    
                    # Spin blade around its axis
                    blade.setR(blade.getR() + node.corkscrew_frequency * 360 * dt)
            
            # Update threat visuals
            current_threats = {t.threat_id for t in self.arena.spawner.threats if t.alive}
            
            # Remove dead threats
            for threat_id in list(self.threat_nodes.keys()):
                if threat_id not in current_threats:
                    # Create explosion at threat position
                    node = self.threat_nodes[threat_id]
                    pos = node.getPos()
                    self._create_explosion(np.array([pos.x, pos.y, pos.z]), 2.0)
                    
                    # Remove visual
                    node.removeNode()
                    del self.threat_nodes[threat_id]
            
            # Add new threats
            for threat in self.arena.spawner.threats:
                if threat.alive and threat.threat_id not in self.threat_nodes:
                    self.threat_nodes[threat.threat_id] = self._create_threat_visual(threat)
                elif threat.alive and threat.threat_id in self.threat_nodes:
                    # Update position
                    pos = threat.position
                    self.threat_nodes[threat.threat_id].setPos(pos[0], pos[1], pos[2])
                    
                    # Rotate asteroid
                    node = self.threat_nodes[threat.threat_id]
                    node.setH(node.getH() + 30 * dt)
                    node.setP(node.getP() + 20 * dt)
            
            # Clean up old explosions
            new_explosions = []
            for exp in self.explosion_nodes:
                age = current_time - exp['birth_time']
                if age < exp['lifetime']:
                    # Fade out
                    alpha = 1.0 - age / exp['lifetime']
                    # Scale up
                    scale = 1.0 + age * 3
                    exp['node'].setScale(scale)
                    new_explosions.append(exp)
                else:
                    exp['node'].removeNode()
            self.explosion_nodes = new_explosions
            
            # Animate ocean waves
            self._update_ocean(dt)
            
            # Update HUD
            self._update_hud()
            
            # Warning light intensity based on closest threat
            closest = self.arena.spawner.get_closest_threat(self.arena.lattice.root.position)
            if closest:
                dist = np.linalg.norm(closest.position - self.arena.lattice.root.position)
                intensity = max(0, 1.0 - dist / 200)
                self.warning_light.setColor(Vec4(intensity, intensity * 0.1, 0, 1))
            
            return Task.cont


def run_combat_visualization(champion_path: str = None):
    """Launch the combat visualizer with DreamerV3 brain and cascade collective!
    
    Args:
        champion_path: Path to champion capsule .py file. If None, searches default locations.
    """
    if not PANDA3D_AVAILABLE:
        print("Panda3D not available!")
        print("Install with: pip install panda3d")
        return
    
    import sys
    import importlib.util
    from pathlib import Path
    
    # Add project root
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))
    
    # Search paths for champion capsules (priority order)
    search_paths = [
        # Explicit path
        champion_path,
        # HuggingFace key-data repo (your evolved champions)
        "F:/End-Game/glassboxgames/children/key-data-repo/models/champion_gen52.py",
        "F:/End-Game/glassboxgames/children/key-data-repo/models/champion_gen42.py",
        # Project root fallback
        str(project_root / "champion_gen42.py"),
    ]
    
    # Load brain from first valid path
    brain = None
    for path in search_paths:
        if path is None:
            continue
        capsule_path = Path(path)
        if capsule_path.exists():
            try:
                print(f"[BRAIN] Loading champion from: {capsule_path}")
                
                # Dynamic import from file path
                spec = importlib.util.spec_from_file_location("champion", str(capsule_path))
                champion = importlib.util.module_from_spec(spec)
                sys.modules["champion"] = champion
                spec.loader.exec_module(champion)
                
                # Try to get QuineBrain (standalone numpy inference)
                if hasattr(champion, 'QuineBrain'):
                    brain = champion.QuineBrain()
                    print(f"[BRAIN] QuineBrain loaded! action_dim={brain.action_dim}, latent_dim={brain.latent_dim}")
                    if hasattr(brain, 'get_merkle_hash'):
                        print(f"[BRAIN] Merkle hash: {brain.get_merkle_hash()[:32]}...")
                    break
                    
                # Fallback: CapsuleAgent (full featured with CASCADE)
                elif hasattr(champion, 'CapsuleAgent'):
                    agent = champion.CapsuleAgent(observe=False, observe_visual=False)
                    brain = agent.brain
                    print(f"[BRAIN] CapsuleAgent loaded! Generation: {getattr(champion, '_GENERATION', '?')}")
                    print(f"[BRAIN] Quine hash: {getattr(champion, '_QUINE_HASH', 'unknown')[:32]}...")
                    break
                    
            except Exception as e:
                print(f"[BRAIN] Failed to load {capsule_path.name}: {e}")
                continue
    
    if brain is None:
        print("[BRAIN] No valid champion found - using fallback reactive controller")
    
    # Create the combat arena with collective intelligence
    from src.ai.collective_intelligence import create_collective_arena
    arena, collective = create_collective_arena()
    
    # Attach the real brain to the collective
    if brain:
        for qid, cascade_brain in collective.quines.items():
            cascade_brain.champion = brain
            cascade_brain.brain_hash = brain.get_merkle_hash()
        print(f"[COLLECTIVE] Attached QuineBrain to {len(collective.quines)} quines")
    
    # Create visualizer
    viz = CombatVisualizer(arena)
    viz.collective = collective
    viz.brain = brain
    
    print("\n" + "=" * 60)
    print("🚀 KAPS COMBAT ARENA - DREAMERV3 POWERED")
    print("=" * 60)
    print(f"  Quines: {len(collective.quines)}")
    print(f"  Genesis Root: {collective.quines[list(collective.quines.keys())[0]].last_cid or 'pending'}")
    print(f"  Arena Radius: {arena.arena_radius}m")
    print("=" * 60 + "\n")
    
    viz.run()


if __name__ == "__main__":
    run_combat_visualization()
