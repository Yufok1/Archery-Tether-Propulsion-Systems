"""
Modern OpenGL Renderer for KAPS
================================

Clean, professional visualization using ModernGL.
NOT cartoony - proper shading, depth, and lighting.

This plugs directly into the existing physics and AI systems.
"""

import numpy as np
import moderngl
import moderngl_window as mglw
from moderngl_window import geometry
from pyrr import Matrix44, Vector3, matrix44
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

# Vertex shader with proper lighting
VERTEX_SHADER = """
#version 330

in vec3 in_position;
in vec3 in_normal;
in vec3 in_color;

out vec3 v_position;
out vec3 v_normal;
out vec3 v_color;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main() {
    v_position = vec3(model * vec4(in_position, 1.0));
    v_normal = mat3(transpose(inverse(model))) * in_normal;
    v_color = in_color;
    gl_Position = projection * view * model * vec4(in_position, 1.0);
}
"""

# Fragment shader with Blinn-Phong lighting
FRAGMENT_SHADER = """
#version 330

in vec3 v_position;
in vec3 v_normal;
in vec3 v_color;

out vec4 fragColor;

uniform vec3 light_pos;
uniform vec3 view_pos;
uniform vec3 ambient_color;
uniform float ambient_strength;
uniform float specular_strength;
uniform float shininess;

void main() {
    // Ambient
    vec3 ambient = ambient_strength * ambient_color;
    
    // Diffuse
    vec3 norm = normalize(v_normal);
    vec3 light_dir = normalize(light_pos - v_position);
    float diff = max(dot(norm, light_dir), 0.0);
    vec3 diffuse = diff * vec3(1.0, 0.98, 0.95);
    
    // Specular (Blinn-Phong)
    vec3 view_dir = normalize(view_pos - v_position);
    vec3 halfway_dir = normalize(light_dir + view_dir);
    float spec = pow(max(dot(norm, halfway_dir), 0.0), shininess);
    vec3 specular = specular_strength * spec * vec3(1.0, 1.0, 1.0);
    
    // Combine
    vec3 result = (ambient + diffuse + specular) * v_color;
    
    // Gamma correction
    result = pow(result, vec3(1.0/2.2));
    
    fragColor = vec4(result, 1.0);
}
"""

# Line shader for cables
LINE_VERTEX_SHADER = """
#version 330
in vec3 in_position;
in vec3 in_color;
out vec3 v_color;
uniform mat4 mvp;
void main() {
    v_color = in_color;
    gl_Position = mvp * vec4(in_position, 1.0);
}
"""

LINE_FRAGMENT_SHADER = """
#version 330
in vec3 v_color;
out vec4 fragColor;
void main() {
    fragColor = vec4(v_color, 1.0);
}
"""


def create_cylinder_mesh(radius: float = 1.0, height: float = 2.0, segments: int = 16):
    """Create cylinder vertices for fuselage."""
    vertices = []
    normals = []
    
    # Side faces
    for i in range(segments):
        angle1 = 2 * np.pi * i / segments
        angle2 = 2 * np.pi * (i + 1) / segments
        
        x1, z1 = np.cos(angle1) * radius, np.sin(angle1) * radius
        x2, z2 = np.cos(angle2) * radius, np.sin(angle2) * radius
        
        # Two triangles per segment
        # Bottom triangle
        vertices.extend([x1, -height/2, z1, x2, -height/2, z2, x1, height/2, z1])
        n1 = [np.cos(angle1), 0, np.sin(angle1)]
        n2 = [np.cos(angle2), 0, np.sin(angle2)]
        normals.extend(n1 + n2 + n1)
        
        # Top triangle
        vertices.extend([x1, height/2, z1, x2, -height/2, z2, x2, height/2, z2])
        normals.extend(n1 + n2 + n2)
    
    return np.array(vertices, dtype='f4'), np.array(normals, dtype='f4')


def create_cone_mesh(radius: float = 1.0, height: float = 2.0, segments: int = 16):
    """Create cone vertices for nose/tail."""
    vertices = []
    normals = []
    
    for i in range(segments):
        angle1 = 2 * np.pi * i / segments
        angle2 = 2 * np.pi * (i + 1) / segments
        
        x1, z1 = np.cos(angle1) * radius, np.sin(angle1) * radius
        x2, z2 = np.cos(angle2) * radius, np.sin(angle2) * radius
        
        # Triangle from base to tip
        vertices.extend([x1, 0, z1, x2, 0, z2, 0, height, 0])
        
        # Approximate normals
        n1 = np.array([np.cos(angle1), 0.5, np.sin(angle1)])
        n1 = n1 / np.linalg.norm(n1)
        n2 = np.array([np.cos(angle2), 0.5, np.sin(angle2)])
        n2 = n2 / np.linalg.norm(n2)
        nt = np.array([0, 1, 0])
        normals.extend(list(n1) + list(n2) + list(nt))
    
    return np.array(vertices, dtype='f4'), np.array(normals, dtype='f4')


def create_wing_mesh(span: float = 6.0, chord: float = 2.0, thickness: float = 0.3):
    """Create delta wing mesh."""
    vertices = []
    normals = []
    
    # Simple delta wing - nose, left tip, right tip, trailing edges
    # Top surface
    verts_top = [
        [chord, 0, thickness/2],           # Nose
        [-chord*0.3, span/2, thickness/4], # Right tip
        [-chord*0.3, -span/2, thickness/4], # Left tip
        [-chord*0.5, 0, thickness/4],      # Tail center
    ]
    
    # Top triangles
    tris_top = [
        (0, 1, 3),  # Nose to right to tail
        (0, 3, 2),  # Nose to tail to left
    ]
    
    for tri in tris_top:
        for idx in tri:
            vertices.extend(verts_top[idx])
            normals.extend([0, 0, 1])
    
    # Bottom surface
    verts_bot = [
        [chord, 0, -thickness/4],
        [-chord*0.3, span/2, -thickness/4],
        [-chord*0.3, -span/2, -thickness/4],
        [-chord*0.5, 0, -thickness/4],
    ]
    
    tris_bot = [
        (0, 3, 1),  # Reversed winding
        (0, 2, 3),
    ]
    
    for tri in tris_bot:
        for idx in tri:
            vertices.extend(verts_bot[idx])
            normals.extend([0, 0, -1])
    
    # Leading edges
    vertices.extend(verts_top[0] + verts_bot[0] + verts_top[1])
    normals.extend([1, 0, 0] * 3)
    vertices.extend(verts_bot[0] + verts_bot[1] + verts_top[1])
    normals.extend([1, 0, 0] * 3)
    
    vertices.extend(verts_top[0] + verts_top[2] + verts_bot[0])
    normals.extend([1, 0, 0] * 3)
    vertices.extend(verts_bot[0] + verts_top[2] + verts_bot[2])
    normals.extend([1, 0, 0] * 3)
    
    return np.array(vertices, dtype='f4'), np.array(normals, dtype='f4')


@dataclass
class EntityState:
    """State of a renderable entity."""
    position: np.ndarray
    velocity: np.ndarray
    rotation: np.ndarray = None
    color: Tuple[float, float, float] = (0.5, 0.5, 0.5)
    scale: float = 1.0


class KAPSModernRenderer(mglw.WindowConfig):
    """
    Modern OpenGL renderer for KAPS visualization.
    
    Clean, professional look with proper lighting and shaders.
    """
    
    gl_version = (3, 3)
    title = "KAPS - Kinetic Active Protection System"
    window_size = (1280, 720)
    aspect_ratio = 16 / 9
    resizable = True
    samples = 4  # Anti-aliasing
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Enable depth testing
        self.ctx.enable(moderngl.DEPTH_TEST)
        
        # Shaders
        self.prog = self.ctx.program(
            vertex_shader=VERTEX_SHADER,
            fragment_shader=FRAGMENT_SHADER,
        )
        
        self.line_prog = self.ctx.program(
            vertex_shader=LINE_VERTEX_SHADER,
            fragment_shader=LINE_FRAGMENT_SHADER,
        )
        
        # Create meshes
        self._create_meshes()
        
        # Camera - orbiting for proper 3D view
        self.camera_distance = 200.0
        self.camera_height = 80.0
        self.camera_angle = 0.0  # Will animate
        self.camera_pos = np.array([150.0, -150.0, 100.0])
        self.camera_target = np.array([0.0, 0.0, 50.0])
        self.camera_up = np.array([0.0, 0.0, 1.0])
        
        # Create ground grid
        self._create_ground_grid()
        
        # Entity states - TABs spread in 3D space around Buzzard
        self.buzzard_state = EntityState(
            position=np.array([0.0, 0.0, 50.0]),
            velocity=np.array([0.0, 30.0, 0.0]),
            color=(0.2, 0.35, 0.8)
        )
        
        # TABs at diagonal positions for proper 3D spread
        cable_len = 40.0
        self.tab_states: Dict[str, EntityState] = {
            "UP": EntityState(np.array([25, 25, 75]), np.zeros(3), color=(0.2, 0.8, 0.2)),
            "DOWN": EntityState(np.array([-25, -25, 25]), np.zeros(3), color=(0.8, 0.2, 0.2)),
            "LEFT": EntityState(np.array([-30, 20, 50]), np.zeros(3), color=(0.8, 0.8, 0.2)),
            "RIGHT": EntityState(np.array([30, -20, 50]), np.zeros(3), color=(0.8, 0.2, 0.8)),
        }
        
        self.tabs_attached = {"UP": True, "DOWN": True, "LEFT": True, "RIGHT": True}
        
        # Simulation interface
        self.sim = None
        self.env = None
        
        # Time tracking
        self.time = 0
        
    def _create_meshes(self):
        """Create GPU meshes."""
        # Buzzard fuselage (cylinder + cones) - BIGGER
        cyl_v, cyl_n = create_cylinder_mesh(radius=6.0, height=30.0, segments=16)
        nose_v, nose_n = create_cone_mesh(radius=6.0, height=12.0, segments=16)
        tail_v, tail_n = create_cone_mesh(radius=4.0, height=8.0, segments=16)
        
        # Offset nose and tail
        nose_v = nose_v.reshape(-1, 3)
        nose_v[:, 1] += 15  # Move to front
        nose_v = nose_v.flatten()
        
        tail_v = tail_v.reshape(-1, 3)
        tail_v[:, 1] = -tail_v[:, 1] - 15  # Flip and move to back
        tail_v = tail_v.flatten()
        
        # Combine fuselage
        fuse_v = np.concatenate([cyl_v, nose_v, tail_v])
        fuse_n = np.concatenate([cyl_n, nose_n, tail_n])
        fuse_c = np.tile([0.2, 0.35, 0.8], len(fuse_v) // 3).astype('f4')
        
        fuse_data = np.zeros(len(fuse_v) // 3, dtype=[
            ('in_position', 'f4', 3),
            ('in_normal', 'f4', 3),
            ('in_color', 'f4', 3),
        ])
        fuse_data['in_position'] = fuse_v.reshape(-1, 3)
        fuse_data['in_normal'] = fuse_n.reshape(-1, 3)
        fuse_data['in_color'] = fuse_c.reshape(-1, 3)
        
        self.buzzard_vbo = self.ctx.buffer(fuse_data.tobytes())
        self.buzzard_vao = self.ctx.vertex_array(
            self.prog,
            [(self.buzzard_vbo, '3f 3f 3f', 'in_position', 'in_normal', 'in_color')]
        )
        self.buzzard_vertex_count = len(fuse_v) // 3
        
        # TAB wings
        wing_v, wing_n = create_wing_mesh(span=20.0, chord=10.0, thickness=2.0)
        wing_c = np.tile([0.8, 0.8, 0.8], len(wing_v) // 3).astype('f4')  # Brighter
        
        wing_data = np.zeros(len(wing_v) // 3, dtype=[
            ('in_position', 'f4', 3),
            ('in_normal', 'f4', 3),
            ('in_color', 'f4', 3),
        ])
        wing_data['in_position'] = wing_v.reshape(-1, 3)
        wing_data['in_normal'] = wing_n.reshape(-1, 3)
        wing_data['in_color'] = wing_c.reshape(-1, 3)
        
        self.wing_vbo = self.ctx.buffer(wing_data.tobytes())
        self.wing_vao = self.ctx.vertex_array(
            self.prog,
            [(self.wing_vbo, '3f 3f 3f', 'in_position', 'in_normal', 'in_color')]
        )
        self.wing_vertex_count = len(wing_v) // 3
        
        # Cable line buffer (dynamic)
        self.cable_vbo = self.ctx.buffer(reserve=1024 * 6 * 4)  # Reserve space
    
    def _create_ground_grid(self):
        """Create a ground reference grid for depth perception."""
        lines = []
        colors = []
        grid_size = 500
        spacing = 50
        
        for i in range(-grid_size, grid_size + 1, spacing):
            # X lines
            lines.extend([i, -grid_size, 0, i, grid_size, 0])
            colors.extend([0.3, 0.4, 0.3] * 2)
            # Y lines
            lines.extend([-grid_size, i, 0, grid_size, i, 0])
            colors.extend([0.3, 0.4, 0.3] * 2)
        
        grid_data = np.zeros(len(lines) // 3, dtype=[
            ('in_position', 'f4', 3),
            ('in_color', 'f4', 3),
        ])
        grid_data['in_position'] = np.array(lines, dtype='f4').reshape(-1, 3)
        grid_data['in_color'] = np.array(colors, dtype='f4').reshape(-1, 3)
        
        self.grid_vbo = self.ctx.buffer(grid_data.tobytes())
        self.grid_vao = self.ctx.vertex_array(
            self.line_prog,
            [(self.grid_vbo, '3f 3f', 'in_position', 'in_color')]
        )
        self.grid_line_count = len(lines) // 3
        
    def set_simulation(self, sim, env=None):
        """Connect to KAPS simulation."""
        self.sim = sim
        self.env = env
        
    def update_from_sim(self):
        """Pull state from simulation."""
        if self.sim is None:
            return
            
        # Buzzard
        self.buzzard_state.position = self.sim.mother_drone.position.copy()
        self.buzzard_state.velocity = self.sim.mother_drone.velocity.copy()
        
        # TABs
        for tab_id in ["UP", "DOWN", "LEFT", "RIGHT"]:
            tab = self.sim.tab_array.tabs.get(tab_id)
            if tab:
                self.tab_states[tab_id].position = tab.position.copy()
                self.tab_states[tab_id].velocity = tab.velocity.copy()
                self.tabs_attached[tab_id] = tab.is_attached
        
        # Camera follows buzzard
        self.camera_target = self.buzzard_state.position.copy()
        
        # Chase camera position
        vel = self.buzzard_state.velocity
        speed = np.linalg.norm(vel)
        if speed > 1:
            behind = -vel / speed * 80
        else:
            behind = np.array([0, -80, 0])
        self.camera_pos = self.camera_target + behind + np.array([0, 0, 40])
    
    def on_render(self, time: float, frame_time: float):
        """Render frame."""
        self.time = time
        
        # Update from simulation if connected
        self.update_from_sim()
        
        # Orbit camera around scene for demo (when no sim)
        if self.sim is None:
            self.camera_angle = time * 0.3  # Slow orbit
            self.camera_pos = np.array([
                np.cos(self.camera_angle) * self.camera_distance,
                np.sin(self.camera_angle) * self.camera_distance,
                self.camera_height
            ])
            self.camera_target = np.array([0.0, 0.0, 50.0])
            
            # Animate TABs orbiting around Buzzard
            orbit_radius = 60.0
            for i, tab_id in enumerate(["UP", "DOWN", "LEFT", "RIGHT"]):
                angle = time * 0.5 + i * np.pi / 2  # Phase offset
                z_offset = 20 * np.sin(time * 0.7 + i)  # Bobbing
                self.tab_states[tab_id].position = self.buzzard_state.position + np.array([
                    np.cos(angle) * orbit_radius,
                    np.sin(angle) * orbit_radius,
                    z_offset
                ])
        
        # Clear
        self.ctx.clear(0.1, 0.15, 0.2)  # Dark blue-grey sky
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.enable(moderngl.CULL_FACE)
        
        # Matrices
        proj = Matrix44.perspective_projection(60.0, self.aspect_ratio, 0.1, 2000.0)
        view = Matrix44.look_at(
            tuple(self.camera_pos),
            tuple(self.camera_target),
            tuple(self.camera_up),
        )
        
        # Draw ground grid first
        mvp = proj * view
        self.line_prog['mvp'].write(mvp.astype('f4').tobytes())
        self.grid_vao.render(moderngl.LINES)
        
        # Lighting
        self.prog['light_pos'].value = tuple(self.camera_pos + np.array([50, 50, 100]))
        self.prog['view_pos'].value = tuple(self.camera_pos)
        self.prog['ambient_color'].value = (0.6, 0.7, 0.9)
        self.prog['ambient_strength'].value = 0.3
        self.prog['specular_strength'].value = 0.5
        self.prog['shininess'].value = 32.0
        self.prog['view'].write(view.astype('f4').tobytes())
        self.prog['projection'].write(proj.astype('f4').tobytes())
        
        # Render Buzzard
        model = Matrix44.from_translation(self.buzzard_state.position)
        # Orient along velocity
        vel = self.buzzard_state.velocity
        if np.linalg.norm(vel) > 1:
            yaw = np.arctan2(vel[0], vel[1])
            model = model * Matrix44.from_z_rotation(yaw)
        self.prog['model'].write(model.astype('f4').tobytes())
        self.buzzard_vao.render(moderngl.TRIANGLES)
        
        # Render TABs - orient to face direction of orbit
        for i, (tab_id, state) in enumerate(self.tab_states.items()):
            if not self.tabs_attached.get(tab_id, False):
                continue
                
            model = Matrix44.from_translation(state.position)
            
            # Calculate direction from Buzzard to TAB for orientation
            to_buzzard = self.buzzard_state.position - state.position
            dist = np.linalg.norm(to_buzzard)
            
            if dist > 1:
                # Orient so leading edge faces tangent to orbit (perpendicular to cable)
                # First get the cable direction
                cable_dir = to_buzzard / dist
                
                # Tangent is perpendicular to cable in XY plane
                tangent = np.array([-cable_dir[1], cable_dir[0], 0])
                tangent = tangent / (np.linalg.norm(tangent) + 0.001)
                
                # Yaw to face tangent direction
                yaw = np.arctan2(tangent[0], tangent[1])
                
                # Bank angle - tilt into the turn
                bank = np.pi / 6  # 30 degree bank
                
                model = model * Matrix44.from_z_rotation(yaw) * Matrix44.from_y_rotation(bank)
            
            self.prog['model'].write(model.astype('f4').tobytes())
            self.wing_vao.render(moderngl.TRIANGLES)
        
        # Render cables
        self._render_cables(proj, view)
    
    def _update_wing_color(self, color: Tuple[float, float, float]):
        """Update wing VBO with new color."""
        # For efficiency, we'd want per-instance coloring, but for now:
        pass  # Using uniform color from original mesh
    
    def _render_cables(self, proj, view):
        """Render cable lines with catenary sag."""
        lines = []
        colors = []
        
        segments = 12  # Points per cable for curve
        
        for tab_id, state in self.tab_states.items():
            if not self.tabs_attached.get(tab_id, False):
                continue
            
            # Cable from buzzard to TAB
            start = self.buzzard_state.position
            end = state.position
            
            # Tension based on distance
            dist = np.linalg.norm(end - start)
            tension = min(1.0, dist / 80.0)
            sag = (1.0 - tension) * 15.0 + 5.0  # More sag when slack
            
            # Color: green when slack, red when taut
            color = (0.3 + tension * 0.6, 0.7 * (1 - tension), 0.2)
            
            # Generate curved cable points (catenary approximation)
            for i in range(segments):
                t1 = i / segments
                t2 = (i + 1) / segments
                
                # Lerp positions
                p1 = start * (1 - t1) + end * t1
                p2 = start * (1 - t2) + end * t2
                
                # Add sag (parabolic, max at middle)
                sag1 = sag * 4 * t1 * (1 - t1)
                sag2 = sag * 4 * t2 * (1 - t2)
                p1[2] -= sag1
                p2[2] -= sag2
                
                lines.extend(list(p1) + list(p2))
                colors.extend(list(color) * 2)
        
        if not lines:
            return
        
        # Create line VAO
        line_data = np.array(lines, dtype='f4')
        color_data = np.array(colors, dtype='f4')
        
        combined = np.zeros(len(line_data) // 3, dtype=[
            ('in_position', 'f4', 3),
            ('in_color', 'f4', 3),
        ])
        combined['in_position'] = line_data.reshape(-1, 3)
        combined['in_color'] = color_data.reshape(-1, 3)
        
        vbo = self.ctx.buffer(combined.tobytes())
        vao = self.ctx.vertex_array(
            self.line_prog,
            [(vbo, '3f 3f', 'in_position', 'in_color')]
        )
        
        mvp = proj * view
        self.line_prog['mvp'].write(mvp.astype('f4').tobytes())
        
        self.ctx.line_width = 3.0  # Note: May not work on all drivers
        vao.render(moderngl.LINES)
        vbo.release()
        vao.release()


def run_standalone():
    """Run renderer standalone for testing."""
    mglw.run_window_config(KAPSModernRenderer)


if __name__ == "__main__":
    run_standalone()
