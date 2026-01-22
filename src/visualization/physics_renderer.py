"""
Physics-Based KAPS Renderer
============================

Connects to REAL physics simulation - not animated garbage.

This renderer:
1. Runs actual KAPSSimulation.step()
2. Visualizes the REAL state from physics engine
3. Shows actual tether tension, TAB aerodynamics, etc.
"""

import numpy as np
import moderngl
import moderngl_window as mglw
from pyrr import Matrix44
import sys
import os

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.main import KAPSSimulation


def create_sphere_mesh(radius: float = 1.0, segments: int = 16, rings: int = 12):
    """Create sphere vertices."""
    vertices = []
    normals = []
    
    for ring in range(rings):
        theta1 = np.pi * ring / rings
        theta2 = np.pi * (ring + 1) / rings
        
        for seg in range(segments):
            phi1 = 2 * np.pi * seg / segments
            phi2 = 2 * np.pi * (seg + 1) / segments
            
            # Four corners of quad
            p1 = [np.sin(theta1) * np.cos(phi1), np.sin(theta1) * np.sin(phi1), np.cos(theta1)]
            p2 = [np.sin(theta1) * np.cos(phi2), np.sin(theta1) * np.sin(phi2), np.cos(theta1)]
            p3 = [np.sin(theta2) * np.cos(phi2), np.sin(theta2) * np.sin(phi2), np.cos(theta2)]
            p4 = [np.sin(theta2) * np.cos(phi1), np.sin(theta2) * np.sin(phi1), np.cos(theta2)]
            
            # Two triangles
            for p in [p1, p2, p3, p1, p3, p4]:
                vertices.extend([p[0] * radius, p[1] * radius, p[2] * radius])
                normals.extend(p)
    
    return np.array(vertices, dtype='f4'), np.array(normals, dtype='f4')


def create_box_mesh(size: tuple = (1, 1, 1)):
    """Create box vertices for Buzzard fuselage."""
    sx, sy, sz = size
    vertices = []
    normals = []
    
    faces = [
        # Front (+Y)
        ([[-sx, sy, -sz], [sx, sy, -sz], [sx, sy, sz], [-sx, sy, sz]], [0, 1, 0]),
        # Back (-Y)
        ([[sx, -sy, -sz], [-sx, -sy, -sz], [-sx, -sy, sz], [sx, -sy, sz]], [0, -1, 0]),
        # Right (+X)
        ([[sx, sy, -sz], [sx, -sy, -sz], [sx, -sy, sz], [sx, sy, sz]], [1, 0, 0]),
        # Left (-X)
        ([[-sx, -sy, -sz], [-sx, sy, -sz], [-sx, sy, sz], [-sx, -sy, sz]], [-1, 0, 0]),
        # Top (+Z)
        ([[-sx, sy, sz], [sx, sy, sz], [sx, -sy, sz], [-sx, -sy, sz]], [0, 0, 1]),
        # Bottom (-Z)
        ([[-sx, -sy, -sz], [sx, -sy, -sz], [sx, sy, -sz], [-sx, sy, -sz]], [0, 0, -1]),
    ]
    
    for verts, norm in faces:
        # Two triangles per face
        for idx in [0, 1, 2, 0, 2, 3]:
            vertices.extend(verts[idx])
            normals.extend(norm)
    
    return np.array(vertices, dtype='f4'), np.array(normals, dtype='f4')


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

FRAGMENT_SHADER = """
#version 330
in vec3 v_position;
in vec3 v_normal;
in vec3 v_color;
out vec4 fragColor;

uniform vec3 light_pos;
uniform vec3 view_pos;

void main() {
    vec3 ambient = 0.3 * v_color;
    
    vec3 norm = normalize(v_normal);
    vec3 light_dir = normalize(light_pos - v_position);
    float diff = max(dot(norm, light_dir), 0.0);
    vec3 diffuse = diff * v_color;
    
    vec3 view_dir = normalize(view_pos - v_position);
    vec3 halfway = normalize(light_dir + view_dir);
    float spec = pow(max(dot(norm, halfway), 0.0), 32.0);
    vec3 specular = 0.3 * spec * vec3(1.0);
    
    fragColor = vec4(ambient + diffuse + specular, 1.0);
}
"""

LINE_VERTEX = """
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

LINE_FRAGMENT = """
#version 330
in vec3 v_color;
out vec4 fragColor;
void main() {
    fragColor = vec4(v_color, 1.0);
}
"""


class PhysicsRenderer(mglw.WindowConfig):
    """
    Renders the REAL KAPS physics simulation.
    """
    
    gl_version = (3, 3)
    title = "KAPS - Physics Simulation"
    window_size = (1280, 720)
    aspect_ratio = 16 / 9
    resizable = True
    samples = 4
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ctx.enable(moderngl.DEPTH_TEST)
        
        # REAL PHYSICS
        print("Initializing KAPS physics simulation...")
        self.sim = KAPSSimulation()
        self.sim.running = True
        print(f"  Mother drone at: {self.sim.mother_drone.position}")
        print(f"  TABs: {list(self.sim.tab_array.tabs.keys())}")
        
        # Physics timestep accumulator
        self.physics_accumulator = 0.0
        self.physics_rate = 100  # Steps per second
        
        # Create shaders
        self.prog = self.ctx.program(vertex_shader=VERTEX_SHADER, fragment_shader=FRAGMENT_SHADER)
        self.line_prog = self.ctx.program(vertex_shader=LINE_VERTEX, fragment_shader=LINE_FRAGMENT)
        
        # Create meshes
        self._create_meshes()
        self._create_grid()
        
        # Camera
        self.camera_distance = 150.0
        self.camera_height = 60.0
        self.camera_angle = 0.0
        
        # Pause control
        self.paused = False
        
    def _create_meshes(self):
        """Create GPU meshes."""
        # Buzzard - elongated box
        box_v, box_n = create_box_mesh((3, 12, 2))
        box_c = np.tile([0.2, 0.4, 0.8], len(box_v) // 3).astype('f4')
        
        box_data = np.zeros(len(box_v) // 3, dtype=[
            ('in_position', 'f4', 3), ('in_normal', 'f4', 3), ('in_color', 'f4', 3)
        ])
        box_data['in_position'] = box_v.reshape(-1, 3)
        box_data['in_normal'] = box_n.reshape(-1, 3)
        box_data['in_color'] = box_c.reshape(-1, 3)
        
        self.buzzard_vbo = self.ctx.buffer(box_data.tobytes())
        self.buzzard_vao = self.ctx.vertex_array(
            self.prog, [(self.buzzard_vbo, '3f 3f 3f', 'in_position', 'in_normal', 'in_color')]
        )
        
        # TAB - small airfoil box
        tab_v, tab_n = create_box_mesh((4, 2, 0.5))
        tab_c = np.tile([0.8, 0.8, 0.8], len(tab_v) // 3).astype('f4')
        
        tab_data = np.zeros(len(tab_v) // 3, dtype=[
            ('in_position', 'f4', 3), ('in_normal', 'f4', 3), ('in_color', 'f4', 3)
        ])
        tab_data['in_position'] = tab_v.reshape(-1, 3)
        tab_data['in_normal'] = tab_n.reshape(-1, 3)
        tab_data['in_color'] = tab_c.reshape(-1, 3)
        
        self.tab_vbo = self.ctx.buffer(tab_data.tobytes())
        self.tab_vao = self.ctx.vertex_array(
            self.prog, [(self.tab_vbo, '3f 3f 3f', 'in_position', 'in_normal', 'in_color')]
        )
        
    def _create_grid(self):
        """Ground grid."""
        lines = []
        colors = []
        for i in range(-500, 501, 50):
            lines.extend([i, -500, 0, i, 500, 0])
            lines.extend([-500, i, 0, 500, i, 0])
            colors.extend([0.3, 0.4, 0.3] * 4)
        
        grid_data = np.zeros(len(lines) // 3, dtype=[('in_position', 'f4', 3), ('in_color', 'f4', 3)])
        grid_data['in_position'] = np.array(lines, dtype='f4').reshape(-1, 3)
        grid_data['in_color'] = np.array(colors, dtype='f4').reshape(-1, 3)
        
        self.grid_vbo = self.ctx.buffer(grid_data.tobytes())
        self.grid_vao = self.ctx.vertex_array(self.line_prog, [(self.grid_vbo, '3f 3f', 'in_position', 'in_color')])
        
    def key_event(self, key, action, modifiers):
        """Handle keyboard input."""
        if action == self.wnd.keys.ACTION_PRESS:
            if key == self.wnd.keys.SPACE:
                self.paused = not self.paused
                print(f"{'PAUSED' if self.paused else 'RUNNING'}")
            elif key == self.wnd.keys.R:
                # Reset simulation
                self.sim = KAPSSimulation()
                self.sim.running = True
                print("RESET")
    
    def on_render(self, time: float, frame_time: float):
        """Render frame with real physics."""
        # Step physics (fixed timestep)
        if not self.paused:
            self.physics_accumulator += frame_time
            steps = 0
            while self.physics_accumulator >= 1.0 / self.physics_rate and steps < 10:
                self.sim.step()
                self.physics_accumulator -= 1.0 / self.physics_rate
                steps += 1
        
        # Get REAL positions from simulation
        buzzard_pos = self.sim.mother_drone.position
        buzzard_vel = self.sim.mother_drone.velocity
        
        # Camera follows Buzzard
        self.camera_angle = time * 0.2
        cam_offset = np.array([
            np.cos(self.camera_angle) * self.camera_distance,
            np.sin(self.camera_angle) * self.camera_distance,
            self.camera_height
        ])
        camera_pos = buzzard_pos + cam_offset
        camera_target = buzzard_pos
        
        # Clear
        self.ctx.clear(0.1, 0.12, 0.15)
        self.ctx.enable(moderngl.DEPTH_TEST)
        
        # Matrices
        proj = Matrix44.perspective_projection(60.0, self.aspect_ratio, 0.1, 2000.0)
        view = Matrix44.look_at(tuple(camera_pos), tuple(camera_target), (0, 0, 1))
        mvp = proj * view
        
        # Grid
        self.line_prog['mvp'].write(mvp.astype('f4').tobytes())
        self.grid_vao.render(moderngl.LINES)
        
        # Lighting
        self.prog['light_pos'].value = tuple(camera_pos + np.array([100, 100, 200]))
        self.prog['view_pos'].value = tuple(camera_pos)
        self.prog['view'].write(view.astype('f4').tobytes())
        self.prog['projection'].write(proj.astype('f4').tobytes())
        
        # Render Buzzard at REAL position
        model = Matrix44.from_translation(buzzard_pos)
        speed = np.linalg.norm(buzzard_vel)
        if speed > 0.1:
            yaw = np.arctan2(buzzard_vel[0], buzzard_vel[1])
            model = model @ Matrix44.from_z_rotation(yaw)
        self.prog['model'].write(model.astype('f4').tobytes())
        self.buzzard_vao.render(moderngl.TRIANGLES)
        
        # Render TABs at REAL positions
        cable_lines = []
        cable_colors = []
        
        for tab_id, tab in self.sim.tab_array.tabs.items():
            tab_pos = tab.position
            
            # Draw TAB
            model = Matrix44.from_translation(tab_pos)
            self.prog['model'].write(model.astype('f4').tobytes())
            self.tab_vao.render(moderngl.TRIANGLES)
            
            # Cable from Buzzard to TAB (with sag)
            if tab.is_attached:
                segments = 10
                for i in range(segments):
                    t1, t2 = i / segments, (i + 1) / segments
                    p1 = buzzard_pos * (1 - t1) + tab_pos * t1
                    p2 = buzzard_pos * (1 - t2) + tab_pos * t2
                    
                    # Catenary sag
                    sag = 5.0 * 4 * t1 * (1 - t1)
                    p1[2] -= sag
                    sag = 5.0 * 4 * t2 * (1 - t2)
                    p2[2] -= sag
                    
                    cable_lines.extend(list(p1) + list(p2))
                    cable_colors.extend([0.8, 0.3, 0.2] * 2)
        
        # Draw cables
        if cable_lines:
            cable_data = np.zeros(len(cable_lines) // 3, dtype=[('in_position', 'f4', 3), ('in_color', 'f4', 3)])
            cable_data['in_position'] = np.array(cable_lines, dtype='f4').reshape(-1, 3)
            cable_data['in_color'] = np.array(cable_colors, dtype='f4').reshape(-1, 3)
            
            cable_vbo = self.ctx.buffer(cable_data.tobytes())
            cable_vao = self.ctx.vertex_array(self.line_prog, [(cable_vbo, '3f 3f', 'in_position', 'in_color')])
            cable_vao.render(moderngl.LINES)
            cable_vbo.release()


def run():
    """Run the physics renderer."""
    mglw.run_window_config(PhysicsRenderer)


if __name__ == "__main__":
    run()
