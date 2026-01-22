"""
YAKA DERVISH Visualization - Marionette Spool System
=====================================================

ModernGL renderer for the sail constellation with:
- Central SPOOL (the marionette master / champion brain)
- Sails on cables that unravel during throw
- Tacking wind to stay aloft
"""

import numpy as np
import moderngl
import moderngl_window as mglw
from pyrr import Matrix44
from typing import Dict, List, Optional

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.physics.hubless_dervish import (
    HublessDervish,
    create_ring_constellation,
    BolaLauncher,
    LaunchPhase
)

from src.physics.marionette_spool import (
    MarionetteSpool,
    create_marionette_system,
    SpoolState
)


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
    vec3 ambient = 0.25 * v_color;
    
    vec3 norm = normalize(v_normal);
    vec3 light_dir = normalize(light_pos - v_position);
    float diff = max(dot(norm, light_dir), 0.0);
    vec3 diffuse = diff * v_color;
    
    vec3 view_dir = normalize(view_pos - v_position);
    vec3 halfway = normalize(light_dir + view_dir);
    float spec = pow(max(dot(norm, halfway), 0.0), 32.0);
    vec3 specular = 0.4 * spec * vec3(1.0);
    
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


def create_sail_mesh(width: float = 3.0, height: float = 5.0):
    """
    Create a HORIZONTAL rotor blade / sail mesh.
    
    Like the top of a "T" or a nail head - a flat horizontal surface
    that spins around to generate lift. Think helicopter blade or
    ceiling fan.
    
    The sail lies FLAT (in X-Y plane) and rotates around center.
    Slight pitch for lift generation.
    """
    vertices = []
    normals = []
    
    # Blade dimensions - long and narrow like a rotor
    blade_length = width * 2.0   # Long axis (radial direction when mounted)
    blade_width = height * 0.4   # Narrow chord
    
    # Create a flat blade with slight camber
    segments_l = 6
    segments_w = 3
    
    hl = blade_length / 2
    hw = blade_width / 2
    
    # Generate grid points - blade lies in X-Y plane
    points = []
    for j in range(segments_w + 1):
        row = []
        y = -hw + blade_width * j / segments_w
        for i in range(segments_l + 1):
            x = -hl + blade_length * i / segments_l
            
            # Slight camber (curve) for aerodynamics
            y_frac = 1.0 - abs(2.0 * j / segments_w - 1.0)
            camber = 0.05 * blade_width * y_frac
            
            # Slight twist along span (like real rotor blade)
            twist = 0.02 * blade_width * (i / segments_l - 0.5)
            
            z = camber + twist  # Small Z offset for shape
            row.append([x, y, z])
        points.append(row)
    
    # Generate triangles - TOP surface (generates lift)
    for j in range(segments_w):
        for i in range(segments_l):
            p00 = points[j][i]
            p10 = points[j][i+1]
            p01 = points[j+1][i]
            p11 = points[j+1][i+1]
            
            # Triangle 1
            vertices.extend(p00 + p10 + p11)
            normals.extend([0, 0, 1] * 3)  # Pointing UP
            
            # Triangle 2
            vertices.extend(p00 + p11 + p01)
            normals.extend([0, 0, 1] * 3)
    
    # BOTTOM surface
    for j in range(segments_w):
        for i in range(segments_l):
            p00 = points[j][i]
            p10 = points[j][i+1]
            p01 = points[j+1][i]
            p11 = points[j+1][i+1]
            
            # Reversed winding
            vertices.extend(p00 + p11 + p10)
            normals.extend([0, 0, -1] * 3)
            
            vertices.extend(p00 + p01 + p11)
            normals.extend([0, 0, -1] * 3)
    
    return np.array(vertices, dtype='f4'), np.array(normals, dtype='f4')


class DervishVisualizer(mglw.WindowConfig):
    """
    Real-time visualization of the hubless dervish constellation.
    """
    
    gl_version = (3, 3)
    title = "Hubless Dervish - Constellation Flight"
    window_size = (1280, 720)
    aspect_ratio = 16 / 9
    resizable = True
    samples = 4
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ctx.enable(moderngl.DEPTH_TEST)
        
        # Shaders
        self.prog = self.ctx.program(vertex_shader=VERTEX_SHADER, fragment_shader=FRAGMENT_SHADER)
        self.line_prog = self.ctx.program(vertex_shader=LINE_VERTEX, fragment_shader=LINE_FRAGMENT)
        
        # Create meshes
        self._create_meshes()
        self._create_grid()
        
        # Create dervish and launcher
        self._init_simulation()
        
        # Camera
        self.camera_distance = 80.0
        self.camera_height = 40.0
        self.camera_angle = 0.0
        
        # State
        self.paused = False
        self.show_trajectory = True
        self.trajectory: List[np.ndarray] = []
        
        # Physics timing
        self.physics_accumulator = 0.0
        self.physics_dt = 0.01
        
    def _init_simulation(self):
        """Initialize the dervish simulation - MARIONETTE SPOOL style."""
        print("\n=== YAKA DERVISH - MARIONETTE SPOOL CONTROL ===")
        print("Controls:")
        print("  SPACE - Pause/Resume")
        print("  R - Reset (new throw)")
        print("  T - Toggle trajectory")
        print("  Arrow keys - COMMAND (Yondu whistle)")
        print("  W/S - Altitude command")
        print()
        
        # Create the MARIONETTE SPOOL (central brain)
        self.spool = create_marionette_system(n_sails=6)
        
        # Spool starts at altitude (already deployed and hovering)
        spool_altitude = 50.0
        self.spool.position = np.array([0.0, 0.0, spool_altitude])
        self.spool.state = SpoolState.CONTROL  # Already in control mode
        
        # Create the physics constellation - SPREAD OUT in ring
        radius = 25.0  # 25m radius ring - MUCH bigger
        self.dervish = create_ring_constellation(n_nodes=6, radius=radius)
        
        # Position sails in ring around spool, ALREADY DEPLOYED
        for i, (node_id, node) in enumerate(self.dervish.nodes.items()):
            angle = 2 * np.pi * i / 6
            node.position = self.spool.position + np.array([
                radius * np.cos(angle),
                radius * np.sin(angle),
                0.0  # Same altitude as spool
            ])
            # Give initial spin velocity (tangential)
            spin_speed = 20.0  # m/s
            tangent = np.array([-np.sin(angle), np.cos(angle), 0])
            node.velocity = tangent * spin_speed
        
        # Set tether rest lengths to current radius
        for tether in self.dervish.tethers.values():
            tether.rest_length = radius * 1.05  # Ring segment length
        
        # Update spool drums
        for i, drum in enumerate(self.spool.drums.values()):
            drum.deployed_length = radius
        
        # Launcher for physics
        self.launcher = BolaLauncher(self.dervish)
        self.launcher.phase = LaunchPhase.DEPLOYED
        
        # Set pitch for lift
        self.dervish.collective_pitch = np.radians(12)
        self.dervish.cyclic_amplitude = np.radians(3)
        
        self.trajectory = []
        self.command_target = None
        self.hover_mode = True
        self.dervish.cyclic_amplitude = np.radians(3)
        
        self.trajectory = []
        self.command_target = None  # Hover position when no command
        self.hover_mode = True
        
    def _create_meshes(self):
        """Create GPU meshes."""
        # Sail mesh - BIGGER horizontal blades
        wing_v, wing_n = create_sail_mesh(width=8.0, height=12.0)
        # Sail colors - canvas white with slight tint
        wing_c = np.tile([0.95, 0.92, 0.85], len(wing_v) // 3).astype('f4')
        
        wing_data = np.zeros(len(wing_v) // 3, dtype=[
            ('in_position', 'f4', 3), ('in_normal', 'f4', 3), ('in_color', 'f4', 3)
        ])
        wing_data['in_position'] = wing_v.reshape(-1, 3)
        wing_data['in_normal'] = wing_n.reshape(-1, 3)
        wing_data['in_color'] = wing_c.reshape(-1, 3)
        
        self.node_vbo = self.ctx.buffer(wing_data.tobytes())
        self.node_vao = self.ctx.vertex_array(
            self.prog, [(self.node_vbo, '3f 3f 3f', 'in_position', 'in_normal', 'in_color')]
        )
        
    def _create_grid(self):
        """Ground grid."""
        lines = []
        colors = []
        for i in range(-500, 501, 25):
            lines.extend([i, -500, 0, i, 500, 0])
            lines.extend([-500, i, 0, 500, i, 0])
            colors.extend([0.25, 0.35, 0.25] * 4)
        
        grid_data = np.zeros(len(lines) // 3, dtype=[('in_position', 'f4', 3), ('in_color', 'f4', 3)])
        grid_data['in_position'] = np.array(lines, dtype='f4').reshape(-1, 3)
        grid_data['in_color'] = np.array(colors, dtype='f4').reshape(-1, 3)
        
        self.grid_vbo = self.ctx.buffer(grid_data.tobytes())
        self.grid_vao = self.ctx.vertex_array(self.line_prog, [(self.grid_vbo, '3f 3f', 'in_position', 'in_color')])
    
    def key_event(self, key, action, modifiers):
        """Handle keyboard - YONDU WHISTLE COMMANDS."""
        if action == self.wnd.keys.ACTION_PRESS:
            if key == self.wnd.keys.SPACE:
                self.paused = not self.paused
                print(f"{'PAUSED' if self.paused else 'RUNNING'}")
            elif key == self.wnd.keys.R:
                self._init_simulation()
                print("*WHOOSH* New throw!")
            elif key == self.wnd.keys.T:
                self.show_trajectory = not self.show_trajectory
            elif key == self.wnd.keys.UP:
                self.spool.command(np.array([0, 1, 0]), 1.0)
                print("*whistle* FORWARD!")
            elif key == self.wnd.keys.DOWN:
                self.spool.command(np.array([0, -1, 0]), 1.0)
                print("*whistle* BACK!")
            elif key == self.wnd.keys.LEFT:
                self.spool.command(np.array([-1, 0, 0]), 1.0)
                print("*whistle* LEFT!")
            elif key == self.wnd.keys.RIGHT:
                self.spool.command(np.array([1, 0, 0]), 1.0)
                print("*whistle* RIGHT!")
            elif key == self.wnd.keys.W:
                self.spool.command(np.array([0, 0, 1]), 1.0)
                print("*whistle* UP!")
            elif key == self.wnd.keys.S:
                self.spool.command(np.array([0, 0, -0.5]), 0.5)
                print("*whistle* DOWN!")
    
    def on_render(self, time: float, frame_time: float):
        """Render frame."""
        # Step physics with SPOOL CONTROL
        if not self.paused:
            self.physics_accumulator += frame_time
            steps = 0
            while self.physics_accumulator >= self.physics_dt and steps < 10:
                centroid = self.dervish.compute_centroid()
                
                # Build physics state for spool
                sail_positions = {f"sail_{i}": node.position.copy() 
                                 for i, node in enumerate(self.dervish.nodes.values())}
                sail_forces = {f"sail_{i}": node.aero_force.copy()
                              for i, node in enumerate(self.dervish.nodes.values())}
                
                # Step the SPOOL (marionette master)
                cable_lengths = self.spool.step(self.physics_dt, sail_forces, sail_positions)
                
                # Update tether rest lengths from spool (the control mechanism!)
                for i, tether in enumerate(self.dervish.tethers.values()):
                    sail_id = f"sail_{i % len(cable_lengths)}"
                    if sail_id in cable_lengths:
                        # Spool controls cable length → affects tether tension → controls pitch
                        tether.rest_length = max(1.0, cable_lengths[sail_id])
                
                # Maintain spin - sails tacking the wind
                for node in self.dervish.nodes.values():
                    r = node.position - centroid
                    r[2] = 0
                    r_mag = np.linalg.norm(r)
                    if r_mag > 0.5:
                        tangent = np.array([-r[1], r[0], 0]) / r_mag
                        current_spin = np.dot(node.velocity, tangent)
                        target_spin = 12.0
                        if current_spin < target_spin:
                            node.velocity += tangent * 0.3 * self.physics_dt * 60
                
                # Update spool spin phase to match constellation
                self.spool.spin_phase = self.dervish.spin_phase
                
                self.launcher.step(self.physics_dt, np.zeros(3))
                self.physics_accumulator -= self.physics_dt
                steps += 1
            
            # Record trajectory
            if len(self.trajectory) == 0 or steps > 0:
                centroid = self.dervish.compute_centroid()
                if not np.any(np.isnan(centroid)):
                    self.trajectory.append(centroid.copy())
                    if len(self.trajectory) > 500:
                        self.trajectory.pop(0)
        
        # Camera follows centroid
        centroid = self.dervish.compute_centroid()
        if np.any(np.isnan(centroid)):
            centroid = np.array([0, 0, 5])
        
        # Calculate constellation size (for camera distance)
        max_dist = 0
        for node in self.dervish.nodes.values():
            d = np.linalg.norm(node.position - centroid)
            max_dist = max(max_dist, d)
        
        # Camera pulls back to see full ring
        target_distance = max(60.0, max_dist * 3)
        self.camera_distance += (target_distance - self.camera_distance) * 0.02
        
        # Orbit around - HIGHER to look down at the spinning ring
        self.camera_angle = time * 0.15
        cam_offset = np.array([
            np.cos(self.camera_angle) * self.camera_distance * 0.7,
            np.sin(self.camera_angle) * self.camera_distance * 0.7,
            self.camera_distance * 0.8  # HIGH up looking down
        ])
        camera_pos = centroid + cam_offset
        
        # Clear
        self.ctx.clear(0.05, 0.08, 0.12)
        self.ctx.enable(moderngl.DEPTH_TEST)
        
        # Matrices
        proj = Matrix44.perspective_projection(60.0, self.aspect_ratio, 0.1, 2000.0)
        view = Matrix44.look_at(tuple(camera_pos), tuple(centroid), (0, 0, 1))
        mvp = proj * view
        
        # Grid
        self.line_prog['mvp'].write(mvp.astype('f4').tobytes())
        self.grid_vao.render(moderngl.LINES)
        
        # Lighting
        self.prog['light_pos'].value = tuple(camera_pos + np.array([100, 100, 200]))
        self.prog['view_pos'].value = tuple(camera_pos)
        self.prog['view'].write(view.astype('f4').tobytes())
        self.prog['projection'].write(proj.astype('f4').tobytes())
        
        # Render sails - HORIZONTAL blades spinning like rotor
        for node in self.dervish.nodes.values():
            if np.any(np.isnan(node.position)):
                continue
            
            model = Matrix44.from_translation(node.position)
            
            # Sail is HORIZONTAL (like top of T or nail head)
            # Blade extends radially outward from center
            r = node.position - centroid
            r[2] = 0
            r_mag = np.linalg.norm(r)
            
            if r_mag > 0.1:
                # Yaw: blade points RADIALLY (outward from center)
                radial_angle = np.arctan2(r[1], r[0])
                
                # Pitch: slight nose-up for lift (like helicopter blade pitch)
                pitch = self.dervish.collective_pitch
                
                # The blade is already horizontal in mesh (X-Y plane)
                # Rotate to point radially, then pitch for lift
                model = model @ Matrix44.from_z_rotation(radial_angle) @ Matrix44.from_y_rotation(pitch)
            
            self.prog['model'].write(model.astype('f4').tobytes())
            self.node_vao.render(moderngl.TRIANGLES)
        
        # Render SPOOL (the marionette master) at centroid
        self._render_spool(mvp, centroid)
        
        # Render tethers (FROM SPOOL to sails)
        self._render_tethers(mvp, centroid)
        
        # Render trajectory
        if self.show_trajectory and len(self.trajectory) > 1:
            self._render_trajectory(mvp)
    
    def _render_spool(self, mvp, centroid):
        """Render the central spool mechanism."""
        # Spool is a small box at the centroid
        spool_pos = centroid
        
        # Create spool cube vertices
        size = 0.5
        lines = []
        colors = []
        
        # Draw as wireframe cube
        for dx in [-1, 1]:
            for dy in [-1, 1]:
                for dz in [-1, 1]:
                    p = spool_pos + np.array([dx, dy, dz]) * size
                    for ax in range(3):
                        p2 = p.copy()
                        p2[ax] *= -1
                        p2 = spool_pos + (p2 - spool_pos)
                        lines.extend(list(p) + list(p2))
                        # Gold color for the marionette master
                        colors.extend([1.0, 0.8, 0.2] * 2)
        
        if lines:
            data = np.zeros(len(lines) // 3, dtype=[('in_position', 'f4', 3), ('in_color', 'f4', 3)])
            data['in_position'] = np.array(lines, dtype='f4').reshape(-1, 3)
            data['in_color'] = np.array(colors, dtype='f4').reshape(-1, 3)
            
            vbo = self.ctx.buffer(data.tobytes())
            vao = self.ctx.vertex_array(self.line_prog, [(vbo, '3f 3f', 'in_position', 'in_color')])
            self.line_prog['mvp'].write(mvp.astype('f4').tobytes())
            vao.render(moderngl.LINES)
            vbo.release()

    def _render_tethers(self, mvp, spool_pos=None):
        """Render cables FROM SPOOL to each sail."""
        lines = []
        colors = []
        
        # Use spool position (centroid) as cable origin
        if spool_pos is None:
            spool_pos = self.dervish.compute_centroid()
        
        # Draw cables from spool to each sail
        for i, node in enumerate(self.dervish.nodes.values()):
            if np.any(np.isnan(node.position)):
                continue
            
            # Cable from spool to sail
            lines.extend(list(spool_pos) + list(node.position))
            
            # Color based on spool drum tension
            sail_id = f"sail_{i}"
            if sail_id in self.spool.drums:
                tension = self.spool.drums[sail_id].tension
                tension_frac = min(tension / 300.0, 1.0)
            else:
                tension_frac = 0.3
            
            # Cables: silver-grey with tension coloring
            color = (0.6 + tension_frac * 0.4, 0.6 - tension_frac * 0.3, 0.5)
            colors.extend(list(color) * 2)
        
        # Also draw sail-to-sail tethers (the ring structure)
        for tether in self.dervish.tethers.values():
            node_a = self.dervish.nodes.get(tether.node_a)
            node_b = self.dervish.nodes.get(tether.node_b)
            if not node_a or not node_b:
                continue
            
            if np.any(np.isnan(node_a.position)) or np.any(np.isnan(node_b.position)):
                continue
            
            lines.extend(list(node_a.position) + list(node_b.position))
            
            # Ring tethers: blue-ish
            tension_frac = min(tether.tension / 500.0, 1.0)
            color = (0.2, 0.5 + tension_frac * 0.3, 0.8)
            colors.extend(list(color) * 2)
        
        if not lines:
            return
        
        line_data = np.zeros(len(lines) // 3, dtype=[('in_position', 'f4', 3), ('in_color', 'f4', 3)])
        line_data['in_position'] = np.array(lines, dtype='f4').reshape(-1, 3)
        line_data['in_color'] = np.array(colors, dtype='f4').reshape(-1, 3)
        
        vbo = self.ctx.buffer(line_data.tobytes())
        vao = self.ctx.vertex_array(self.line_prog, [(vbo, '3f 3f', 'in_position', 'in_color')])
        vao.render(moderngl.LINES)
        vbo.release()
    
    def _render_trajectory(self, mvp):
        """Render trajectory trail."""
        if len(self.trajectory) < 2:
            return
        
        lines = []
        colors = []
        
        for i in range(len(self.trajectory) - 1):
            p1 = self.trajectory[i]
            p2 = self.trajectory[i + 1]
            
            if np.any(np.isnan(p1)) or np.any(np.isnan(p2)):
                continue
            
            lines.extend(list(p1) + list(p2))
            
            # Fade older segments
            age = i / len(self.trajectory)
            colors.extend([0.2 + 0.6 * age, 0.4 + 0.4 * age, 0.8] * 2)
        
        if not lines:
            return
        
        line_data = np.zeros(len(lines) // 3, dtype=[('in_position', 'f4', 3), ('in_color', 'f4', 3)])
        line_data['in_position'] = np.array(lines, dtype='f4').reshape(-1, 3)
        line_data['in_color'] = np.array(colors, dtype='f4').reshape(-1, 3)
        
        vbo = self.ctx.buffer(line_data.tobytes())
        vao = self.ctx.vertex_array(self.line_prog, [(vbo, '3f 3f', 'in_position', 'in_color')])
        vao.render(moderngl.LINES)
        vbo.release()


def run():
    """Run the visualizer."""
    mglw.run_window_config(DervishVisualizer)


if __name__ == "__main__":
    run()
