"""
SPINNING SAILS DERVISH - Clean Visualization
=============================================

A ring of sails spinning around a central spool.
Each sail is at the END of a cable, spinning to generate lift.
The system is OMNIDIRECTIONAL - sails can pitch to thrust any direction.

     TOP VIEW:
                    ⎯⎯⎯ sail
                   ╱
                  ╱ cable
       sail ⎯⎯⎯ ● ⎯⎯⎯ sail   (spinning clockwise)
                  ╲
                   ╲ cable
                    ⎯⎯⎯ sail

     SIDE VIEW:
                ═══════════════════  sails (horizontal blades)
                    │   │   │        cables
                    └───┼───┘
                        ●            spool (center)
"""

import numpy as np
import moderngl
import moderngl_window as mglw
from pyrr import Matrix44


VERTEX_SHADER = """
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

FRAGMENT_SHADER = """
#version 330
in vec3 v_color;
out vec4 fragColor;
void main() {
    fragColor = vec4(v_color, 1.0);
}
"""


class SpinningSailsViz(mglw.WindowConfig):
    """
    Clean visualization of spinning sail dervish.
    """
    
    gl_version = (3, 3)
    title = "Spinning Sails Dervish"
    window_size = (1400, 900)
    aspect_ratio = None
    resizable = True
    samples = 4
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        self.prog = self.ctx.program(
            vertex_shader=VERTEX_SHADER,
            fragment_shader=FRAGMENT_SHADER
        )
        
        # System parameters
        self.n_sails = 6
        self.cable_length = 20.0      # Length of each cable
        self.sail_width = 6.0         # Width of sail blade
        self.sail_chord = 1.5         # Depth of sail
        self.spool_altitude = 50.0    # Spool height
        self.cone_angle = 0.4         # Cables angle down (radians) - like spinning skirt
        
        # Dynamics
        self.spin_rate = 1.5          # rad/s rotation
        self.spin_angle = 0.0         # Current rotation angle
        self.collective_pitch = 0.2   # Sail pitch for lift (radians)
        self.cyclic_phase = 0.0       # For directional thrust
        self.cyclic_amp = 0.0         # Cyclic amplitude
        
        # Camera
        self.cam_angle = 0.0
        self.cam_distance = 80.0
        self.cam_height = 30.0
        
        # State
        self.paused = False
        self.time = 0.0
        
        self._create_ground()
        
        print("\n=== SPINNING SAILS DERVISH ===")
        print("Controls:")
        print("  SPACE - Pause/Resume spin")
        print("  Arrow keys - Command direction (cyclic)")
        print("  W/S - Increase/decrease lift")
        print("  +/- - Spin faster/slower")
        print("  R - Reset")
        print()
    
    def _create_ground(self):
        """Create ground grid."""
        lines = []
        colors = []
        
        for i in range(-200, 201, 20):
            # X lines
            lines.extend([i, -200, 0, i, 200, 0])
            colors.extend([0.2, 0.3, 0.2] * 2)
            # Y lines
            lines.extend([-200, i, 0, 200, i, 0])
            colors.extend([0.2, 0.3, 0.2] * 2)
        
        data = np.zeros(len(lines) // 3, dtype=[
            ('in_position', 'f4', 3), ('in_color', 'f4', 3)
        ])
        data['in_position'] = np.array(lines, dtype='f4').reshape(-1, 3)
        data['in_color'] = np.array(colors, dtype='f4').reshape(-1, 3)
        
        self.ground_vbo = self.ctx.buffer(data.tobytes())
        self.ground_vao = self.ctx.vertex_array(
            self.prog, [(self.ground_vbo, '3f 3f', 'in_position', 'in_color')]
        )
    
    def key_event(self, key, action, modifiers):
        if action != self.wnd.keys.ACTION_PRESS:
            return
        
        if key == self.wnd.keys.SPACE:
            self.paused = not self.paused
            print("PAUSED" if self.paused else "SPINNING")
        elif key == self.wnd.keys.R:
            self.spin_angle = 0
            self.cyclic_amp = 0
            self.collective_pitch = 0.15
            print("RESET")
        elif key == self.wnd.keys.UP:
            self.cyclic_amp = 0.2
            self.cyclic_phase = np.pi / 2  # Forward
            print("→ FORWARD thrust")
        elif key == self.wnd.keys.DOWN:
            self.cyclic_amp = 0.2
            self.cyclic_phase = -np.pi / 2  # Back
            print("→ BACKWARD thrust")
        elif key == self.wnd.keys.LEFT:
            self.cyclic_amp = 0.2
            self.cyclic_phase = np.pi  # Left
            print("→ LEFT thrust")
        elif key == self.wnd.keys.RIGHT:
            self.cyclic_amp = 0.2
            self.cyclic_phase = 0  # Right
            print("→ RIGHT thrust")
        elif key == self.wnd.keys.W:
            self.collective_pitch += 0.05
            print(f"Collective: {np.degrees(self.collective_pitch):.1f}°")
        elif key == self.wnd.keys.S:
            self.collective_pitch -= 0.05
            print(f"Collective: {np.degrees(self.collective_pitch):.1f}°")
        elif key == self.wnd.keys.EQUAL:  # +
            self.spin_rate *= 1.2
            print(f"Spin: {self.spin_rate:.1f} rad/s")
        elif key == self.wnd.keys.MINUS:
            self.spin_rate *= 0.8
            print(f"Spin: {self.spin_rate:.1f} rad/s")
    
    def _build_sail_geometry(self, position, angle, pitch):
        """
        Build a single sail blade at given position.
        
        The sail is a horizontal blade (like top of T):
        - Extends perpendicular to the cable
        - Pitched to generate lift
        """
        verts = []
        colors = []
        
        # Sail corners in local coords (flat horizontal blade)
        hw = self.sail_width / 2
        hc = self.sail_chord / 2
        
        # 4 corners of the sail
        corners = [
            np.array([-hw, -hc, 0]),
            np.array([hw, -hc, 0]),
            np.array([hw, hc, 0]),
            np.array([-hw, hc, 0]),
        ]
        
        # Rotation: align blade perpendicular to radius, then pitch
        # Tangent direction (perpendicular to radius)
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        
        # Rotation matrix: first rotate to tangent, then pitch
        cos_p, sin_p = np.cos(pitch), np.sin(pitch)
        
        def transform(p):
            # Pitch around tangent axis (tilt leading edge up)
            x, y, z = p
            y2 = y * cos_p - z * sin_p
            z2 = y * sin_p + z * cos_p
            
            # Rotate to align with tangent (blade perpendicular to radius)
            x3 = x * cos_a - y2 * sin_a
            y3 = x * sin_a + y2 * cos_a
            
            return position + np.array([x3, y3, z2])
        
        # Transform corners
        tc = [transform(c) for c in corners]
        
        # Top surface triangles
        for tri in [(0, 1, 2), (0, 2, 3)]:
            for i in tri:
                verts.extend(tc[i])
                colors.extend([0.95, 0.9, 0.8])  # Sail canvas color
        
        # Bottom surface
        for tri in [(0, 2, 1), (0, 3, 2)]:
            for i in tri:
                verts.extend(tc[i])
                colors.extend([0.7, 0.65, 0.6])  # Darker underside
        
        return verts, colors
    
    def _build_cable(self, start, end, color):
        """Build a cable line."""
        return list(start) + list(end), list(color) * 2
    
    def on_render(self, time_val: float, frame_time: float):
        self.ctx.clear(0.02, 0.05, 0.1)
        self.ctx.enable(moderngl.DEPTH_TEST)
        
        # Update spin
        if not self.paused:
            self.spin_angle += self.spin_rate * frame_time
            self.time += frame_time
            
            # Cone angle depends on spin rate - more spin = more spread out
            # Like a spinning skirt lifting up
            target_cone = 0.3 + self.spin_rate * 0.25  # radians
            target_cone = min(target_cone, 1.2)  # Max ~70 degrees
            self.cone_angle += (target_cone - self.cone_angle) * 0.02
            
            # Decay cyclic command
            self.cyclic_amp *= 0.995
        
        # Camera orbits slowly
        self.cam_angle = time_val * 0.1
        
        cam_pos = np.array([
            self.cam_distance * np.cos(self.cam_angle),
            self.cam_distance * np.sin(self.cam_angle),
            self.cam_height
        ])
        center = np.array([0, 0, self.spool_altitude - 10])  # Look at middle of structure
        
        # Matrices
        proj = Matrix44.perspective_projection(50.0, self.wnd.aspect_ratio, 0.1, 1000.0)
        view = Matrix44.look_at(tuple(cam_pos + np.array([0, 0, self.spool_altitude - 10])), tuple(center + np.array([0, 0, self.spool_altitude - 10])), (0, 0, 1))
        mvp = proj * view
        
        self.prog['mvp'].write(mvp.astype('f4').tobytes())
        
        # Draw ground
        self.ground_vao.render(moderngl.LINES)
        
        # Build dynamic geometry for sails and cables
        all_verts = []
        all_colors = []
        cable_verts = []
        cable_colors = []
        
        # Spool at top
        spool_pos = np.array([0, 0, self.spool_altitude])
        
        for i in range(self.n_sails):
            # Angular position of this sail (spinning!)
            base_angle = 2 * np.pi * i / self.n_sails
            sail_angle = base_angle + self.spin_angle
            
            # Cables hang DOWN and OUT - like a spinning skirt/carousel
            # Centrifugal force spreads them out
            horizontal_dist = self.cable_length * np.sin(self.cone_angle)
            vertical_drop = self.cable_length * np.cos(self.cone_angle)
            
            sail_pos = spool_pos + np.array([
                horizontal_dist * np.cos(sail_angle),
                horizontal_dist * np.sin(sail_angle),
                -vertical_drop  # BELOW the spool
            ])
            
            # Pitch: collective + cyclic
            # Cyclic varies with angular position for directional thrust
            cyclic = self.cyclic_amp * np.sin(sail_angle - self.cyclic_phase)
            pitch = self.collective_pitch + cyclic
            
            # Build sail
            sv, sc = self._build_sail_geometry(sail_pos, sail_angle, pitch)
            all_verts.extend(sv)
            all_colors.extend(sc)
            
            # Cable from spool to sail
            cv, cc = self._build_cable(spool_pos, sail_pos, (0.8, 0.6, 0.2))
            cable_verts.extend(cv)
            cable_colors.extend(cc)
        
        # Draw sails (triangles)
        if all_verts:
            sail_data = np.zeros(len(all_verts) // 3, dtype=[
                ('in_position', 'f4', 3), ('in_color', 'f4', 3)
            ])
            sail_data['in_position'] = np.array(all_verts, dtype='f4').reshape(-1, 3)
            sail_data['in_color'] = np.array(all_colors, dtype='f4').reshape(-1, 3)
            
            vbo = self.ctx.buffer(sail_data.tobytes())
            vao = self.ctx.vertex_array(self.prog, [(vbo, '3f 3f', 'in_position', 'in_color')])
            vao.render(moderngl.TRIANGLES)
            vbo.release()
        
        # Draw cables (lines)
        if cable_verts:
            cable_data = np.zeros(len(cable_verts) // 3, dtype=[
                ('in_position', 'f4', 3), ('in_color', 'f4', 3)
            ])
            cable_data['in_position'] = np.array(cable_verts, dtype='f4').reshape(-1, 3)
            cable_data['in_color'] = np.array(cable_colors, dtype='f4').reshape(-1, 3)
            
            vbo = self.ctx.buffer(cable_data.tobytes())
            vao = self.ctx.vertex_array(self.prog, [(vbo, '3f 3f', 'in_position', 'in_color')])
            vao.render(moderngl.LINES)
            vbo.release()
        
        # Draw spool (center point)
        self._draw_spool(mvp, spool_pos)
        
        # Draw ring connecting sails (the outer ring structure)
        self._draw_outer_ring(mvp, spool_pos)
    
    def _draw_spool(self, mvp, pos):
        """Draw the central spool."""
        # Simple cross at center
        s = 1.0
        lines = [
            pos[0]-s, pos[1], pos[2], pos[0]+s, pos[1], pos[2],
            pos[0], pos[1]-s, pos[2], pos[0], pos[1]+s, pos[2],
            pos[0], pos[1], pos[2]-s, pos[0], pos[1], pos[2]+s,
        ]
        colors = [1, 0.8, 0.2] * 6  # Gold
        
        data = np.zeros(6, dtype=[('in_position', 'f4', 3), ('in_color', 'f4', 3)])
        data['in_position'] = np.array(lines, dtype='f4').reshape(-1, 3)
        data['in_color'] = np.array(colors, dtype='f4').reshape(-1, 3)
        
        vbo = self.ctx.buffer(data.tobytes())
        vao = self.ctx.vertex_array(self.prog, [(vbo, '3f 3f', 'in_position', 'in_color')])
        vao.render(moderngl.LINES)
        vbo.release()
    
    def _draw_outer_ring(self, mvp, spool_pos):
        """Draw ring connecting sail tips."""
        lines = []
        colors = []
        
        horizontal_dist = self.cable_length * np.sin(self.cone_angle)
        vertical_drop = self.cable_length * np.cos(self.cone_angle)
        
        for i in range(self.n_sails):
            a1 = 2 * np.pi * i / self.n_sails + self.spin_angle
            a2 = 2 * np.pi * ((i+1) % self.n_sails) / self.n_sails + self.spin_angle
            
            p1 = spool_pos + np.array([
                horizontal_dist * np.cos(a1),
                horizontal_dist * np.sin(a1),
                -vertical_drop
            ])
            p2 = spool_pos + np.array([
                horizontal_dist * np.cos(a2),
                horizontal_dist * np.sin(a2),
                -vertical_drop
            ])
            
            lines.extend(list(p1) + list(p2))
            colors.extend([0.3, 0.5, 0.8] * 2)  # Blue ring
        
        data = np.zeros(len(lines)//3, dtype=[('in_position', 'f4', 3), ('in_color', 'f4', 3)])
        data['in_position'] = np.array(lines, dtype='f4').reshape(-1, 3)
        data['in_color'] = np.array(colors, dtype='f4').reshape(-1, 3)
        
        vbo = self.ctx.buffer(data.tobytes())
        vao = self.ctx.vertex_array(self.prog, [(vbo, '3f 3f', 'in_position', 'in_color')])
        vao.render(moderngl.LINES)
        vbo.release()


if __name__ == '__main__':
    mglw.run_window_config(SpinningSailsViz)
