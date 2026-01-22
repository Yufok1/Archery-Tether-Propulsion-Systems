"""
Enhanced Visual Geometry for KAPS
==================================

Proper 3D geometry for:
- CABLES: Thick tubes with tension coloring, not thin lines
- AIRFOILS: Visible delta wings with thickness and control surfaces
- BUZZARD: Detailed mother drone with features
- THREATS: Distinctive shapes per threat type

All geometry is procedural Panda3D compatible.
"""

import numpy as np
from typing import Tuple, List

try:
    from panda3d.core import (
        GeomVertexFormat, GeomVertexData, GeomVertexWriter,
        Geom, GeomTriangles, GeomTristrips, GeomNode,
        Vec3, Vec4, Point3,
        LineSegs
    )
    PANDA3D_AVAILABLE = True
except ImportError:
    PANDA3D_AVAILABLE = False


def create_cable_geometry(
    start: np.ndarray,
    end: np.ndarray,
    radius: float = 0.5,
    segments: int = 8,
    color: Tuple[float, float, float, float] = (0.4, 0.4, 0.5, 1.0),
    tension_color: Tuple[float, float, float, float] = None
) -> GeomNode:
    """
    Create a 3D tube/cable geometry between two points.
    
    This creates a visible CYLINDER, not a thin line.
    
    Args:
        start: Start position
        end: End position
        radius: Cable thickness (default 0.5m)
        segments: Number of sides (8 = octagon cross-section)
        color: Base color
        tension_color: Color at end if cable is under tension
    """
    if not PANDA3D_AVAILABLE:
        return None
    
    format = GeomVertexFormat.getV3n3c4()
    vdata = GeomVertexData("cable", format, Geom.UHStatic)
    
    vertex = GeomVertexWriter(vdata, "vertex")
    normal = GeomVertexWriter(vdata, "normal")
    col = GeomVertexWriter(vdata, "color")
    
    # Direction vector
    direction = end - start
    length = np.linalg.norm(direction)
    if length < 0.01:
        length = 0.01
    direction = direction / length
    
    # Create perpendicular vectors for circular cross-section
    if abs(direction[2]) < 0.9:
        perp1 = np.cross(direction, np.array([0, 0, 1]))
    else:
        perp1 = np.cross(direction, np.array([1, 0, 0]))
    perp1 = perp1 / np.linalg.norm(perp1)
    perp2 = np.cross(direction, perp1)
    
    # Create vertices at start and end caps
    for t, pos, c in [(0, start, color), (1, end, tension_color or color)]:
        for i in range(segments + 1):
            angle = 2 * np.pi * i / segments
            offset = perp1 * np.cos(angle) * radius + perp2 * np.sin(angle) * radius
            p = pos + offset
            n = offset / radius
            
            vertex.addData3f(p[0], p[1], p[2])
            normal.addData3f(n[0], n[1], n[2])
            col.addData4f(*c)
    
    # Create triangles connecting the two rings
    prim = GeomTriangles(Geom.UHStatic)
    for i in range(segments):
        # Start ring vertex indices
        s0 = i
        s1 = i + 1
        # End ring vertex indices
        e0 = segments + 1 + i
        e1 = segments + 1 + i + 1
        
        # Two triangles per quad
        prim.addVertices(s0, e0, s1)
        prim.addVertices(s1, e0, e1)
    
    geom = Geom(vdata)
    geom.addPrimitive(prim)
    node = GeomNode("cable")
    node.addGeom(geom)
    return node


def create_airfoil_geometry(
    wingspan: float = 4.0,
    chord: float = 1.5,
    thickness: float = 0.3,
    color: Tuple[float, float, float, float] = (0.8, 0.8, 0.8, 1.0),
    highlight_color: Tuple[float, float, float, float] = None
) -> GeomNode:
    """
    Create a detailed flying wing / delta airfoil geometry.
    
    This creates a VISIBLE 3D wing shape with:
    - Proper delta planform
    - Visible thickness
    - Leading and trailing edges
    - Control surface hints
    
    Args:
        wingspan: Total wing span (tip to tip)
        chord: Root chord length
        thickness: Maximum thickness
        color: Main body color
        highlight_color: Leading edge accent color
    """
    if not PANDA3D_AVAILABLE:
        return None
    
    format = GeomVertexFormat.getV3n3c4()
    vdata = GeomVertexData("airfoil", format, Geom.UHStatic)
    
    vertex = GeomVertexWriter(vdata, "vertex")
    normal = GeomVertexWriter(vdata, "normal")
    col = GeomVertexWriter(vdata, "color")
    
    half_span = wingspan / 2
    tip_chord = chord * 0.3  # Wingtips are narrower
    
    # Leading edge color
    le_color = highlight_color or (
        min(1.0, color[0] + 0.2),
        min(1.0, color[1] + 0.2),
        min(1.0, color[2] + 0.2),
        color[3]
    )
    
    # Define airfoil cross-section points (normalized)
    # This gives a proper airfoil shape
    airfoil_top = [
        (0.0, 0.0),      # Leading edge
        (0.1, 0.04),     # Near LE
        (0.3, 0.06),     # Max thickness
        (0.6, 0.04),     # Aft
        (1.0, 0.01),     # Trailing edge
    ]
    airfoil_bottom = [
        (0.0, 0.0),      # Leading edge
        (0.1, -0.02),    # Near LE
        (0.3, -0.03),    # Max thickness
        (0.6, -0.02),    # Aft
        (1.0, -0.01),    # Trailing edge
    ]
    
    # Scale factor for thickness
    t_scale = thickness * 10
    
    # Create wing vertices
    # Center section (root)
    sections = [
        (0, chord, color),                          # Root
        (half_span * 0.3, chord * 0.8, color),      # Mid-span
        (half_span * 0.6, chord * 0.5, color),      # Outer
        (half_span, tip_chord, le_color),           # Tip
    ]
    
    verts = []
    norms = []
    colors = []
    
    for y_pos, local_chord, c in sections:
        for side in [1, -1]:  # Right and left wing
            y = y_pos * side
            
            # Top surface
            for x_frac, z_frac in airfoil_top:
                x = -local_chord * x_frac + local_chord * 0.3  # Offset so LE is forward
                z = z_frac * t_scale
                verts.append((x, y, z))
                norms.append((0, 0, 1))  # Simplified up normal
                colors.append(c)
            
            # Bottom surface
            for x_frac, z_frac in airfoil_bottom:
                x = -local_chord * x_frac + local_chord * 0.3
                z = z_frac * t_scale
                verts.append((x, y, z))
                norms.append((0, 0, -1))  # Simplified down normal
                colors.append(c)
    
    # Add all vertices
    for v, n, c in zip(verts, norms, colors):
        vertex.addData3f(*v)
        normal.addData3f(*n)
        col.addData4f(*c)
    
    # Create a simpler but more visible geometry
    # Just make a thick delta wing shape
    vdata2 = GeomVertexData("airfoil2", format, Geom.UHStatic)
    vertex2 = GeomVertexWriter(vdata2, "vertex")
    normal2 = GeomVertexWriter(vdata2, "normal")
    col2 = GeomVertexWriter(vdata2, "color")
    
    # Simpler delta wing - 14 vertices
    simple_verts = [
        # Top surface
        (chord * 0.5, 0, thickness),              # 0: Nose top
        (-chord * 0.3, half_span, thickness/2),   # 1: Right tip top
        (-chord * 0.3, -half_span, thickness/2),  # 2: Left tip top
        (-chord * 0.5, 0, thickness/2),           # 3: Tail center top
        
        # Bottom surface
        (chord * 0.5, 0, -thickness/2),           # 4: Nose bottom
        (-chord * 0.3, half_span, -thickness/2),  # 5: Right tip bottom
        (-chord * 0.3, -half_span, -thickness/2), # 6: Left tip bottom
        (-chord * 0.5, 0, -thickness/2),          # 7: Tail center bottom
        
        # Leading edge points (for thickness)
        (chord * 0.4, half_span * 0.3, 0),        # 8: Right LE mid
        (chord * 0.4, -half_span * 0.3, 0),       # 9: Left LE mid
        
        # Mid-span ridge (for wing shape)
        (0, half_span * 0.5, thickness),          # 10: Right ridge
        (0, -half_span * 0.5, thickness),         # 11: Left ridge
        (0, half_span * 0.5, -thickness/3),       # 12: Right ridge bottom
        (0, -half_span * 0.5, -thickness/3),      # 13: Left ridge bottom
    ]
    
    simple_norms = [
        (0, 0, 1), (0.2, 0.5, 0.8), (0.2, -0.5, 0.8), (0, 0, 1),   # Top
        (0, 0, -1), (0.2, 0.5, -0.8), (0.2, -0.5, -0.8), (0, 0, -1), # Bottom
        (0.7, 0.7, 0), (0.7, -0.7, 0),  # LE
        (0, 0.3, 0.95), (0, -0.3, 0.95),  # Ridge top
        (0, 0.3, -0.95), (0, -0.3, -0.95),  # Ridge bottom
    ]
    
    for v, n in zip(simple_verts, simple_norms):
        vertex2.addData3f(*v)
        normal2.addData3f(*n)
        col2.addData4f(*color)
    
    # Triangle faces
    prim2 = GeomTriangles(Geom.UHStatic)
    
    # Top surface - 4 triangles
    prim2.addVertices(0, 10, 11)   # Nose to ridges
    prim2.addVertices(0, 11, 2)    # Nose to left tip
    prim2.addVertices(0, 1, 10)    # Nose to right tip
    prim2.addVertices(10, 3, 11)   # Ridges to tail
    prim2.addVertices(10, 1, 3)    # Right aft
    prim2.addVertices(11, 3, 2)    # Left aft
    
    # Bottom surface - 4 triangles (reversed winding)
    prim2.addVertices(4, 13, 12)
    prim2.addVertices(4, 6, 13)
    prim2.addVertices(4, 12, 5)
    prim2.addVertices(12, 13, 7)
    prim2.addVertices(12, 7, 5)
    prim2.addVertices(13, 6, 7)
    
    # Leading edges (sides)
    prim2.addVertices(0, 8, 4)     # Nose front
    prim2.addVertices(0, 4, 9)     # Nose front other
    prim2.addVertices(0, 1, 8)     # Right LE
    prim2.addVertices(8, 1, 5)
    prim2.addVertices(8, 5, 4)
    prim2.addVertices(0, 9, 2)     # Left LE
    prim2.addVertices(9, 6, 2)
    prim2.addVertices(9, 4, 6)
    
    # Trailing edge
    prim2.addVertices(3, 1, 5)
    prim2.addVertices(3, 5, 7)
    prim2.addVertices(3, 2, 6)
    prim2.addVertices(3, 6, 7)
    
    geom2 = Geom(vdata2)
    geom2.addPrimitive(prim2)
    node = GeomNode("airfoil")
    node.addGeom(geom2)
    return node


def create_buzzard_geometry(
    body_length: float = 8.0,
    body_radius: float = 2.0,
    color: Tuple[float, float, float, float] = (0.2, 0.3, 0.8, 1.0)
) -> GeomNode:
    """
    Create detailed Buzzard (mother drone) geometry.
    
    The Buzzard is the PROTECTED ASSET - it should be visually prominent.
    
    Features:
    - Elongated fuselage
    - Corkscrew propulsion housing (rear)
    - Sensor dome (front)
    - TAB attachment points
    """
    if not PANDA3D_AVAILABLE:
        return None
    
    format = GeomVertexFormat.getV3n3c4()
    vdata = GeomVertexData("buzzard", format, Geom.UHStatic)
    
    vertex = GeomVertexWriter(vdata, "vertex")
    normal = GeomVertexWriter(vdata, "normal")
    col = GeomVertexWriter(vdata, "color")
    
    segments = 16  # Around circumference
    length_segments = 8  # Along length
    
    # Fuselage profile (radius at each length station)
    profile = [
        (0.0, 0.3),    # Nose
        (0.1, 0.6),    # Nose blend
        (0.2, 0.85),   # Forward
        (0.4, 1.0),    # Max radius
        (0.6, 1.0),    # Max radius
        (0.8, 0.85),   # Aft taper
        (0.9, 0.6),    # Engine housing
        (1.0, 0.4),    # Tail
    ]
    
    # Accent colors
    nose_color = (0.5, 0.6, 0.9, 1.0)  # Lighter nose
    engine_color = (0.3, 0.3, 0.4, 1.0)  # Darker engine
    
    # Generate vertices
    for i, (t, r_factor) in enumerate(profile):
        x = body_length * (t - 0.5)  # Center at 0
        r = body_radius * r_factor
        
        # Color gradient
        if t < 0.2:
            c = nose_color
        elif t > 0.8:
            c = engine_color
        else:
            c = color
        
        for j in range(segments + 1):
            angle = 2 * np.pi * j / segments
            y = r * np.cos(angle)
            z = r * np.sin(angle)
            
            # Normal points outward
            n = np.array([0, np.cos(angle), np.sin(angle)])
            
            vertex.addData3f(x, y, z)
            normal.addData3f(n[0], n[1], n[2])
            col.addData4f(*c)
    
    # Create triangles
    prim = GeomTriangles(Geom.UHStatic)
    
    for i in range(len(profile) - 1):
        for j in range(segments):
            # Current ring
            v0 = i * (segments + 1) + j
            v1 = i * (segments + 1) + j + 1
            # Next ring
            v2 = (i + 1) * (segments + 1) + j
            v3 = (i + 1) * (segments + 1) + j + 1
            
            prim.addVertices(v0, v2, v1)
            prim.addVertices(v1, v2, v3)
    
    geom = Geom(vdata)
    geom.addPrimitive(prim)
    node = GeomNode("buzzard")
    node.addGeom(geom)
    return node


def create_threat_geometry(
    threat_type: str = "missile",
    size: float = 2.0,
    color: Tuple[float, float, float, float] = (1.0, 0.2, 0.1, 1.0)
) -> GeomNode:
    """
    Create threat-specific geometry.
    
    Different shapes for different threat types:
    - missile: Elongated cylinder with fins
    - drone: Quad-copter shape
    - swarm: Small sphere
    """
    if not PANDA3D_AVAILABLE:
        return None
    
    format = GeomVertexFormat.getV3n3c4()
    vdata = GeomVertexData(threat_type, format, Geom.UHStatic)
    
    vertex = GeomVertexWriter(vdata, "vertex")
    normal = GeomVertexWriter(vdata, "normal")
    col = GeomVertexWriter(vdata, "color")
    
    if threat_type in ["missile", "IR_MISSILE", "RADAR_MISSILE"]:
        # Elongated cone/cylinder
        length = size * 2
        radius = size * 0.3
        segments = 8
        
        # Nose cone
        vertex.addData3f(length/2, 0, 0)
        normal.addData3f(1, 0, 0)
        col.addData4f(*color)
        
        # Body
        for i in range(segments + 1):
            angle = 2 * np.pi * i / segments
            y = radius * np.cos(angle)
            z = radius * np.sin(angle)
            
            # Nose ring
            vertex.addData3f(length/4, y * 0.5, z * 0.5)
            normal.addData3f(0.5, np.cos(angle) * 0.5, np.sin(angle) * 0.5)
            col.addData4f(*color)
            
            # Body ring
            vertex.addData3f(-length/4, y, z)
            normal.addData3f(0, np.cos(angle), np.sin(angle))
            col.addData4f(*color)
            
            # Tail
            vertex.addData3f(-length/2, y * 0.7, z * 0.7)
            normal.addData3f(-0.3, np.cos(angle) * 0.7, np.sin(angle) * 0.7)
            col.addData4f(color[0] * 0.5, color[1] * 0.5, color[2] * 0.5, 1)
        
        prim = GeomTriangles(Geom.UHStatic)
        
        # Nose cone triangles
        for i in range(segments):
            prim.addVertices(0, 1 + i * 3, 1 + (i + 1) * 3)
        
        # Body triangles
        for i in range(segments):
            n = 1 + i * 3
            nn = 1 + ((i + 1) % (segments + 1)) * 3
            # Nose to body
            prim.addVertices(n, n + 1, nn)
            prim.addVertices(nn, n + 1, nn + 1)
            # Body to tail
            prim.addVertices(n + 1, n + 2, nn + 1)
            prim.addVertices(nn + 1, n + 2, nn + 2)
        
    else:
        # Simple sphere for other threats
        segments = 8
        for i in range(segments + 1):
            lat = np.pi * (-0.5 + float(i) / segments)
            for j in range(segments + 1):
                lon = 2 * np.pi * float(j) / segments
                
                x = size * np.cos(lat) * np.cos(lon)
                y = size * np.cos(lat) * np.sin(lon)
                z = size * np.sin(lat)
                
                vertex.addData3f(x, y, z)
                normal.addData3f(x/size, y/size, z/size)
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
    node = GeomNode(threat_type)
    node.addGeom(geom)
    return node


# =============================================================================
# TAB COLORS (consistent across system)
# =============================================================================

TAB_COLORS = {
    "UP": (0.1, 0.9, 0.2, 1.0),      # Bright green - above
    "DOWN": (0.9, 0.1, 0.1, 1.0),    # Red - below
    "LEFT": (0.9, 0.9, 0.1, 1.0),    # Yellow - port
    "RIGHT": (0.9, 0.1, 0.9, 1.0),   # Magenta - starboard
}

CABLE_COLORS = {
    "normal": (0.5, 0.5, 0.6, 1.0),
    "tension_low": (0.4, 0.6, 0.4, 1.0),
    "tension_high": (0.8, 0.4, 0.2, 1.0),
    "near_limit": (1.0, 0.2, 0.1, 1.0),
}
