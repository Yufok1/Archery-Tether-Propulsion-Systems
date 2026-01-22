"""
Simple but VISIBLE 3D Geometry for KAPS
========================================

Replacing the complex geometry with SIMPLE, VISIBLE shapes:
- CABLES: Thick line strips (not tubes that may fail)
- AIRFOILS: Simple triangle meshes that are OBVIOUSLY wings
- BUZZARD: Elongated shape that's NOT a sphere

These MUST be visible. No complex vertex generation.
"""

import numpy as np
from typing import Tuple

try:
    from panda3d.core import (
        GeomVertexFormat, GeomVertexData, GeomVertexWriter,
        Geom, GeomTriangles, GeomNode, GeomLines, GeomLinestrips,
        Vec3, Vec4, Point3,
        LineSegs, CardMaker
    )
    PANDA3D_AVAILABLE = True
except ImportError:
    PANDA3D_AVAILABLE = False


def create_visible_buzzard(
    length: float = 12.0,
    width: float = 4.0,
    height: float = 3.0,
    color: Tuple[float, float, float, float] = (0.2, 0.4, 0.9, 1.0)
) -> GeomNode:
    """
    Create a VISIBLE Buzzard - elongated fuselage shape.
    
    This is a simple hexagonal prism with tapered nose/tail.
    NOT A SPHERE.
    """
    if not PANDA3D_AVAILABLE:
        return None
    
    format = GeomVertexFormat.getV3c4()
    vdata = GeomVertexData("buzzard", format, Geom.UHStatic)
    
    vertex = GeomVertexWriter(vdata, "vertex")
    col = GeomVertexWriter(vdata, "color")
    
    # Colors
    nose_color = (0.4, 0.5, 1.0, 1.0)  # Lighter nose
    body_color = color
    tail_color = (0.3, 0.3, 0.5, 1.0)  # Darker tail
    
    # Define vertices for an elongated hexagonal fuselage
    # Nose point
    vertex.addData3f(length/2, 0, 0)  # 0: nose tip
    col.addData4f(*nose_color)
    
    # Forward hexagon (at 1/4 from nose)
    fwd_x = length/4
    fwd_r = width/2 * 0.6
    for i in range(6):
        angle = i * np.pi / 3
        y = fwd_r * np.cos(angle)
        z = fwd_r * np.sin(angle) * (height/width)
        vertex.addData3f(fwd_x, y, z)  # 1-6
        col.addData4f(*nose_color)
    
    # Mid hexagon (widest)
    mid_x = 0
    mid_r = width/2
    for i in range(6):
        angle = i * np.pi / 3
        y = mid_r * np.cos(angle)
        z = mid_r * np.sin(angle) * (height/width)
        vertex.addData3f(mid_x, y, z)  # 7-12
        col.addData4f(*body_color)
    
    # Aft hexagon
    aft_x = -length/4
    aft_r = width/2 * 0.7
    for i in range(6):
        angle = i * np.pi / 3
        y = aft_r * np.cos(angle)
        z = aft_r * np.sin(angle) * (height/width)
        vertex.addData3f(aft_x, y, z)  # 13-18
        col.addData4f(*tail_color)
    
    # Tail point
    vertex.addData3f(-length/2, 0, 0)  # 19: tail tip
    col.addData4f(*tail_color)
    
    # Create triangles
    prim = GeomTriangles(Geom.UHStatic)
    
    # Nose cone (from tip to forward hex)
    for i in range(6):
        i1 = 1 + i
        i2 = 1 + (i + 1) % 6
        prim.addVertices(0, i1, i2)
    
    # Forward to mid section
    for i in range(6):
        f1 = 1 + i
        f2 = 1 + (i + 1) % 6
        m1 = 7 + i
        m2 = 7 + (i + 1) % 6
        prim.addVertices(f1, m1, f2)
        prim.addVertices(f2, m1, m2)
    
    # Mid to aft section
    for i in range(6):
        m1 = 7 + i
        m2 = 7 + (i + 1) % 6
        a1 = 13 + i
        a2 = 13 + (i + 1) % 6
        prim.addVertices(m1, a1, m2)
        prim.addVertices(m2, a1, a2)
    
    # Tail cone
    for i in range(6):
        i1 = 13 + i
        i2 = 13 + (i + 1) % 6
        prim.addVertices(19, i2, i1)
    
    geom = Geom(vdata)
    geom.addPrimitive(prim)
    node = GeomNode("buzzard")
    node.addGeom(geom)
    return node


def create_visible_airfoil(
    wingspan: float = 6.0,
    chord: float = 2.0,
    thickness: float = 0.5,
    color: Tuple[float, float, float, float] = (0.8, 0.8, 0.2, 1.0)
) -> GeomNode:
    """
    Create a VISIBLE delta wing airfoil.
    
    This is a simple swept wing that is OBVIOUSLY a wing, not a blob.
    """
    if not PANDA3D_AVAILABLE:
        return None
    
    format = GeomVertexFormat.getV3c4()
    vdata = GeomVertexData("airfoil", format, Geom.UHStatic)
    
    vertex = GeomVertexWriter(vdata, "vertex")
    col = GeomVertexWriter(vdata, "color")
    
    half_span = wingspan / 2
    tip_chord = chord * 0.3
    
    # Leading edge color (brighter)
    le_color = (
        min(1.0, color[0] + 0.3),
        min(1.0, color[1] + 0.3),
        min(1.0, color[2] + 0.3),
        color[3]
    )
    
    # Trailing edge color (darker)
    te_color = (
        color[0] * 0.7,
        color[1] * 0.7,
        color[2] * 0.7,
        color[3]
    )
    
    # Top surface vertices
    vertex.addData3f(chord, 0, thickness/2)           # 0: nose top
    col.addData4f(*le_color)
    
    vertex.addData3f(chord * 0.3, half_span, thickness/3)   # 1: right tip top
    col.addData4f(*color)
    
    vertex.addData3f(chord * 0.3, -half_span, thickness/3)  # 2: left tip top
    col.addData4f(*color)
    
    vertex.addData3f(-chord * 0.5, half_span * 0.3, thickness/4)  # 3: right TE top
    col.addData4f(*te_color)
    
    vertex.addData3f(-chord * 0.5, -half_span * 0.3, thickness/4) # 4: left TE top
    col.addData4f(*te_color)
    
    vertex.addData3f(-chord * 0.3, 0, thickness/4)    # 5: center TE top
    col.addData4f(*te_color)
    
    # Bottom surface vertices (same positions, below)
    vertex.addData3f(chord, 0, -thickness/4)          # 6: nose bottom
    col.addData4f(*le_color)
    
    vertex.addData3f(chord * 0.3, half_span, -thickness/4)  # 7: right tip bottom
    col.addData4f(*color)
    
    vertex.addData3f(chord * 0.3, -half_span, -thickness/4) # 8: left tip bottom
    col.addData4f(*color)
    
    vertex.addData3f(-chord * 0.5, half_span * 0.3, -thickness/4)  # 9: right TE bottom
    col.addData4f(*te_color)
    
    vertex.addData3f(-chord * 0.5, -half_span * 0.3, -thickness/4) # 10: left TE bottom
    col.addData4f(*te_color)
    
    vertex.addData3f(-chord * 0.3, 0, -thickness/4)   # 11: center TE bottom
    col.addData4f(*te_color)
    
    prim = GeomTriangles(Geom.UHStatic)
    
    # Top surface triangles
    prim.addVertices(0, 1, 3)    # Nose to right wing
    prim.addVertices(0, 3, 5)    # Nose to center TE
    prim.addVertices(0, 5, 4)    # Nose to left TE
    prim.addVertices(0, 4, 2)    # Nose to left tip
    prim.addVertices(1, 3, 5)    # Right side
    prim.addVertices(2, 5, 4)    # Left side
    
    # Bottom surface triangles (reversed winding)
    prim.addVertices(6, 9, 7)
    prim.addVertices(6, 11, 9)
    prim.addVertices(6, 10, 11)
    prim.addVertices(6, 8, 10)
    prim.addVertices(7, 11, 9)
    prim.addVertices(8, 10, 11)
    
    # Leading edge (connect top and bottom)
    prim.addVertices(0, 6, 1)
    prim.addVertices(1, 6, 7)
    prim.addVertices(0, 2, 6)
    prim.addVertices(2, 8, 6)
    
    # Trailing edge
    prim.addVertices(3, 9, 5)
    prim.addVertices(5, 9, 11)
    prim.addVertices(4, 5, 10)
    prim.addVertices(5, 11, 10)
    
    # Wing tips
    prim.addVertices(1, 7, 3)
    prim.addVertices(3, 7, 9)
    prim.addVertices(2, 4, 8)
    prim.addVertices(4, 10, 8)
    
    geom = Geom(vdata)
    geom.addPrimitive(prim)
    node = GeomNode("airfoil")
    node.addGeom(geom)
    return node


def create_visible_cable(
    start: np.ndarray,
    end: np.ndarray,
    color: Tuple[float, float, float, float] = (0.6, 0.5, 0.3, 1.0)
) -> GeomNode:
    """
    Create a VISIBLE cable using thick lines.
    
    Returns a GeomNode with thick lines, not a complex tube mesh.
    """
    if not PANDA3D_AVAILABLE:
        return None
    
    lines = LineSegs("cable")
    lines.setThickness(4.0)  # THICK lines
    lines.setColor(*color)
    lines.moveTo(Point3(float(start[0]), float(start[1]), float(start[2])))
    lines.drawTo(Point3(float(end[0]), float(end[1]), float(end[2])))
    
    return lines.create()


def create_visible_bola(
    radius: float = 3.0,
    color: Tuple[float, float, float, float] = (0.9, 0.6, 0.2, 1.0)
) -> GeomNode:
    """
    Create a visible bola (consolidated TAB mass).
    
    This is a chunky octahedron-like shape that screams "I AM A HEAVY MASS".
    """
    if not PANDA3D_AVAILABLE:
        return None
    
    format = GeomVertexFormat.getV3c4()
    vdata = GeomVertexData("bola", format, Geom.UHStatic)
    
    vertex = GeomVertexWriter(vdata, "vertex")
    col = GeomVertexWriter(vdata, "color")
    
    r = radius
    
    # Octahedron vertices (6 points)
    verts = [
        (r, 0, 0),    # 0: +X
        (-r, 0, 0),   # 1: -X
        (0, r, 0),    # 2: +Y
        (0, -r, 0),   # 3: -Y
        (0, 0, r),    # 4: +Z
        (0, 0, -r),   # 5: -Z
    ]
    
    colors = [
        (0.9, 0.5, 0.2, 1.0),  # Orange faces
        (0.8, 0.4, 0.1, 1.0),
        (0.95, 0.6, 0.3, 1.0),
        (0.85, 0.45, 0.15, 1.0),
        (1.0, 0.7, 0.4, 1.0),  # Brighter top
        (0.7, 0.35, 0.1, 1.0), # Darker bottom
    ]
    
    for v, c in zip(verts, colors):
        vertex.addData3f(*v)
        col.addData4f(*c)
    
    prim = GeomTriangles(Geom.UHStatic)
    
    # 8 triangular faces
    faces = [
        (0, 2, 4), (2, 1, 4), (1, 3, 4), (3, 0, 4),  # Top 4
        (0, 5, 2), (2, 5, 1), (1, 5, 3), (3, 5, 0),  # Bottom 4
    ]
    
    for f in faces:
        prim.addVertices(*f)
    
    geom = Geom(vdata)
    geom.addPrimitive(prim)
    node = GeomNode("bola")
    node.addGeom(geom)
    return node


def create_visible_grid_fin(
    size: float = 1.5,
    color: Tuple[float, float, float, float] = (0.5, 0.5, 0.6, 1.0)
) -> GeomNode:
    """
    Create a visible grid fin - flat panel with grid pattern.
    """
    if not PANDA3D_AVAILABLE:
        return None
    
    lines = LineSegs("gridfin")
    lines.setThickness(2.0)
    lines.setColor(*color)
    
    # Draw grid pattern
    s = size
    grid = 4
    
    for i in range(grid + 1):
        t = -s/2 + i * s / grid
        # Horizontal lines
        lines.moveTo(Point3(-s/2, 0, t))
        lines.drawTo(Point3(s/2, 0, t))
        # Vertical lines
        lines.moveTo(Point3(t, 0, -s/2))
        lines.drawTo(Point3(t, 0, s/2))
    
    return lines.create()


# Color palettes
VISIBLE_TAB_COLORS = {
    "UP": (0.2, 1.0, 0.2, 1.0),     # Bright green
    "DOWN": (1.0, 0.2, 0.2, 1.0),   # Bright red
    "LEFT": (1.0, 1.0, 0.2, 1.0),   # Bright yellow
    "RIGHT": (1.0, 0.2, 1.0, 1.0),  # Bright magenta
}

VISIBLE_CABLE_COLORS = {
    "slack": (0.3, 0.6, 0.3, 1.0),    # Green
    "taut": (0.8, 0.3, 0.2, 1.0),     # Red
    "normal": (0.5, 0.5, 0.4, 1.0),   # Tan
}


if __name__ == "__main__":
    print("Testing simple geometry creation...")
    
    buz = create_visible_buzzard()
    print(f"Buzzard: {buz}")
    
    air = create_visible_airfoil()
    print(f"Airfoil: {air}")
    
    bola = create_visible_bola()
    print(f"Bola: {bola}")
    
    print("All geometry created successfully!")
