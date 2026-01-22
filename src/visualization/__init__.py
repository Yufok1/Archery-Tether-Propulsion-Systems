"""
Visualization Module
====================
3D rendering for the KAPS simulation using Panda3D.
"""

try:
    from .renderer import KAPSVisualizer, run_visualization, PANDA3D_AVAILABLE
except ImportError:
    PANDA3D_AVAILABLE = False
    KAPSVisualizer = None
    run_visualization = None

__all__ = ['KAPSVisualizer', 'run_visualization', 'PANDA3D_AVAILABLE']
