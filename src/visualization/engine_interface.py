"""
KAPS Visualization Engine Interface
=====================================

Pluggable renderer abstraction.
Swap between engines without touching physics or AI.

Supported engines:
- ModernGL (default) - Clean OpenGL 3.3+, proper shaders
- Panda3D (legacy) - If you really want cartoons
- Headless - No rendering, just state logging

Usage:
    from visualization.engine_interface import create_renderer
    
    renderer = create_renderer('moderngl')  # or 'panda3d', 'headless'
    renderer.set_simulation(sim)
    renderer.run()
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple, Any
import numpy as np
from dataclasses import dataclass


@dataclass
class RenderState:
    """Common state representation for all renderers."""
    
    # Buzzard (mother drone)
    buzzard_position: np.ndarray
    buzzard_velocity: np.ndarray
    buzzard_rotation: Optional[np.ndarray] = None
    
    # TABs
    tab_states: Dict[str, Dict] = None  # {id: {pos, vel, attached, color}}
    
    # Cables
    cables: list = None  # [(start, end, tension), ...]
    
    # Threats
    threats: list = None  # [{pos, vel, type, active}, ...]
    
    # Bola/slingshot mode
    bola_active: bool = False
    bola_position: Optional[np.ndarray] = None
    bola_rotation: float = 0.0
    
    # Camera
    camera_mode: str = "CHASE"
    camera_position: Optional[np.ndarray] = None
    camera_target: Optional[np.ndarray] = None
    
    # HUD
    hud_text: Dict[str, str] = None


class RendererInterface(ABC):
    """Abstract base for all KAPS renderers."""
    
    @abstractmethod
    def initialize(self):
        """Initialize the rendering engine."""
        pass
    
    @abstractmethod
    def set_simulation(self, sim, env=None):
        """Connect to KAPS simulation."""
        pass
    
    @abstractmethod
    def update_state(self, state: RenderState):
        """Push new state to renderer."""
        pass
    
    @abstractmethod
    def render_frame(self, dt: float):
        """Render one frame."""
        pass
    
    @abstractmethod
    def run(self):
        """Run the render loop (blocking)."""
        pass
    
    @abstractmethod
    def is_running(self) -> bool:
        """Check if renderer is still active."""
        pass
    
    @abstractmethod
    def shutdown(self):
        """Clean up resources."""
        pass
    
    # Camera controls
    @abstractmethod
    def set_camera_mode(self, mode: str):
        """Set camera mode: CHASE, ORBIT, FREE, TOP_DOWN, etc."""
        pass
    
    @abstractmethod
    def set_camera_position(self, position: np.ndarray, target: np.ndarray):
        """Manually set camera position and target."""
        pass


class HeadlessRenderer(RendererInterface):
    """No-op renderer for training without display."""
    
    def __init__(self):
        self._running = False
        self._frame_count = 0
        self.sim = None
        
    def initialize(self):
        self._running = True
        print("[Headless] Renderer initialized (no display)")
        
    def set_simulation(self, sim, env=None):
        self.sim = sim
        
    def update_state(self, state: RenderState):
        pass  # No-op
        
    def render_frame(self, dt: float):
        self._frame_count += 1
        
    def run(self):
        self._running = True
        print("[Headless] Running (no visual output)")
        
    def is_running(self) -> bool:
        return self._running
        
    def shutdown(self):
        self._running = False
        print(f"[Headless] Shutdown after {self._frame_count} frames")
        
    def set_camera_mode(self, mode: str):
        pass
        
    def set_camera_position(self, position: np.ndarray, target: np.ndarray):
        pass


class ModernGLRenderer(RendererInterface):
    """ModernGL-based renderer with proper shaders."""
    
    def __init__(self):
        self._window = None
        self._running = False
        self.sim = None
        self.env = None
        
    def initialize(self):
        # Lazy import to avoid requiring moderngl if not used
        from visualization.modern_renderer import KAPSModernRenderer
        import moderngl_window as mglw
        
        self._renderer_class = KAPSModernRenderer
        self._running = True
        print("[ModernGL] Renderer initialized")
        
    def set_simulation(self, sim, env=None):
        self.sim = sim
        self.env = env
        
    def update_state(self, state: RenderState):
        if self._window:
            # Push state to window
            pass
            
    def render_frame(self, dt: float):
        # Handled by moderngl_window event loop
        pass
        
    def run(self):
        import moderngl_window as mglw
        
        # Configure window with simulation
        class ConfiguredRenderer(self._renderer_class):
            pass
        
        ConfiguredRenderer.sim_instance = self.sim
        ConfiguredRenderer.env_instance = self.env
        
        # Inject simulation in __init__
        original_init = ConfiguredRenderer.__init__
        sim = self.sim
        env = self.env
        
        def new_init(self, **kwargs):
            original_init(self, **kwargs)
            self.set_simulation(sim, env)
        
        ConfiguredRenderer.__init__ = new_init
        
        mglw.run_window_config(ConfiguredRenderer)
        
    def is_running(self) -> bool:
        return self._running
        
    def shutdown(self):
        self._running = False
        
    def set_camera_mode(self, mode: str):
        pass
        
    def set_camera_position(self, position: np.ndarray, target: np.ndarray):
        pass


class Panda3DRenderer(RendererInterface):
    """Legacy Panda3D renderer (cartoonish but functional)."""
    
    def __init__(self):
        self._app = None
        self._running = False
        
    def initialize(self):
        print("[Panda3D] Use visual_trainer.py for Panda3D rendering")
        self._running = True
        
    def set_simulation(self, sim, env=None):
        pass
        
    def update_state(self, state: RenderState):
        pass
        
    def render_frame(self, dt: float):
        pass
        
    def run(self):
        # Delegate to existing trainer
        from training.visual_trainer import VisualTrainer
        trainer = VisualTrainer()
        trainer.run()
        
    def is_running(self) -> bool:
        return self._running
        
    def shutdown(self):
        self._running = False
        
    def set_camera_mode(self, mode: str):
        pass
        
    def set_camera_position(self, position: np.ndarray, target: np.ndarray):
        pass


# Registry of available engines
RENDERERS = {
    'moderngl': ModernGLRenderer,
    'opengl': ModernGLRenderer,  # Alias
    'panda3d': Panda3DRenderer,
    'panda': Panda3DRenderer,  # Alias
    'headless': HeadlessRenderer,
    'none': HeadlessRenderer,
}


def create_renderer(engine: str = 'moderngl') -> RendererInterface:
    """
    Create a renderer instance.
    
    Args:
        engine: One of 'moderngl', 'panda3d', 'headless'
        
    Returns:
        Initialized renderer
    """
    engine = engine.lower()
    
    if engine not in RENDERERS:
        available = ', '.join(RENDERERS.keys())
        raise ValueError(f"Unknown engine '{engine}'. Available: {available}")
    
    renderer = RENDERERS[engine]()
    renderer.initialize()
    
    return renderer


def list_available_engines() -> list:
    """List available rendering engines."""
    return list(set(RENDERERS.values()))


def get_recommended_engine() -> str:
    """Get the recommended engine for this system."""
    try:
        import moderngl
        return 'moderngl'
    except ImportError:
        pass
    
    try:
        from panda3d.core import PandaSystem
        return 'panda3d'
    except ImportError:
        pass
    
    return 'headless'
