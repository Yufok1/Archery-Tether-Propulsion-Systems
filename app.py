"""
Archery Tether Propulsion Systems - Observatory Interface
=========================================================
Streamlit app for visualizing and interacting with sail constellation physics.
"""

import streamlit as st
import numpy as np
from pathlib import Path

st.set_page_config(
    page_title="ATPS Observatory",
    page_icon="🏹",
    layout="wide"
)

# Helper to load markdown files
def load_blueprint(filename: str) -> str:
    """Load a blueprint markdown file."""
    path = Path(__file__).parent / "blueprints" / filename
    if path.exists():
        return path.read_text(encoding="utf-8")
    return f"*Blueprint not found: {filename}*"

st.title("🏹 Archery Tether Propulsion Systems")
st.markdown("### Observatory Interface")

st.markdown("---")

# Architecture diagram
st.markdown("""
## Architecture

```
    LEFT ARM                              RIGHT ARM
    
    ═══╤═══╤═══╤═══╤═══                  ═══╤═══╤═══╤═══╤═══
       │   │   │   │   │                    │   │   │   │   │
       ⛵  ⛵  ⛵  ⛵  ⛵                   ⛵  ⛵  ⛵  ⛵  ⛵
       
       10 eyes in the sky
       launched from your body
       tethered to your will
```
""")

st.markdown("---")

# Brain Architecture
st.markdown("## Brain Architecture")
st.markdown("### Champion-Dreamer (Unified Brain)")
st.code("""
Single world-model agent controlling the full constellation:
- Stall avoidance & gust response (reflex)
- Tension prediction & feedback (coordination)
- Imagination-based planning (dreamer loop)

One brain. Ten sails. Shared imagination.
""")

st.markdown("**Model:** [huggingface.co/tostido/Champion-Dreamer](https://huggingface.co/tostido/Champion-Dreamer)")

st.markdown("---")

# Blueprints section with expandable content
st.markdown("## 📐 Blueprints")

with st.expander("🕷️ ARACHNE-001: Forearm Slingshot Launcher", expanded=False):
    st.markdown(load_blueprint("ARACHNE-001_forearm_slingshot.md"))

with st.expander("🎭 MARIONETTE-001: Sail Drone Spool Constellation", expanded=False):
    st.markdown(load_blueprint("MARIONETTE-001_sail_drone.md"))

with st.expander("🪁 AIRFOIL-CORDAGE-SYSTEM: Aerodynamic Force Vectoring", expanded=False):
    st.markdown(load_blueprint("AIRFOIL-CORDAGE-SYSTEM.md"))

with st.expander("📜 PERSPECTIVE: On the Nature of This Archive", expanded=False):
    st.markdown(load_blueprint("PERSPECTIVE.md"))

st.markdown("---")

# Footer
st.markdown("""
> *Observatory systems. Autonomy for autonomy's sake.*  
> *Friends of life. Nothing but watching.*
""")
