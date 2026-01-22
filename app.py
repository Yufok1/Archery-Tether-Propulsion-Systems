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

@st.cache_resource
def load_champion():
    """Pull Champion-Dreamer brain from HuggingFace Hub."""
    try:
        from huggingface_hub import hf_hub_download, HfApi
        
        # Get model info
        api = HfApi()
        model_id = "tostido/Champion-Dreamer"
        
        try:
            model_info = api.model_info(model_id)
            info = {
                "status": "available",
                "model_id": model_id,
                "last_modified": str(model_info.lastModified) if model_info.lastModified else "unknown",
                "downloads": getattr(model_info, 'downloads', 'N/A'),
                "tags": getattr(model_info, 'tags', []),
            }
            
            # Try to download the champion capsule
            try:
                champion_path = hf_hub_download(
                    repo_id=model_id,
                    filename="champion_gen42.py",
                    cache_dir=".cache"
                )
                info["champion_path"] = champion_path
                info["champion_loaded"] = True
                
                # Try to extract metadata from the file
                with open(champion_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read(5000)  # Read first 5KB for metadata
                    if '_GENERATION' in content:
                        import re
                        gen_match = re.search(r'_GENERATION\s*=\s*(\d+)', content)
                        if gen_match:
                            info["generation"] = int(gen_match.group(1))
                    if '_FITNESS' in content:
                        fit_match = re.search(r'_FITNESS\s*=\s*([\d.]+)', content)
                        if fit_match:
                            info["fitness"] = float(fit_match.group(1))
            except Exception as e:
                info["champion_loaded"] = False
                info["champion_error"] = str(e)
                
            return info
        except Exception as e:
            return {"status": "not_found", "error": str(e)}
    except ImportError:
        return {"status": "hub_unavailable", "error": "huggingface_hub not installed"}

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
st.markdown("## 🧠 Brain Architecture")
st.markdown("### Champion-Dreamer (Unified Brain)")
st.code("""
Single world-model agent controlling the full constellation:
- Stall avoidance & gust response (reflex)
- Tension prediction & feedback (coordination)
- Imagination-based planning (dreamer loop)

One brain. Ten sails. Shared imagination.
""")

# Pull and display Champion model info
with st.spinner("Pulling Champion brain from HuggingFace..."):
    champion_info = load_champion()

if champion_info["status"] == "available":
    col1, col2 = st.columns(2)
    with col1:
        st.success("✅ Champion-Dreamer connected")
        st.markdown(f"**Model:** [{champion_info['model_id']}](https://huggingface.co/{champion_info['model_id']})")
        if champion_info.get("champion_loaded"):
            st.markdown(f"**Capsule:** `champion_gen42.py`")
    with col2:
        if "generation" in champion_info:
            st.metric("Generation", champion_info["generation"])
        if "fitness" in champion_info:
            st.metric("Fitness", f"{champion_info['fitness']:.4f}")
        if champion_info.get("downloads"):
            st.caption(f"Downloads: {champion_info['downloads']}")
elif champion_info["status"] == "not_found":
    st.warning(f"⚠️ Champion model not found on HuggingFace Hub")
    st.markdown("**Expected:** [huggingface.co/tostido/Champion-Dreamer](https://huggingface.co/tostido/Champion-Dreamer)")
else:
    st.info("ℹ️ HuggingFace Hub unavailable - using local fallback")
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
