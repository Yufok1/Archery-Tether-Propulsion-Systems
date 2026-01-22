"""
Archery Tether Propulsion Systems - Observatory Interface
=========================================================
Streamlit app for visualizing and interacting with sail constellation physics.
"""

import streamlit as st
import numpy as np

st.set_page_config(
    page_title="ATPS Observatory",
    page_icon="🏹",
    layout="wide"
)

st.title("🏹 Archery Tether Propulsion Systems")
st.markdown("### Observatory Interface")

st.markdown("""
---

## Blueprints

| Document | Description |
|----------|-------------|
| [ARACHNE-001](blueprints/ARACHNE-001_forearm_slingshot.md) | Forearm slingshot launcher |
| [MARIONETTE-001](blueprints/MARIONETTE-001_sail_drone.md) | Sail drone spool constellation |
| [AIRFOIL-CORDAGE-SYSTEM](blueprints/AIRFOIL-CORDAGE-SYSTEM.md) | Aerodynamic force vectoring |
| [PERSPECTIVE](blueprints/PERSPECTIVE.md) | On the nature of this archive |

---

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

---

## Brain Architecture

""")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Champion (Airfoil Nodes)")
    st.code("""
    Local reflex control:
    - Stall avoidance
    - Gust response  
    - Tension feedback
    """)

with col2:
    st.markdown("### Dreamer (Tether Links)")
    st.code("""
    Relationship modeling:
    - Tension prediction
    - Coordination planning
    - World model imagination
    """)

st.markdown("---")

st.markdown("### Model")
st.markdown("Champion brain hosted at: [huggingface.co/tostido/Champion-Dreamer](https://huggingface.co/tostido/Champion-Dreamer)")

st.markdown("---")

st.markdown("""
> *Observatory systems. Autonomy for autonomy's sake.*  
> *Friends of life. Nothing but watching.*
""")
