# MARIONETTE-001: Sail Drone Spool System

**Classification:** Tethered Multi-Sail Propulsion  
**Status:** IMPLEMENTED v1.0  
**Date:** 2026-01-22  
**Source:** `src/physics/marionette_spool.py`  

---

## 1. CONCEPT OVERVIEW

The MARIONETTE SPOOL is a **tethered sail constellation** where:
- Multiple sails on cables orbit a central spool
- The spool is **thrown** by human, then takes over autonomously
- A **Champion brain** (DreamerV3) controls sail pitch via cable lengths
- Uses **cascade-lattice** as computational substrate

```
    COLLAPSED (pre-throw)          DEPLOYED (marionette mode)
    
         ┌───────┐                      SAILS
         │███████│                    ⛵     ⛵
         │███████│◄── All cables        \   /
         │███████│    wound tight        \ /
         │SPOOL  │                    ┌───┴───┐
         └───────┘                    │ SPOOL │◄── Champion brain
              │                       │CONTROL│    runs HERE
              ▼                       └───┬───┘
           Human                          │
           hand                      individual
                                     cable control
```

---

## 2. OPERATIONAL PHASES

### Phase 1: WOUND
```
All cables coiled on spool drums.
Sails stacked/collapsed against spool body.
Human holds spool in hand.
```

### Phase 2: THROW
```
Human throws spool with spin.
Velocity: 15-25 m/s
Spin: 10-30 rad/s
```

### Phase 3: UNRAVELING
```
Centrifugal force pays out cables.
Sails deploy radially.
Spin rate increases (conservation of angular momentum).
```

### Phase 4: TENSIONED
```
Cables at max extension.
Sails spread in constellation.
System is aerodynamically active.
```

### Phase 5: MARIONETTE CONTROL
```
Champion brain reads tension from each drum.
Computes optimal cable length adjustments.
Controls sail pitch for:
  - Wind tacking (staying aloft)
  - Trajectory control
  - Formation maneuvers
```

---

## 3. SPOOL ARCHITECTURE

### 3.1 Physical Components

```
              ┌─────────────────────────────────────────┐
              │            SPOOL BODY                   │
              │  ┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐    │
              │  │DRUM │  │DRUM │  │DRUM │  │DRUM │    │
              │  │  0  │  │  1  │  │  2  │  │  N  │    │
              │  │ ◎◎◎ │  │ ◎◎◎ │  │ ◎◎◎ │  │ ◎◎◎ │    │
              │  └──┬──┘  └──┬──┘  └──┬──┘  └──┬──┘    │
              │     │        │        │        │       │
              │  ╔══╧════════╧════════╧════════╧══╗    │
              │  ║     CHAMPION BRAIN (RSSM)      ║    │
              │  ║  ┌───────────────────────────┐ ║    │
              │  ║  │ Tension → State → Actions │ ║    │
              │  ║  └───────────────────────────┘ ║    │
              │  ╚════════════════════════════════╝    │
              └─────────────────────────────────────────┘
                    │        │        │        │
                    ▼        ▼        ▼        ▼
                  CABLE    CABLE    CABLE    CABLE
                    │        │        │        │
                    ▼        ▼        ▼        ▼
                  SAIL     SAIL     SAIL     SAIL
                   ⛵       ⛵       ⛵       ⛵
```

### 3.2 Cable Drum Specs

| Property | Value | Notes |
|----------|-------|-------|
| Drum radius | 30mm | Per-sail drum |
| Cable length | 15m | Total wound |
| Max deployment | 12m | 80% of total |
| Cable diameter | 2mm | Dyneema/Spectra |
| Tension sensor | Strain gauge | Per-drum |
| Brake | Magnetic | Variable engagement |

### 3.3 Champion Brain

The spool contains a trained **DreamerV3 RSSM** that:

1. **Observes**: Tension from each cable drum
2. **Imagines**: Future states via world model
3. **Decides**: Cable length adjustments via actor
4. **Acts**: Motor commands to drums

```python
# Observation space (per drum)
tension: float      # N - current tension
tension_rate: float # N/s - rate of change
deployed_length: float  # m - cable extension

# Action space (per drum)  
length_adjust: float    # m - reel in (+) / pay out (-)
brake_engage: float     # 0-1 brake force
```

---

## 4. SAIL CONFIGURATION

### Default: 6-Sail Ring

```
            ⛵ Sail 0 (0°)
           / 
          /   
    ⛵ 5 /     \ ⛵ 1 (60°)
    (300°)     \
         \     /
          \   /
    ⛵ 4   ╳   ⛵ 2 (120°)
    (240°) │
           │
         ⛵ 3 (180°)
```

| Sail | Angle | Specialty |
|------|-------|-----------|
| 0 | 0° | Upwind tack |
| 1 | 60° | Beam reach |
| 2 | 120° | Broad reach |
| 3 | 180° | Downwind |
| 4 | 240° | Broad reach |
| 5 | 300° | Beam reach |

---

## 5. CONTROL MODES

### 5.1 Collective

All cables adjust together:
- **Reel in**: System descends, spin increases
- **Pay out**: System ascends, spin decreases

### 5.2 Cyclic

Sinusoidal variation around the ring:
- Creates **tacking** motion
- Allows **trajectory control**
- Amplitude + phase controlled by champion

### 5.3 External Command ("Yondu Whistle")

Human can provide high-level direction:
- Pitch/heading commands
- Override vector
- Recall signal

---

## 6. CASCADE-LATTICE INTEGRATION

The spool uses **cascade-lattice** for:

1. **HOLD**: Human can inspect/override any decision
2. **Merkle Provenance**: All actions are traced
3. **Distributed Compute**: Multi-sail coordination

```python
# From marionette_spool.py
cascade.register_component(self.spool_id, self)
```

---

## 7. PERFORMANCE ESTIMATES

| Parameter | Value | Notes |
|-----------|-------|-------|
| Launch velocity | 15-25 m/s | Human throw |
| Initial spin | 10-30 rad/s | Wrist flick |
| Deployment time | 2-3 s | Cables fully extend |
| Loiter time | Minutes | Wind dependent |
| Max range | 50-100m | Before recall |
| Control bandwidth | 10 Hz | Champion inference |

---

## 8. CROSS-REFERENCES

- [ARACHNE-001](ARACHNE-001_forearm_slingshot.md) - Forearm launcher (throw mechanism)
- `src/physics/hubless_dervish.py` - Sail physics
- `src/visualization/dervish_viz.py` - Visualization
- `src/ai/collective_intelligence.py` - Multi-agent coordination

---

## 9. INTEGRATION: ARACHNE + MARIONETTE

The **ARACHNE forearm launcher** can serve as the throw mechanism:

```
    FOREARM RUNWAY ─────────────────────────► SAIL CONSTELLATION
    
    [ELBOW]                                          ⛵
       ║                                           ⛵ ╲ ╱ ⛵
       ║══elastic══[WRIST]══SPOOL══════throw══════►  ╲│╱
       ║                                            [SPOOL]
    [THUMB]                                           │
       ↑                                          CHAMPION
    release                                        CONTROL
```

**Advantages:**
- Higher launch velocity (elastic assist)
- Consistent spin imparted
- Human hand free after launch
- Forearm becomes recall anchor

---

## 10. REVISION HISTORY

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-01-22 | Initial documentation from source |
