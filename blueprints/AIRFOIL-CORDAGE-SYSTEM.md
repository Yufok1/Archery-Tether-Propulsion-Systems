# Airfoil-Cordage Force Direction System

**Classification:** Aerodynamic Force Vectoring via Tether Control  
**Status:** REFERENCE DOCUMENT  
**Date:** 2026-01-22  

---

## 1. CORE PRINCIPLE

**Cordage (tethers/cables) direct wind forces by controlling airfoil orientation.**

The tether is NOT just a passive connector—it's an **active control linkage** that:
1. Transmits force from airfoil to anchor
2. Controls airfoil angle of attack via tension
3. Enables collective/cyclic pitch across multiple sails

```
         WIND ───────────────────────────────────────────────►
                                  
                           ╔═══════════════╗
                           ║   AIRFOIL     ║
                    ┌──────╫───────────────╫──────┐
                    │      ╚═══════════════╝      │
                    │             ▲               │
                    │     LIFT    │               │
                    │             │               │
                    │      ───────┼───────        │
                    │             │ α (pitch)     │
                    │             │               │
                    │             └───────        │
                    └────────────────┬────────────┘
                                     │
                              TETHER │ (tension)
                                     │
                                     ▼
                             ┌───────────────┐
                             │    ANCHOR     │
                             │  (human/spool)│
                             └───────────────┘
```

---

## 2. AIRFOIL FORCE GENERATION

### 2.1 Lift Equation

```
L = ½ ρ V² S Cₗ

Where:
  L  = Lift force (N)
  ρ  = Air density (1.225 kg/m³ at sea level)
  V  = Airspeed (m/s)
  S  = Wing area (m²)
  Cₗ = Lift coefficient (function of α)
```

### 2.2 Lift Coefficient vs Angle of Attack

```
    Cₗ
    ↑
1.4 │                  ●──────● STALL
    │              ●
1.0 │          ●
    │      ●
0.5 │  ●
    │
  0 ├──────────────────────────────► α (angle of attack)
    0°    5°    10°   15°   20°
              ▲
              │
        LINEAR REGION
        Cₗ = 2π sin(α) ≈ 2πα
```

**Source:** `src/physics/aerodynamics.py` → `AerodynamicsEngine.compute_lift_coefficient()`

### 2.3 Drag Components

```
    TOTAL DRAG = PARASITIC + INDUCED + STALL
    
    Cd = Cd₀ + Cₗ²/(π·e·AR) + Cd_stall

         ↑       ↑              ↑
    skin    drag due      separation
    friction   to lift       drag
    + pressure             (post-stall)
```

**Source:** `src/physics/aerodynamics.py` → `AerodynamicsEngine.compute_drag_coefficient()`

---

## 3. CORDAGE FORCE TRANSMISSION

### 3.1 Tether as Force Vector

The tether transmits force along its length. By controlling **anchor position** and **tension**, you control the airfoil's resultant force vector.

```
                     AIRFOIL
                       ⛵
                      /│\
                     / │ \
           LIFT ────►  │  ◄──── SIDE FORCE
                       │
                       │ DRAG
                       ▼
                       │
                       │ TETHER
                       │ (transmits NET force)
                       │
                       │
                       ▼
                    ANCHOR
                    
                    
    ════════════════════════════════════════════
    
    FORCE RESOLUTION:
    
         ↑ LIFT (L)
         │
         │    ┌──► NET TETHER FORCE
         │   /     (what anchor feels)
         │  /
         │θ/
         │/        θ = atan(D/L)
         ●───────►
           DRAG (D)
```

### 3.2 Multi-Tether Airfoil Control

A single airfoil with **multiple tethers** gains attitude control:

```
    ┌─────────────────────────────────────┐
    │           AIRFOIL                   │
    │  ╔═════════════════════════════╗    │
    │  ║         WING SURFACE        ║    │
    │  ╚═════════════════════════════╝    │
    │     ▲              ▲              ▲  │
    │     │              │              │  │
    │  TETHER A      TETHER B      TETHER C│
    │     │              │              │  │
    └─────┼──────────────┼──────────────┼──┘
          │              │              │
          │              │              │
          ▼              ▼              ▼
       DRUM A         DRUM B         DRUM C
       └──────────────────────────────────┘
                   SPOOL BODY


    PITCH CONTROL:
    ─────────────────
    Shorten A, lengthen C → Nose UP
    Shorten C, lengthen A → Nose DOWN
    
    ROLL CONTROL:
    ─────────────────
    Differential tension → Bank left/right
```

---

## 4. CONSTELLATION GEOMETRY

### 4.1 Hubless Dervish Topology

Multiple airfoils connected by tethers with **no central body**.

```
              ⛵ NODE 0
             ╱│╲
            ╱ │ ╲
           ╱  │  ╲    ← TETHER LINKS
          ╱   │   ╲
         ╱    │    ╲
    ⛵ 5      │      ⛵ 1
      ╲      │      ╱
       ╲     ╳     ╱   ← VIRTUAL CENTROID
        ╲    │    ╱      (no physical mass)
         ╲   │   ╱
          ╲  │  ╱
           ╲ │ ╱
            ╲│╱
             ⛵ 3
            ╱ ╲
           ╱   ╲
          ╱     ╲
     ⛵ 4       ⛵ 2


    The "vehicle" IS the network.
    Center of mass is emergent.
```

**Source:** `src/physics/hubless_dervish.py` → Lines 1-20

### 4.2 Force Balance in Constellation

Each node experiences:
1. **Aero forces** (lift/drag from its airfoil)
2. **Tether forces** (tension from connected nodes)
3. **Gravity**

```
         LIFT ↑
              │
              │    TETHER
              │   TENSION ↗
    DRAG ◄────●─────────►
              │         ↘
              │          TETHER
              ▼           TENSION
           GRAVITY


    EQUILIBRIUM CONDITION:
    ───────────────────────
    Σ F_aero + Σ F_tether + F_gravity = 0
```

**Source:** `src/physics/hubless_dervish.py` → `HublessDervish.step()`

---

## 5. CYCLIC PITCH CONTROL

### 5.1 Collective vs Cyclic

```
    COLLECTIVE PITCH                CYCLIC PITCH
    ────────────────               ──────────────
    All airfoils same α            α varies with position
    
         ⛵ α=10°                      ⛵ α=15°
        ╱    ╲                       ╱    ╲
       ╱      ╲                     ╱      ╲
    ⛵         ⛵                 ⛵         ⛵
    α=10°     α=10°              α=5°      α=10°
       ╲      ╱                     ╲      ╱
        ╲    ╱                       ╲    ╱
         ⛵ α=10°                      ⛵ α=5°
    
    Effect: Uniform lift          Effect: NET THRUST
    (hover/climb)                 (directional movement)
```

### 5.2 Cyclic Pitch Formula

```
    α(θ) = α_collective + α_cyclic · sin(θ - φ_cyclic)

    Where:
      θ = Node angular position in constellation
      φ_cyclic = Phase angle (determines thrust direction)
      α_cyclic = Amplitude (determines thrust magnitude)
```

**Source:** `src/physics/hubless_dervish.py` → `HublessDervish.get_node_pitch()`

```python
def get_node_pitch(self, node: AirfoilNode) -> float:
    # Node's angular position relative to centroid
    r = node.position - self.centroid
    angular_pos = np.arctan2(r[1], r[0]) + self.spin_phase
    
    # Cyclic variation
    cyclic = self.cyclic_amplitude * np.sin(angular_pos - self.cyclic_phase)
    
    return self.collective_pitch + cyclic
```

---

## 6. CORDAGE TENSION DYNAMICS

### 6.1 Spring-Damper Model

```
    NODE A ───────[SPRING]───────[DAMPER]─────── NODE B
           ←────── stretch ──────→
           ←────── velocity ─────→

    TENSION = k · Δx + c · Δv

    Where:
      k = Stiffness (50,000 N/m typical)
      Δx = Stretch beyond rest length
      c = Damping coefficient (500 Ns/m)
      Δv = Relative velocity along cable
```

**Source:** `src/physics/hubless_dervish.py` → `TetherLink` class, `compute_tether_forces()`

### 6.2 Tension-Only Constraint

```
    CABLES CAN PULL, NOT PUSH
    
    IF stretch < 0:
        tension = 0  (slack cable)
    ELSE:
        tension = k * stretch + damping
    
    ┌──────────────────────────────────────────────┐
    │  TAUT            │  SLACK                    │
    │                  │                           │
    │  ─────●─────●    │    ●                      │
    │  (tension > 0)   │     ╲                     │
    │                  │      ╲ (catenary sag)     │
    │                  │       ●                   │
    └──────────────────────────────────────────────┘
```

---

## 7. AIRFOIL NODE SPECIFICATION

### 7.1 Physical Properties

| Property | Symbol | Value | Units | Source |
|----------|--------|-------|-------|--------|
| Mass | m | 2.0 | kg | `AirfoilNode.mass` |
| Wing area | S | 0.5 | m² | `AirfoilNode.wing_area` |
| Aspect ratio | AR | 6.0 | - | `AirfoilNode.aspect_ratio` |
| Pitch authority | - | ±15° | rad | `HublessDervish` |
| Roll authority | - | ±30° | rad | - |

### 7.2 Aerodynamic Surface Properties

| Property | Symbol | Value | Units | Source |
|----------|--------|-------|-------|--------|
| Span | b | 1.5 | m | `TABAerodynamics.tab_wing` |
| Chord | c | 0.3 | m | `TABAerodynamics.tab_wing` |
| Lift slope | Cₗ_α | 5.7 | /rad | `TABAerodynamics.tab_wing` |
| Max Cₗ | Cₗ_max | 1.4 | - | `TABAerodynamics.tab_wing` |
| Zero-lift drag | Cd₀ | 0.04 | - | `TABAerodynamics.tab_wing` |
| Stall angle | α_stall | 15° | deg | `AerodynamicsEngine` |

**Source:** `src/physics/aerodynamics.py` → `AeroSurface`, `TABAerodynamics`

---

## 8. FLIGHT REGIME CLASSIFICATION

```
    ┌────────────────────────────────────────────────────────────┐
    │                                                            │
    │   ATTACHED     STALL       DEEP        POST                │
    │   FLOW         ONSET       STALL       STALL               │
    │                                                            │
    │   │←────────→│←────→│←──────────→│←──────────────→│       │
    │   0°        15°   18°          45°              90°        │
    │                                                            │
    │   Cₗ linear   Cₗ drops   Cₗ ≈ 2sin(α)cos(α)               │
    │   Cd low      Cd rises   Cd massive                        │
    │                                                            │
    └────────────────────────────────────────────────────────────┘
```

**Source:** `src/physics/aerodynamics.py` → `FlightRegime` enum

---

## 9. CROSS-SYSTEM INTEGRATION

### 9.1 MARIONETTE Sail Control

The MARIONETTE spool uses **drum-based cordage** to control sail pitch:

```
    SPOOL DRUM ─────cable────→ SAIL AIRFOIL
    
    REEL IN  = Increase α = More lift
    PAY OUT  = Decrease α = Less lift
    
    Champion brain reads tension → decides cable lengths
```

**Cross-reference:** [MARIONETTE-001](MARIONETTE-001_sail_drone.md)

### 9.2 ARACHNE Launch Integration

The ARACHNE launcher provides **initial velocity** to deploy the constellation:

```
    FOREARM ──elastic──→ SPOOL ──throw──→ CONSTELLATION
                                              │
                                              ▼
                                         CENTRIFUGAL
                                         DEPLOYMENT
                                              │
                                              ▼
                                         CYCLIC PITCH
                                         CONTROL
```

**Cross-reference:** [ARACHNE-001](ARACHNE-001_forearm_slingshot.md)

---

## 10. OPERATIONAL SECTORS

Each TAB operates in a designated wedge-shaped sector:

```
                    UP (+Z)
                     │
                     │ 90°
          135° ╲     │     ╱ 45°
                ╲    │    ╱
                 ╲   │   ╱
    LEFT          ╲  │  ╱          RIGHT
    (-Y) ──────────╲─┼─╱────────── (+Y)
    180°           ╲│╱             0°
                   ╱│╲
                  ╱ │ ╲
                 ╱  │  ╲
          -135°╱    │    ╲-45°
                    │
                    │ -90°
                  DOWN (-Z)
                  
                  
    SECTOR BOUNDARIES:
    ─────────────────
    UP:    45° to 135°
    DOWN: -45° to -135°
    LEFT: 135° to 225° (or -135° to -180°, 180° to 135°)
    RIGHT: -45° to 45°
```

**Source:** `src/physics/cable_geometry.py` → Lines 1-45, `OperationalSector`

---

## 11. SOURCE FILE INDEX

| File | Purpose | Key Classes/Functions |
|------|---------|----------------------|
| `src/physics/aerodynamics.py` | Lift/drag/side-force | `AerodynamicsEngine`, `AeroSurface` |
| `src/physics/hubless_dervish.py` | Constellation physics | `HublessDervish`, `AirfoilNode`, `TetherLink` |
| `src/physics/cable_geometry.py` | Cable intersection | `CableGeometry`, `OperationalSector` |
| `src/physics/marionette_spool.py` | Sail spool control | `MarionetteSpool`, `CableDrum` |
| `src/physics/tether_dynamics.py` | Tether physics | (referenced) |
| `src/physics/slingshot_dynamics.py` | Launcher physics | (referenced) |

---

## 12. KEY EQUATIONS SUMMARY

### Aerodynamics
```
Lift:     L = ½ρV²SCₗ
Drag:     D = ½ρV²SCd
Cₗ:       Cₗ = 2π·sin(α) · η    (thin airfoil, η = efficiency)
Cd:       Cd = Cd₀ + Cₗ²/(π·e·AR)
```

### Tether Mechanics
```
Tension:  T = k·Δx + c·Δv      (spring-damper)
Force:    F⃗ = T·n̂             (along cable direction)
```

### Constellation Control
```
Pitch:    α(θ) = α_coll + α_cyc·sin(θ - φ)
Thrust:   F⃗_thrust = Σ L_asymmetric
Centroid: r⃗_c = Σ(m_i·r⃗_i) / Σm_i
```

---

*Document generated from source code analysis. All equations and parameters traced to implementation.*
