# ARACHNE-001: Forearm Slingshot Launcher

**Classification:** Wearable Elastic Propulsion  
**Status:** DRAFT v0.1  
**Date:** 2026-01-22  

---

## 1. CONCEPT OVERVIEW

```
                    FOREARM SLINGSHOT - "ARACHNE LAUNCHER"
                    ═══════════════════════════════════════
    
    ELBOW VERTEX (Anchor Point)
    ┌───────────────────────────────────────────────────────────────────┐
    │                                                                   │
    │   ╔═══╗                                                          │
    │   ║ ◉ ║ ← Elbow mount (elastic anchor + tensioner)               │
    │   ╚═╤═╝                                                          │
    │     │                                                            │
    │     │  ══════════════════════════════════════                    │
    │     │  ║ ELASTIC BAND (high-tension latex)  ║                    │
    │     │  ══════════════════════════════════════                    │
    │     │                    │                                       │
    │     │         F O R E A R M   R U N W A Y                        │
    │     │                    │                                       │
    │     ▼                    ▼                                       │
    │   ┌─────────────────────────────────────────┐                    │
    │   │▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓│ ← Forearm sleeve   │
    │   │▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓│   (guides elastic) │
    │   └─────────────────────────────────────────┘                    │
    │                         │                                        │
    │                         ▼                                        │
    │              ╔═══════════════════╗                               │
    │              ║  WRIST COIL UNIT  ║ ← Spring-loaded spool         │
    │              ║  ┌───┐  ┌───┐     ║   (stores elastic tension)    │
    │              ║  │ ◎ │──│ ◎ │     ║   (uncoils at MACH release)   │
    │              ║  └───┘  └───┘     ║                               │
    │              ╚════════╤══════════╝                               │
    │                       │                                          │
    │                       ▼                                          │
    │   ┌───────────────────────────────────────────────┐              │
    │   │             BACK OF HAND                      │              │
    │   │         ╔═══════════════════╗                 │              │
    │   │         ║   SPRINGBOARD     ║ ← Launch ramp   │              │
    │   │         ║   ▲▲▲▲▲▲▲▲▲▲▲▲▲   ║   (curved for   │              │
    │   │         ║  /             \  ║    trajectory)  │              │
    │   │         ╚═══════════════════╝                 │              │
    │   │                 │                             │              │
    │   └─────────────────┼─────────────────────────────┘              │
    │                     │                                            │
    │   ┌─────────────────┼─────────────────────────────────┐          │
    │   │    T H U M B    │    T R I G G E R               │          │
    │   │                 ▼                                 │          │
    │   │        ┌─────────────┐                           │          │
    │   │        │ ◉ LATCH ◉   │ ← Thumb holds elastic     │          │
    │   │        │  ╲     ╱    │   at tension vertex       │          │
    │   │        │   ╲   ╱     │                           │          │
    │   │        │    ╲ ╱      │   RELEASE = thumb lifts   │          │
    │   │        │     V       │   → elastic snaps back    │          │
    │   │        └─────────────┘                           │          │
    │   └───────────────────────────────────────────────────┘          │
    │                                                                   │
    └───────────────────────────────────────────────────────────────────┘
```

---

## 2. LAUNCH SEQUENCE

### Phase 1: ARMED
```
    [ELBOW]
       ║
       ║←─────────────────────┐
       ║  elastic stretched   │
       ║  (potential energy)  │
       ║                      │
    [WRIST]                   │
       ║  coiled tight        │
       ║                      │
    [HAND]                    │
       ║                      │
    [THUMB]═══════════════════┘
       ↑                       
    LATCHED (vertex)          
```

### Phase 2: RELEASE
```
    [ELBOW]
       ║
       ║ ←──── elastic SNAPS
       ║       (kinetic energy)
       ║
    [WRIST]
       ║ UNCOILS → MACH WAVE
       ║
    [HAND]
       ║
    [THUMB] ← released
```

### Phase 3: TUNNEL ACCELERATION
```
    Elastic wave propagates through wrist coil at supersonic speed:

        ════►════►════►
           MACH WAVE
```

### Phase 4: SPRINGBOARD LAUNCH
```
    Projectile hits curved ramp on back of hand, redirects trajectory:

           ╱ ↗ PROJECTILE
          ╱
    ═════╱ springboard
         ↖ incoming wave
```

---

## 3. COMPONENT SPECIFICATIONS

### 3.1 Elbow Anchor Assembly

| Component | Material | Dimensions | Notes |
|-----------|----------|------------|-------|
| Mount plate | Aluminum 6061-T6 | 60mm × 40mm × 3mm | Anodized |
| Elastic hook | Stainless 316 | 8mm diameter | Polished loop |
| Strap | Neoprene + Velcro | Adjustable | Silicone grip lining |
| Tensioner | Spring steel | 2mm wire | Preload adjustment |

**Load Rating:** 150N static, 300N dynamic

### 3.2 Forearm Runway Sleeve

| Component | Material | Dimensions | Notes |
|-----------|----------|------------|-------|
| Outer shell | Ripstop nylon | 250mm length | Low friction |
| Inner channel | PTFE tube | 12mm ID | Guides elastic |
| Reinforcement | Carbon fiber strips | 3mm × 200mm | Prevents twist |

### 3.3 Wrist Coil Unit

```
          ┌──────────────────────┐
          │   SPRING HOUSING     │
          │  ┌────────────────┐  │
          │  │ ╭──────────╮   │  │
          │  │ │ ◎◎◎◎◎◎◎◎ │   │  │ ← Coiled elastic
          │  │ │ ◎◎◎◎◎◎◎◎ │   │  │   (compressed spiral)
          │  │ ╰──────────╯   │  │
          │  └───────┬────────┘  │
          │          │           │
          │      ════╧════       │ ← Release channel
          │     (MACH TUNNEL)    │
          └──────────────────────┘
```

| Component | Material | Dimensions | Notes |
|-----------|----------|------------|-------|
| Housing | Glass-filled nylon | 50mm × 30mm × 20mm | Injection molded |
| Spool core | Delrin | 25mm diameter | Low friction |
| Spring | Music wire | 0.8mm × 15 coils | 2N/mm rate |
| Channel | Polished steel tube | 8mm ID | Chrome lined |

### 3.4 Springboard (Back of Hand)

| Component | Material | Dimensions | Notes |
|-----------|----------|------------|-------|
| Ramp surface | Flex-polymer (TPU) | 80mm × 50mm | 15° curve |
| Base plate | Carbon fiber | 2mm thickness | Bonded to glove |
| Shock absorber | Silicone gel pad | 5mm thickness | Dampens impact |

### 3.5 Thumb Trigger

| Component | Material | Dimensions | Notes |
|-----------|----------|------------|-------|
| Latch body | Titanium | 20mm × 15mm | Lightweight |
| Release mechanism | Ball detent | 3mm ball | 10N release force |
| Thumb pad | Textured rubber | 15mm diameter | Ergonomic |

---

## 4. ELASTIC SPECIFICATIONS

### Primary Band
- **Material:** Natural latex surgical tubing
- **Dimensions:** 6mm OD × 3mm ID × 400mm length
- **Elongation:** 600% max
- **Tensile strength:** 25 MPa
- **Fatigue life:** ~5000 cycles at 300% stretch

### Reinforcement (at vertices)
- **Material:** Kevlar sleeve
- **Coverage:** 30mm at each anchor point
- **Purpose:** Prevent abrasion failure

---

## 5. PERFORMANCE ESTIMATES

| Parameter | Value | Notes |
|-----------|-------|-------|
| Draw length | 300mm | Elbow to thumb |
| Draw force | 50-80N | User adjustable |
| Stored energy | 7-12 J | At full draw |
| Projectile mass | 5-20g | Foam darts, etc. |
| Launch velocity | 25-40 m/s | 90-145 km/h |
| Effective range | 15-30m | Flat trajectory |

---

## 6. ERGONOMICS & FIT

### Sizing Chart

| Size | Forearm Length | Wrist Circumference |
|------|----------------|---------------------|
| S | 220-250mm | 140-160mm |
| M | 250-280mm | 160-180mm |
| L | 280-310mm | 180-200mm |
| XL | 310-340mm | 200-220mm |

### Fit Concerns & Solutions

| Concern | Risk | Solution |
|---------|------|----------|
| Elbow anchor slip | Misfire | Velcro + silicone grip |
| Wrist coil bulk | Discomfort | Low-profile spiral housing |
| Thumb fatigue | Strain injury | Mechanical latch (click-release) |
| Back-of-hand impact | Bruising | Flex-polymer springboard |
| Elastic wear at vertices | Failure | Kevlar reinforcement |

### Fit Rating

```
HAND-FIT RATING: ████████░░ 8/10
(Prototype viable, needs iteration on wrist bulk)
```

---

## 7. SAFETY WARNINGS

⚠️ **EYE PROTECTION REQUIRED** - Always wear safety glasses

⚠️ **ELASTIC INSPECTION** - Check for cracks/wear before each use

⚠️ **NEVER DRY FIRE** - Always have projectile loaded

⚠️ **TEMPERATURE LIMITS** - Latex degrades below 0°C and above 40°C

⚠️ **FINGER CLEARANCE** - Keep fingers clear of release channel

---

## 8. REVISION HISTORY

| Version | Date | Changes |
|---------|------|---------|
| 0.1 | 2026-01-22 | Initial draft |

---

## 9. RELATED DESIGNS

- ARACHNE-002: Dual-arm synchronized launcher (PLANNED)
- ARACHNE-003: Magnetic assist variant (PLANNED)
- ARACHNE-004: Compressed gas hybrid (PLANNED)

---

## 10. CROSS-REFERENCE: MARIONETTE SPOOL SYSTEM

The ARACHNE forearm launcher shares architectural DNA with the **MARIONETTE SPOOL** 
sail-drone propulsion system (`src/physics/marionette_spool.py`).

### Shared Principles

| Principle | ARACHNE | MARIONETTE |
|-----------|---------|------------|
| Tether Energy Storage | Elastic band on wrist coil | Cables wound on spool drums |
| Release Mechanism | Thumb latch (vertex) | Centrifugal unreeling |
| Runway/Guide | Forearm sleeve | Cable channels per sail |
| Human Control Point | Thumb trigger | Throw + whistle command |
| Terminal Guidance | Springboard redirect | Champion brain + HOLD |

### MARIONETTE SPOOL Architecture

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

### Integration Concept: ARACHNE-MARIONETTE Hybrid

The ARACHNE launcher could serve as the **throw mechanism** for a MARIONETTE spool:

1. **Pre-throw**: SPOOL is wound, cables coiled
2. **ARACHNE Launch**: Forearm runway provides velocity + spin to spool
3. **Unraveling**: Centrifugal force deploys sails
4. **Marionette Mode**: Champion brain takes over cable control
5. **Recall**: Sails collapse, cables reel back to spool

```
    FOREARM ──────────────────────────────────► DEPLOYED CONSTELLATION
    
    [ELBOW]                                          ⛵
       ║                                           ⛵ ╲ ╱ ⛵
       ║══elastic══[WRIST]══SPOOL══════throw══════►  ╲│╱
       ║                                            [SPOOL]
    [THUMB]                                           │
       ↑                                          MARIONETTE
    release                                        CONTROL
```

### Source References

- `src/physics/marionette_spool.py` - Spool mechanism + Champion brain integration
- `src/visualization/dervish_viz.py` - Visualization of sail constellation
- `src/physics/hubless_dervish.py` - Sail physics + wind tacking
