"""
KAPS Cable Geometry System
===========================
Documentation for the cable intersection detection and sector constraint system.

PROBLEM STATEMENT
-----------------
The 4 TABs (Towed Aerodynamic Bodies) are connected to the Mother Drone via cables.
These cables are PHYSICAL OBJECTS that cannot pass through each other.

If the dreamer requests maneuvers that cross cables:
- Cables tangle
- Drag increases dramatically
- System can become unstable
- One TAB may need emergency release

SOLUTION: OPERATIONAL SECTORS
-----------------------------
Each TAB is assigned a WEDGE-SHAPED operational sector in the YZ plane
(perpendicular to flight direction).

The formation viewed from behind the Mother Drone:

                        UP
                       /  \
                      /    \
                     / 50°-130° \
                    /____________\
                   |      |      |
           LEFT    |      +      |    RIGHT
          140°-220°|   (Drone)   |-40° to 40°
                   |_____________|
                    \            /
                     \ -50° to  /
                      \ -130°  /
                       \______/
                        DOWN

Angles measured from +Y axis (right) in the YZ plane:
- UP:    50° to 130°   (upper quadrant)
- DOWN:  -130° to -50° (lower quadrant)  
- LEFT:  140° to -140° (wraps around 180°, left side)
- RIGHT: -40° to 40°   (right side)

The 10° gaps between sectors ensure cables NEVER intersect, even at sector boundaries.

INTERSECTION DETECTION
----------------------
If the agent somehow requests movement outside its sector, the action is CLAMPED.
The system also detects if cables do cross (due to physics perturbations) and
applies appropriate consequences:

1. CLEAR - Cables well separated (>2m)
2. PROXIMITY - Getting close (0.5-2m) - warning state
3. CROSSED - Cables have intersected (<0.5m)
4. TANGLED - Been crossed for multiple timesteps
5. LOCKED - Cannot separate without release

Consequences:
- PROXIMITY: Small drag increase
- CROSSED: Significant drag penalty
- TANGLED: Major drag, reduced maneuverability
- LOCKED: One TAB must be released (forced by system)

OBSERVATION SPACE ADDITIONS
---------------------------
The environment observation now includes cable geometry info:

Per TAB (2 values):
- sector_y: Y position in sector (normalized -1 to 1)
- sector_z: Z position in sector (normalized -1 to 1)

Per cable pair (2 values each, 6 pairs total):
- distance: Minimum cable separation (normalized)
- tangle_state: 0=clear, 0.25=proximity, 0.5=crossed, 0.75=tangled, 1=locked

Physics state (1 value):
- cable_drag: Current drag penalty from tangling

ACTION SPACE CONSTRAINTS
------------------------
The SectorConstrainedActionSpace clamps all TAB control actions to valid
regions BEFORE they're applied. This means:

- The agent CANNOT request impossible maneuvers
- Actions that would leave the sector are dampened/redirected
- The dreamer learns only physically valid strategies

This is NOT a penalty - the impossible actions are REMOVED from the space.
The dreamer learns the REAL dynamics of a cable-constrained system.

USAGE
-----
```python
from src.training.exploration_env import ExplorationKAPSEnv

# Environment automatically enables cable constraints
env = ExplorationKAPSEnv()

# Check that it's active
if env.cable_detector is not None:
    print("Cable geometry enforced!")
    print(f"Sectors: {env.cable_detector.sectors.keys()}")

# Run with any action - impossible actions are auto-clamped
obs, reward, done, trunc, info = env.step(action)

# Info includes cable state
print(info.get('forced_releases', 0))  # Any forced releases due to tangles
```

FILES
-----
- src/physics/cable_geometry.py - Core intersection detection
- src/training/exploration_env.py - Gym environment with constraints
- src/training/threat_environment.py - Threat system

VISUALIZATION
-------------
The cable sectors can be visualized using:
```python
from src.physics.cable_geometry import get_sector_boundaries_for_viz
viz_data = get_sector_boundaries_for_viz()
# Returns dict of tab_id -> list of boundary points
```
"""

# This file is documentation - it can be imported to print the docstring
if __name__ == "__main__":
    print(__doc__)
