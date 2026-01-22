"""
KAPS Architecture - AIRFOIL DEFENDERS
=====================================

THE ONLY GOAL: PROTECT THE BUZZARD

System Architecture:
                                        
         [TAB-UP]    ← AIRFOIL (1.5m wingspan, flies actively)
            |
            | cable (simple tether)
            |
    [TAB-L]─┼─[TAB-R]   ← Each TAB is an AIRFOIL, not a weight
            |             They generate lift and maneuver
            |
         [BUZZARD]      ← MOTHER DRONE (what we protect)
            |
            | cable
            |
         [TAB-DOWN]  ← AIRFOIL

COMPONENT BREAKDOWN:

1. CABLES (4 total)
   - Simple tethers connecting Buzzard to each TAB
   - ~30m length
   - Can store tension energy for release
   - Subject to intersection constraints (sectors)

2. TABs (4 total) - THE AIRFOIL DEFENDERS
   - Mass: 8kg
   - Wingspan: 1.5m  
   - Chord: 0.3m
   - Control surfaces: elevator, aileron, rudder
   - They FLY - generating lift and maneuvering
   - PURPOSE: Intercept threats, sacrifice for Buzzard

3. BUZZARD (1 total) - THE THING WE PROTECT
   - The mother drone
   - Carries payload, mission systems
   - MUST NOT BE HIT

OPERATIONAL SECTORS (prevents cable crossing):

    Viewed from behind:
    
           UP (50°-130°)
          ╱          ╲
         ╱            ╲
    LEFT ┼────────────┼ RIGHT
  (140°-220°)        (-40°-40°)
         ╲            ╱
          ╲          ╱
          DOWN (-130° to -50°)

REWARD PRIORITIES:

1. BUZZARD SURVIVAL (+1.0/step)      - Primary
2. BUZZARD DAMAGED (-500 per hit)    - Catastrophic failure  
3. THREAT INTERCEPTED (+200)         - Successful defense
4. TAB SACRIFICE (+50)               - Heroic death
5. TAB LOST WITHOUT KILL (-20)       - Wasteful

The reward structure makes it clear:
- TABs are EXPENDABLE
- The Buzzard is PRICELESS
- A TAB dying to kill a threat is GOOD
- A TAB dying for nothing is BAD
- The Buzzard taking ANY damage is TERRIBLE
"""

if __name__ == "__main__":
    print(__doc__)
