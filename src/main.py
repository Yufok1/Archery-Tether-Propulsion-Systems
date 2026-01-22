"""
Archery Tether Propulsion Systems - Main Simulation
====================================================
Entry point for the KAPS (Kinetic Active Protection System) simulation.

This simulation demonstrates:
1. Cross-formation flight with 4 TABs
2. 360° defensive bubble mechanics
3. Slingshot intercept maneuvers
4. Speed burst escape mechanics
"""

import numpy as np
import yaml
import time
from pathlib import Path
from typing import Dict, Optional

# Physics engines
from .physics import (
    TetherArray, CableProperties, CableState,
    MomentumEngine, SlingshotManeuver, SlingshotParameters, MomentumState,
    ReleaseMode
)

# Entities
from .entities import (
    MotherDrone, MotherDroneConfig,
    TABArray, TowedAerodynamicBody, FormationPosition
)

# AI Controllers
from .ai import (
    DefensiveMatrixAI, DefenseConfig, ThreatType,
    FormationController, FormationMode
)


class KAPSSimulation:
    """
    Main simulation controller for the Kinetic Active Protection System.
    
    Orchestrates:
    - Physics integration
    - Entity updates
    - AI decision making
    - Visualization callbacks
    """
    
    def __init__(self, config_path: Optional[str] = None):
        # Load configuration
        if config_path:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = self._default_config()
        
        # Initialize physics engines
        self.momentum_engine = MomentumEngine()
        
        cable_props = CableProperties(
            length=self.config['cables']['length'],
            diameter=self.config['cables']['diameter'],
            breaking_strength=self.config['cables']['breaking_strength'],
            stiffness=self.config['cables']['stiffness'],
            damping=self.config['cables']['damping'],
            linear_density=self.config['cables']['linear_density'],
            max_tension=self.config['cables']['max_tension'],
            min_tension=self.config['cables']['min_tension']
        )
        self.tether_array = TetherArray(cable_props, num_cables=4)
        
        # Initialize entities
        drone_config = MotherDroneConfig(
            mass=self.config['mother_drone']['mass'],
            wingspan=self.config['mother_drone']['wingspan'],
            max_thrust=self.config['mother_drone']['max_thrust'],
            cruise_thrust_ratio=self.config['mother_drone']['cruise_thrust_ratio']
        )
        self.mother_drone = MotherDrone(drone_config)
        self.tab_array = TABArray(
            cable_length=cable_props.length,
            mother_position=self.mother_drone.position
        )
        
        # Initialize AI
        self.defensive_ai = DefensiveMatrixAI(DefenseConfig(
            detection_radius=self.config['defensive_ai']['detection_radius'],
            tracking_update_rate=self.config['defensive_ai']['tracking_update_rate']
        ))
        self.formation_ctrl = FormationController()
        
        # Slingshot maneuver handler
        slingshot_params = SlingshotParameters(
            spiral_rate=np.radians(self.config['slingshot']['spiral_rate']),
            wind_up_time=2.0,
            release_angle=np.pi/2,
            cable_length=cable_props.length,
            conservation_efficiency=self.config['slingshot']['conservation_efficiency']
        )
        self.slingshot = SlingshotManeuver(self.momentum_engine, slingshot_params)
        
        # Simulation state
        self.time = 0.0
        self.dt = self.config['simulation']['timestep']
        self.running = False
        
        # Telemetry history
        self.telemetry_log = []
        
    def _default_config(self) -> Dict:
        """Default configuration if no file provided"""
        return {
            'mother_drone': {
                'mass': 150.0,
                'wingspan': 8.0,
                'max_thrust': 2500.0,
                'cruise_thrust_ratio': 0.9
            },
            'cables': {
                'length': 30.0,
                'diameter': 0.004,
                'breaking_strength': 15000.0,
                'stiffness': 1.2e9,
                'damping': 0.15,
                'linear_density': 0.012,
                'max_tension': 12000.0,
                'min_tension': 50.0
            },
            'defensive_ai': {
                'detection_radius': 500.0,
                'tracking_update_rate': 100.0
            },
            'slingshot': {
                'spiral_rate': 45.0,
                'conservation_efficiency': 0.95
            },
            'simulation': {
                'timestep': 0.001,
                'duration': 60.0
            }
        }
    
    def step(self) -> Dict:
        """
        Execute one simulation timestep.
        
        Returns telemetry data for visualization.
        """
        # 1. Get current positions/velocities
        tab_positions = self.tab_array.get_positions()
        tab_velocities = self.tab_array.get_velocities()
        
        # 2. Compute cable forces
        tether_results = self.tether_array.compute_all_forces(
            self.mother_drone.position,
            self.mother_drone.velocity,
            tab_positions,
            tab_velocities,
            self.dt
        )
        
        # 3. Formation control
        formation_commands = self.formation_ctrl.compute_control_commands(
            self.mother_drone.position,
            self.mother_drone.velocity,
            tab_positions,
            tab_velocities,
            self.dt
        )
        
        # Apply formation commands to TABs
        for tab_id, cmd in formation_commands.items():
            if tab_id in self.tab_array.tabs:
                self.tab_array.tabs[tab_id].set_control_targets(
                    elevator=cmd['elevator'],
                    rudder=cmd['rudder']
                )
        
        # 4. Defensive AI response
        tab_states = {
            tab_id: tab.get_status_report()
            for tab_id, tab in self.tab_array.tabs.items()
        }
        defense_response = self.defensive_ai.calculate_defensive_response(
            self.mother_drone.position,
            self.mother_drone.velocity,
            tab_states
        )
        
        # 5. Execute any release commands
        for release_cmd in defense_response['release_commands']:
            tab_id = release_cmd['tab_id']
            if tab_id in self.tab_array.tabs:
                tab = self.tab_array.tabs[tab_id]
                
                # Calculate release velocity
                release_vel = self.momentum_engine.compute_release_velocity(
                    self.mother_drone.momentum_state,
                    tab.momentum_state,
                    tether_results.get(tab_id, {}).get('tension', 0),
                    ReleaseMode.SLINGSHOT if release_cmd['mode'] == 'slingshot' else ReleaseMode.INSTANT,
                    self.slingshot.params
                )
                
                # Execute release
                tab.execute_release(release_vel)
                self.tether_array.release_cable(tab_id)
                self.mother_drone.release_tab(tab_id)
        
        # 6. Update mother drone
        total_tether_force = tether_results.get('total_mother_force', np.zeros(3))
        mother_telemetry = self.mother_drone.update(self.dt, total_tether_force)
        
        # 7. Update all TABs
        cable_forces = {
            tab_id: tether_results.get(tab_id, {}).get('force_on_tab', np.zeros(3))
            for tab_id in self.tab_array.tabs
        }
        cable_states = {
            tab_id: tether_results.get(tab_id, {}).get('state', CableState.ATTACHED)
            for tab_id in self.tab_array.tabs
        }
        tab_telemetry = self.tab_array.update_all(self.dt, cable_forces, cable_states)
        
        # 8. Update time
        self.time += self.dt
        
        # 9. Compile telemetry
        telemetry = {
            'time': self.time,
            'mother_drone': mother_telemetry,
            'tabs': tab_telemetry,
            'cables': {
                tab_id: {
                    'tension': tether_results.get(tab_id, {}).get('tension', 0),
                    'length': tether_results.get(tab_id, {}).get('length', 0),
                    'state': str(tether_results.get(tab_id, {}).get('state', CableState.ATTACHED))
                }
                for tab_id in self.tab_array.tabs
            },
            'defense': self.defensive_ai.get_defensive_bubble_status(),
            'formation': self.formation_ctrl.get_formation_status(
                self.mother_drone.position,
                tab_positions
            )
        }
        
        return telemetry
    
    def run(self, duration: Optional[float] = None, callback=None):
        """
        Run simulation for specified duration.
        
        Args:
            duration: Simulation time in seconds (default from config)
            callback: Optional function called each step with telemetry
        """
        if duration is None:
            duration = self.config['simulation']['duration']
        
        self.running = True
        start_time = time.time()
        
        print(f"Starting KAPS Simulation - Duration: {duration}s")
        print("=" * 50)
        
        while self.time < duration and self.running:
            telemetry = self.step()
            
            if callback:
                callback(telemetry)
            
            # Progress update every second
            if int(self.time * 10) % 10 == 0 and int(self.time) != int(self.time - self.dt):
                self._print_status(telemetry)
        
        real_time = time.time() - start_time
        print("=" * 50)
        print(f"Simulation complete. Sim time: {self.time:.2f}s, Real time: {real_time:.2f}s")
        print(f"Speed ratio: {self.time/real_time:.1f}x realtime")
    
    def _print_status(self, telemetry: Dict):
        """Print compact status line"""
        md = telemetry['mother_drone']
        defense = telemetry['defense']
        
        print(f"T={self.time:6.1f}s | "
              f"Alt: {md['altitude']:6.1f}m | "
              f"Speed: {md['speed']:5.1f}m/s | "
              f"TABs: {self.tab_array.count_attached()}/4 | "
              f"Alert: {defense['alert_level']}")
    
    def inject_threat(self, 
                      position: np.ndarray = None,
                      velocity: np.ndarray = None,
                      threat_type: ThreatType = ThreatType.MISSILE_IR):
        """
        Inject a threat for testing defensive response.
        
        Args:
            position: Threat starting position (default: 400m ahead)
            velocity: Threat velocity vector (default: toward mother drone)
            threat_type: Type of threat
        """
        if position is None:
            # Default: threat approaching from ahead
            position = self.mother_drone.position + np.array([400, 50, 0])
        
        if velocity is None:
            # Default: heading toward mother drone at 100 m/s
            to_mother = self.mother_drone.position - position
            to_mother = to_mother / (np.linalg.norm(to_mother) + 1e-9)
            velocity = to_mother * 100
        
        threat_id = self.defensive_ai.detect_threat(
            position,
            velocity,
            threat_type,
            self.mother_drone.position
        )
        
        print(f"THREAT INJECTED: {threat_id} - {threat_type.value}")
        return threat_id
    
    def execute_speed_burst(self, verbose: bool = False):
        """
        Execute emergency speed burst maneuver.
        
        Releases all TABs for maximum acceleration.
        """
        if verbose:
            print("EXECUTING SPEED BURST MANEUVER")
        
        # Get pre-release state
        pre_speed = self.mother_drone.speed
        
        # Release all TABs
        released = self.mother_drone.release_all_tabs()
        self.tether_array.release_all()
        
        # Update TAB states
        for tab in self.tab_array.tabs.values():
            release_vel = self.momentum_engine.compute_release_velocity(
                self.mother_drone.momentum_state,
                tab.momentum_state,
                0,  # No tension at release
                ReleaseMode.INSTANT
            )
            tab.execute_release(release_vel)
        
        if verbose:
            print(f"Released {released} TABs")
            print(f"Pre-release speed: {pre_speed:.1f} m/s")
        
        # Calculate expected acceleration
        burst_data = self.momentum_engine.compute_mother_acceleration_burst(
            self.mother_drone.momentum_state,
            [tab.momentum_state for tab in self.tab_array.tabs.values()],
            self.mother_drone.current_thrust,
            self.mother_drone.compute_parasitic_drag()
        )
        
        if verbose:
            print(f"Expected acceleration multiplier: {burst_data['acceleration_multiplier']:.1f}x")
            print(f"Drag reduction: {burst_data['drag_reduction_percent']:.1f}%")
            print(f"Predicted speed in 1s: {burst_data['predicted_speed_1s']:.1f} m/s")


def demo_slingshot_intercept():
    """
    Demonstration: Slingshot intercept of incoming threat.
    
    Shows the "Archery" mechanics where a TAB is released
    on an intercept trajectory.
    """
    print("\n" + "=" * 60)
    print("DEMO: SLINGSHOT INTERCEPT MANEUVER")
    print("=" * 60 + "\n")
    
    # Create simulation
    sim = KAPSSimulation()
    
    # Run for 5 seconds to stabilize formation
    print("Phase 1: Formation stabilization...")
    for _ in range(5000):  # 5 seconds at 0.001s timestep
        sim.step()
    
    print(f"Formation stable at T={sim.time:.1f}s")
    print(f"Mother drone speed: {sim.mother_drone.speed:.1f} m/s")
    print(f"TABs attached: {sim.tab_array.count_attached()}")
    
    # Inject threat
    print("\nPhase 2: Threat injection...")
    sim.inject_threat(
        position=sim.mother_drone.position + np.array([300, 20, -10]),
        velocity=np.array([-80, 0, 0]),  # Heading toward drone
        threat_type=ThreatType.DRONE_SUICIDE
    )
    
    # Run for 3 more seconds (AI should respond)
    print("\nPhase 3: Defensive response...")
    for _ in range(3000):
        telemetry = sim.step()
    
    print(f"\nFinal state at T={sim.time:.1f}s:")
    print(f"TABs remaining: {sim.tab_array.count_attached()}")
    print(f"Intercepts executed: {sim.defensive_ai.intercepts_executed}")
    

def demo_speed_burst():
    """
    Demonstration: Speed burst escape maneuver.
    
    Shows the "mechanical capacitor" effect when all TABs
    are released simultaneously.
    """
    print("\n" + "=" * 60)
    print("DEMO: SPEED BURST MANEUVER")
    print("=" * 60 + "\n")
    
    sim = KAPSSimulation()
    
    # Run for 2 seconds
    print("Pre-burst flight...")
    for _ in range(2000):
        sim.step()
    
    print(f"T={sim.time:.1f}s - Initiating speed burst")
    print(f"Current speed: {sim.mother_drone.speed:.1f} m/s")
    
    # Execute speed burst
    sim.execute_speed_burst()
    
    # Run for 2 more seconds to see acceleration
    print("\nPost-burst acceleration...")
    for _ in range(2000):
        telemetry = sim.step()
        if int(sim.time * 100) % 50 == 0:
            print(f"  T={sim.time:.1f}s - Speed: {telemetry['mother_drone']['speed']:.1f} m/s")
    
    print(f"\nFinal speed: {sim.mother_drone.speed:.1f} m/s")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "slingshot":
            demo_slingshot_intercept()
        elif sys.argv[1] == "burst":
            demo_speed_burst()
        else:
            print("Unknown demo. Options: 'slingshot', 'burst'")
    else:
        # Run main simulation
        config_path = Path(__file__).parent.parent / "config" / "simulation_params.yaml"
        
        if config_path.exists():
            sim = KAPSSimulation(str(config_path))
        else:
            sim = KAPSSimulation()
        
        # Run for 10 seconds as demo
        sim.run(duration=10.0)
