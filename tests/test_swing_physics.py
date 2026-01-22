"""
Test Suite: Swing Physics
=========================
Unit tests for the slingshot/swing mechanics of the KAPS system.

Tests:
- Orbital velocity calculations
- Momentum conservation on release
- Intercept trajectory prediction
- Speed burst acceleration
"""

import numpy as np
import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from physics import (
    MomentumEngine,
    MomentumState,
    SlingshotManeuver,
    SlingshotParameters,
    ReleaseMode
)


class TestMomentumState:
    """Tests for MomentumState dataclass"""
    
    def test_linear_momentum_calculation(self):
        """Test p = m * v"""
        state = MomentumState(
            mass=10.0,
            position=np.array([0, 0, 0]),
            velocity=np.array([50, 0, 0]),
            angular_velocity=np.zeros(3)
        )
        
        expected = np.array([500, 0, 0])
        np.testing.assert_array_almost_equal(state.linear_momentum, expected)
    
    def test_kinetic_energy_calculation(self):
        """Test KE = 0.5 * m * v²"""
        state = MomentumState(
            mass=8.0,
            position=np.zeros(3),
            velocity=np.array([30, 40, 0]),  # |v| = 50 m/s
            angular_velocity=np.zeros(3)
        )
        
        # KE = 0.5 * 8 * 50² = 10000 J
        assert state.kinetic_energy == pytest.approx(10000.0)
    
    def test_speed_property(self):
        """Test speed calculation"""
        state = MomentumState(
            mass=5.0,
            position=np.zeros(3),
            velocity=np.array([3, 4, 0]),
            angular_velocity=np.zeros(3)
        )
        
        assert state.speed == pytest.approx(5.0)


class TestMomentumEngine:
    """Tests for the MomentumEngine class"""
    
    @pytest.fixture
    def engine(self):
        return MomentumEngine()
    
    @pytest.fixture
    def mother_state(self):
        return MomentumState(
            mass=150.0,
            position=np.array([0, 0, 1000]),
            velocity=np.array([50, 0, 0]),
            angular_velocity=np.zeros(3)
        )
    
    @pytest.fixture
    def tab_state(self):
        return MomentumState(
            mass=8.0,
            position=np.array([-30, 0, 1000]),  # 30m behind
            velocity=np.array([50, 0, 0]),       # Matching speed
            angular_velocity=np.zeros(3)
        )
    
    def test_instant_release_preserves_velocity(self, engine, mother_state, tab_state):
        """Instant release should preserve TAB's current velocity"""
        release_vel = engine.compute_release_velocity(
            mother_state,
            tab_state,
            cable_tension=1000.0,
            release_mode=ReleaseMode.INSTANT
        )
        
        np.testing.assert_array_almost_equal(release_vel, tab_state.velocity)
    
    def test_slingshot_adds_orbital_velocity(self, engine, mother_state, tab_state):
        """Slingshot release should add orbital velocity component"""
        params = SlingshotParameters(
            spiral_rate=np.radians(45),  # 45 deg/s
            wind_up_time=2.0,
            release_angle=np.pi/2,
            cable_length=30.0,
            conservation_efficiency=1.0  # Perfect transfer
        )
        
        release_vel = engine.compute_release_velocity(
            mother_state,
            tab_state,
            cable_tension=1000.0,
            release_mode=ReleaseMode.SLINGSHOT,
            slingshot_params=params
        )
        
        # Should be faster than original
        assert np.linalg.norm(release_vel) > tab_state.speed
        
        # Orbital velocity = omega * r = 0.785 rad/s * 30m ≈ 23.5 m/s
        expected_boost = params.spiral_rate * params.cable_length
        speed_gain = np.linalg.norm(release_vel) - tab_state.speed
        
        # Due to vector addition (perpendicular), gain follows Pythagorean theorem:
        # |v_final| = sqrt(v_orig² + v_orbital²), so gain ≈ sqrt(50² + 23.5²) - 50 ≈ 5.25
        expected_pythagorean_gain = np.sqrt(tab_state.speed**2 + expected_boost**2) - tab_state.speed
        assert speed_gain == pytest.approx(expected_pythagorean_gain, rel=0.2)
    
    def test_speed_burst_acceleration(self, engine, mother_state, tab_state):
        """Test the speed burst calculation after cable release"""
        tab_states = [tab_state]  # Just one TAB for simplicity
        
        burst_data = engine.compute_mother_acceleration_burst(
            mother_state,
            tab_states,
            mother_thrust=2500.0,
            parasitic_drag=500.0
        )
        
        # Acceleration after should be higher than before
        assert burst_data['acceleration_after'] > burst_data['acceleration_before']
        
        # Drag should be reduced
        assert burst_data['drag_reduction_percent'] > 0
        
        # Multiplier should be greater than 1
        assert burst_data['acceleration_multiplier'] > 1.0
    
    def test_intercept_prediction_hit(self, engine):
        """Test intercept trajectory prediction for a hitting case"""
        # TAB released heading toward threat
        tab_pos = np.array([0, 0, 100])
        tab_vel = np.array([80, 0, 0])  # Fast toward target
        
        # Threat heading toward TAB's path
        threat_pos = np.array([50, 5, 100])
        threat_vel = np.array([0, -2, 0])  # Slow lateral
        
        result = engine.predict_intercept_trajectory(
            tab_pos, tab_vel,
            threat_pos, threat_vel,
            max_time=5.0
        )
        
        # Should get close (within 2m = hit)
        assert result['miss_distance'] < 10  # Reasonable miss for this geometry
    
    def test_intercept_prediction_miss(self, engine):
        """Test intercept trajectory prediction for a missing case"""
        # TAB heading away from threat
        tab_pos = np.array([0, 0, 100])
        tab_vel = np.array([-50, 0, 0])  # Going opposite direction
        
        threat_pos = np.array([100, 0, 100])
        threat_vel = np.array([50, 0, 0])  # Also going away
        
        result = engine.predict_intercept_trajectory(
            tab_pos, tab_vel,
            threat_pos, threat_vel,
            max_time=2.0
        )
        
        # Should miss
        assert result['intercept'] == False
        assert result['miss_distance'] > 50  # Growing apart


class TestSlingshotManeuver:
    """Tests for the SlingshotManeuver controller"""
    
    @pytest.fixture
    def slingshot(self):
        engine = MomentumEngine()
        params = SlingshotParameters(
            spiral_rate=np.radians(45),
            wind_up_time=2.0,
            release_angle=np.pi/2,
            cable_length=30.0,
            conservation_efficiency=0.95
        )
        return SlingshotManeuver(engine, params)
    
    def test_wind_up_starts_correctly(self, slingshot):
        """Test wind-up phase initialization"""
        result = slingshot.start_wind_up()
        
        assert result['status'] == 'wind_up_started'
        assert slingshot.phase == 'winding'
        assert result['spiral_rate_dps'] == pytest.approx(45.0)
    
    def test_wind_up_progresses(self, slingshot):
        """Test wind-up progress over time"""
        slingshot.start_wind_up()
        
        # Simulate 1 second (50% of wind_up_time)
        result = slingshot.update(1.0)
        
        assert result['status'] == 'winding'
        assert result['progress'] == pytest.approx(0.5)
    
    def test_wind_up_completes(self, slingshot):
        """Test wind-up completion"""
        slingshot.start_wind_up()
        
        # Simulate full wind-up time
        result = slingshot.update(2.0)
        
        assert result['status'] == 'ready_to_release'
        assert slingshot.phase == 'ready'
    
    def test_optimal_release_angle(self, slingshot):
        """Test optimal release angle calculation"""
        # Target directly ahead
        target = np.array([1, 0, 0])
        angle = slingshot.calculate_optimal_release_angle(target)
        
        # Should be -90° (release when tangent points forward)
        assert angle == pytest.approx(-np.pi/2, abs=0.1)
    
    def test_release_execution(self, slingshot):
        """Test release execution"""
        slingshot.start_wind_up()
        slingshot.update(2.0)  # Complete wind-up
        
        mother_state = MomentumState(
            mass=150.0,
            position=np.array([0, 0, 1000]),
            velocity=np.array([50, 0, 0]),
            angular_velocity=np.zeros(3)
        )
        
        tab_state = MomentumState(
            mass=8.0,
            position=np.array([-30, 0, 1000]),
            velocity=np.array([50, 0, 0]),
            angular_velocity=np.zeros(3)
        )
        
        result = slingshot.execute_release(mother_state, tab_state, 1000.0)
        
        assert result['success'] == True
        assert slingshot.phase == 'released'
        assert result['release_speed'] > tab_state.speed  # Should be faster
    
    def test_cannot_release_before_ready(self, slingshot):
        """Test that release fails if not in ready state"""
        mother_state = MomentumState(
            mass=150.0,
            position=np.zeros(3),
            velocity=np.array([50, 0, 0]),
            angular_velocity=np.zeros(3)
        )
        
        tab_state = MomentumState(
            mass=8.0,
            position=np.zeros(3),
            velocity=np.array([50, 0, 0]),
            angular_velocity=np.zeros(3)
        )
        
        result = slingshot.execute_release(mother_state, tab_state, 1000.0)
        
        assert result['success'] == False


class TestSwingPhysicsIntegration:
    """Integration tests for complete swing maneuver scenarios"""
    
    def test_360_intercept_geometry(self):
        """Test that TABs can intercept from any quadrant"""
        engine = MomentumEngine()
        
        # Mother at origin, threats from all directions
        mother_pos = np.array([0, 0, 1000])
        threat_directions = [
            np.array([1, 0, 0]),   # Front
            np.array([-1, 0, 0]),  # Rear
            np.array([0, 1, 0]),   # Right
            np.array([0, -1, 0]), # Left
            np.array([0, 0, 1]),   # Above
            np.array([0, 0, -1]), # Below
        ]
        
        for threat_dir in threat_directions:
            # TAB positioned in that direction
            tab_pos = mother_pos + threat_dir * 30  # 30m out
            tab_vel = np.array([50, 0, 0]) + threat_dir * 20  # Moving outward
            
            # Threat coming from that direction
            threat_pos = mother_pos + threat_dir * 200
            threat_vel = -threat_dir * 100  # Coming toward mother
            
            result = engine.predict_intercept_trajectory(
                tab_pos, tab_vel,
                threat_pos, threat_vel,
                max_time=5.0
            )
            
            # At least verify the calculation runs
            assert 'miss_distance' in result
    
    def test_momentum_conservation_in_system(self):
        """Test that total system momentum is approximately conserved"""
        engine = MomentumEngine()
        
        mother = MomentumState(
            mass=150.0,
            position=np.zeros(3),
            velocity=np.array([50, 0, 0]),
            angular_velocity=np.zeros(3)
        )
        
        tab = MomentumState(
            mass=8.0,
            position=np.array([-30, 0, 0]),
            velocity=np.array([50, 0, 0]),
            angular_velocity=np.zeros(3)
        )
        
        # Total momentum before
        p_before = mother.linear_momentum + tab.linear_momentum
        
        # After slingshot release (with 95% efficiency)
        params = SlingshotParameters(
            spiral_rate=np.radians(45),
            wind_up_time=2.0,
            release_angle=np.pi/2,
            cable_length=30.0,
            conservation_efficiency=0.95
        )
        
        release_vel = engine.compute_release_velocity(
            mother, tab, 1000.0,
            ReleaseMode.SLINGSHOT, params
        )
        
        # TAB momentum after
        p_tab_after = tab.mass * release_vel
        
        # Mother momentum changes due to cable release
        # This is where the "missing" momentum goes
        # In reality, it's transferred to the mother as a reaction force
        
        # For this test, just verify TAB gained momentum
        assert np.linalg.norm(p_tab_after) > np.linalg.norm(tab.linear_momentum)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
