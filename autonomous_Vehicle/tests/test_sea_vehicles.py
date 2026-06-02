import unittest
from autonomous_vehicles.environment import OceanEnvironment
from autonomous_vehicles.sea.sea_vehicles import SurfaceVehicle, UnderwaterVehicle, CargoShip, Rig

class TestSeaVehicles(unittest.TestCase):
    def setUp(self):
        # Initialize a standard test ocean environment
        self.env = OceanEnvironment(temperature=15.0, current_speed=2.0, current_direction="NE")

    # SurfaceVehicle Tests 
    def test_surface_vehicle_moor_safe(self):
        self.env.sea_state = 3  # Safe sea state
        vessel = SurfaceVehicle("V-01", "SeaRunner", 25.0, "monohull", 500, self.env, "planing", 3.5, 120.0)
        self.assertTrue(vessel.moor("Port Alpha"))

    def test_surface_vehicle_moor_unsafe(self):
        self.env.sea_state = 8  # Hazardous sea state (> 6)
        vessel = SurfaceVehicle("V-02", "StormWatcher", 15.0, "monohull", 1200, self.env, "displacement", 5.0, 300.0)
        self.assertFalse(vessel.moor("Port Beta"))

    # UnderwaterVehicle Tests 
    def test_underwater_vehicle_dive_safe(self):
        self.env.max_depth = 500.0  # Safe depth limit
        sub = UnderwaterVehicle("SUB-01", "Nautilus", 12.0, "submarine_hull", 2000, self.env, 10.0, 50.0, 48.0, 400.0)
        self.assertTrue(sub.dive(200.0))
        self.assertEqual(sub.current_depth, 200.0)

    def test_underwater_vehicle_dive_unsafe_environment(self):
        self.env.max_depth = 100.0  # Shallow hazard zone
        sub = UnderwaterVehicle("SUB-02", "DeepDive", 10.0, "submarine_hull", 1500, self.env, 8.0, 40.0, 24.0, 300.0)
        # Should fail because environment max_depth is 100m, but target is 150m
        self.assertFalse(sub.dive(150.0))