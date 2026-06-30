import torch
import numpy as np
import math
import gymnasium as gym
        

# Adjust these imports if your project structure differs slightly
from autonomous_vehicles.sea.surface_vehicle.surface_vehicle_base import SurfaceVehicle
from sensor_suite.sensors.radar import Radar
from sensor_suite.sensors.comm import CommunicationModule

class ThreatShip(SurfaceVehicle):
    """
    Autonomous adversary vessel. 
    Focuses on target acquisition, sensor-driven interception, and coordination.
    """
    def __init__(self, vehicle_id, agent_index, state_tensor, start_position=(0.0, 0.0), dest_position=(0.0, 0.0), **kwargs):
        # 1. Strip the conflicting 'team' key
        kwargs.pop("team", None) 
        
        # 2. Extract physical constants with sensible defaults for a fast, light Threat Ship
        displacement = kwargs.pop("displacement", 1500.0)
        water_drag_coefficient = kwargs.pop("water_drag_coefficient", 0.015)
        turn_radius = kwargs.pop("turn_radius", 120.0)
        draft = kwargs.pop("draft", 3.5)
        sea_state_tolerance = kwargs.pop("sea_state_tolerance", 5)
        visual_range = kwargs.pop("visual_range", 14000.0)

        # Set default initial health safely for the base class to handle
        kwargs["initial_health"] = kwargs.get("initial_health", 100.0)

        # 3. Call the super constructor fulfilling the SurfaceVehicle contract
        super().__init__(
            vehicle_id=vehicle_id,
            team="RED",
            agent_index=agent_index, 
            state_tensor=state_tensor,
            start_position=start_position,
            displacement=displacement,
            water_drag_coefficient=water_drag_coefficient,
            turn_radius=turn_radius,
            draft=draft,
            sea_state_tolerance=sea_state_tolerance,
            visual_range=visual_range,
            dest_position=dest_position,
            team_id_val=0.0,       # 0.0 = Hostile/Red team
            **kwargs               
        )
        
        # 4. Threat-specific hardware
        self.sensor_suite.radar = Radar(radar_range=kwargs.get("radar_range", 12000.0))
        self.comm_module = CommunicationModule()
        
        # 5. Operational parameters
        self.max_speed = kwargs.get("max_speed", 25.0)
        self.is_threat = True
        # REMOVED: self.health = ... (Base class now handles this safely)
        
        if not hasattr(self, 'perception_memory'):
            self.perception_memory = {}
    
    def get_observation(self) -> np.ndarray:
        if self.state_tensor is None:
            return np.zeros(8, dtype=np.float32)

        # Extract local state
        x = self.state_tensor[self.agent_index, 0].item()
        y = self.state_tensor[self.agent_index, 1].item()
        u_vel = self.state_tensor[self.agent_index, self.COL_VELOCITY.start].item()
        v_vel = self.state_tensor[self.agent_index, self.COL_VELOCITY.start + 1].item()
        yaw = self.state_tensor[self.agent_index, self.COL_ORIENTATION.stop - 1].item()
        
        # Tactical: Target Cargo and Intercepting Escort
        cargo_pos = self.sensor_suite.get_nearest_cargo_pos(self.position)
        escort_pos = self.sensor_suite.get_nearest_escort_pos(self.position)
        
        dist_to_cargo = np.linalg.norm(np.array(cargo_pos) - np.array([x, y])) if cargo_pos else 5000.0
        dist_to_escort = np.linalg.norm(np.array(escort_pos) - np.array([x, y])) if escort_pos else 5000.0

        obs = np.array([
            np.clip(u_vel / 20.0, -1.0, 1.0),
            np.clip(v_vel / 10.0, -1.0, 1.0),
            np.clip((yaw % (2 * math.pi)) / math.pi - 1.0, -1.0, 1.0),
            np.clip(dist_to_cargo / 5000.0, 0.0, 1.0),
            np.clip(dist_to_escort / 5000.0, 0.0, 1.0),
            np.clip(self.health / 100.0, 0.0, 1.0),
            np.clip(math.atan2(cargo_pos[1]-y, cargo_pos[0]-x) / math.pi, -1.0, 1.0) if cargo_pos else 0.0,
            np.clip(math.atan2(escort_pos[1]-y, escort_pos[0]-x) / math.pi, -1.0, 1.0) if escort_pos else 0.0
        ], dtype=np.float32)
        
        return obs

    def receive_message(self, message):
        """
        Threats use internal comms to synchronize attacks.
        """
        self.perception_memory["COMMS"] = message

    def _apply_surface_physics(self, action):
        """
        PhysicsEngine handles the actual movement; this is a hook 
        for threat-specific steering or stabilization.
        """
        pass

    # ----------------------------------------------------------------
    # ABSTRACT METHOD IMPLEMENTATIONS (Required by SurfaceVehicle)
    # ----------------------------------------------------------------

    def move(self, action):
        """
        Satisfies the abstract base class. 
        Actual movement is handled globally by the Vectorized PhysicsEngine.
        """
        pass

    def sense(self, environment_state):
        """
        Satisfies the abstract base class.
        Hooks into the vehicle's sensor suite.
        """
        if hasattr(self, 'sensor_suite'):
            self.sensor_suite.scan(self, environment_state)

    def communicate(self, message):
        """
        Satisfies the abstract base class.
        Routes the message to the internal perception memory.
        """
        self.receive_message(message)

    def get_action_space(self):
        """
        Satisfies the abstract base class.
        Defines the control limits for Throttle [-0.5, 1.0] and Rudder [-1.0, 1.0].
        """
        
        return gym.spaces.Box(low=np.array([-0.5, -1.0]), high=np.array([1.0, 1.0]), dtype=np.float32)