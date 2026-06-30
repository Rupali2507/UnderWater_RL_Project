import torch
import numpy as np
import gymnasium as gym
import math

from autonomous_vehicles.sea.surface_vehicle.surface_vehicle_base import SurfaceVehicle
from autonomous_vehicles.base import VehicleStatus, Message


class CargoShip(SurfaceVehicle):
    """
    Represents a commercial maritime vessel transporting cargo.
    Focuses on route planning, load management, and coordination with allied escorts.
    Acts as the primary objective/VIP in the Zone C protection simulation.
    """

    def __init__(self, vehicle_id, agent_index, state_tensor, start_position, dest_position, **kwargs):
        kwargs.pop("team", None)

        displacement            = kwargs.pop("displacement", 50000.0)
        water_drag_coefficient  = kwargs.pop("water_drag_coefficient", 0.8)
        turn_radius             = kwargs.pop("turn_radius", 800.0)
        draft                   = kwargs.pop("draft", 14.0)
        sea_state_tolerance     = kwargs.pop("sea_state_tolerance", 5)
        visual_range            = kwargs.pop("visual_range", 12000.0)
        cargo_type              = kwargs.pop("cargo_type", "GENERAL")
        initial_cargo_load      = kwargs.pop("initial_cargo_load", 100.0)

        super().__init__(
            vehicle_id=vehicle_id,
            team="CARGO",
            agent_index=agent_index,
            state_tensor=state_tensor,
            start_position=start_position,
            dest_position=dest_position,
            displacement=displacement,
            water_drag_coefficient=water_drag_coefficient,
            turn_radius=turn_radius,
            draft=draft,
            sea_state_tolerance=sea_state_tolerance,
            visual_range=visual_range,
            team_id_val=0.0,
            **kwargs
        )

        self._cargo_type        = cargo_type
        self._cargo_load        = initial_cargo_load
        self._escort_vessels    = []
        self._base_displacement = displacement
        self._base_drag         = water_drag_coefficient

        self.adjust_speed_for_load()
        self.status = VehicleStatus.MOVING

    # ==========================================
    # CARGO PROPERTIES & LOGIC
    # ==========================================
    @property
    def cargo_type(self): return self._cargo_type

    @property
    def cargo_load(self): return self._cargo_load

    @cargo_load.setter
    def cargo_load(self, value):
        self._cargo_load = max(0.0, min(100.0, value))
        self.adjust_speed_for_load()

    @property
    def escort_vessels(self): return self._escort_vessels

    def add_escort(self, naval_ship_id):
        if naval_ship_id not in self._escort_vessels:
            self._escort_vessels.append(naval_ship_id)

    def adjust_speed_for_load(self):
        load_factor = self._cargo_load / 100.0
        self._displacement             = self._base_displacement * (1.0 + 0.5 * load_factor)
        self._water_drag_coefficient   = self._base_drag         * (1.0 + 0.2 * load_factor)
        self._push_maritime_constants_to_tensor()

    # ==========================================
    # CORE METHODS
    # ==========================================
    def move(self, action, dt):
        if not self.is_alive():
            return
        self._apply_surface_physics(action)

    def _apply_surface_physics(self, action):
        pass

    def sense(self, environment=None):
        if self.sensor_suite and environment:
            detections = self.sensor_suite.scan(self.position, self.visual_range, environment)
            self._perception_memory.update(detections)

    def communicate(self):
        outbound_messages = []
        while self._inbox:
            self._inbox.pop(0)
        if self.ais_active:
            outbound_messages.append(Message(
                sender_id=self.vehicle_id,
                msg_type="AIS_HEARTBEAT",
                position=self.position,
                payload={"cargo_type": self.cargo_type, "health": self.health},
                timestamp=0.0
            ))
        return outbound_messages

    def broadcast_distress(self):
        return Message(
            sender_id=self.vehicle_id,
            msg_type="DISTRESS_MAYDAY",
            position=self.position,
            payload={"cargo_type": self.cargo_type, "escorts_needed": True},
            timestamp=0.0
        )

    # ==========================================
    # REINFORCEMENT LEARNING INTERFACES
    # ==========================================
    def get_observation(self) -> np.ndarray:
        if self.state_tensor is None:
            return np.zeros(8, dtype=np.float32)

        x     = self.state_tensor[self.agent_index, 0].item()
        y     = self.state_tensor[self.agent_index, 1].item()
        u_vel = self.state_tensor[self.agent_index, self.COL_VELOCITY.start].item()
        v_vel = self.state_tensor[self.agent_index, self.COL_VELOCITY.start + 1].item()
        yaw   = self.state_tensor[self.agent_index, self.COL_ORIENTATION.stop - 1].item()

        dest_x, dest_y  = self._dest_position[0], self._dest_position[1]
        dist_to_dest    = math.sqrt((dest_x - x)**2 + (dest_y - y)**2)

        # Bearing to destination and heading error.
        # Without heading_error, the policy only knows HOW FAR, not WHICH WAY to turn.
        bearing_to_dest = math.atan2(dest_y - y, dest_x - x)
        heading_error   = math.atan2(
            math.sin(bearing_to_dest - yaw),
            math.cos(bearing_to_dest - yaw)
        )  # wrapped to [-pi, pi]

        threat_pos      = self.sensor_suite.get_nearest_threat_pos(self.position)
        dist_to_threat  = np.linalg.norm(np.array(threat_pos) - np.array([x, y])) if threat_pos else 5000.0

        obs = np.array([
            np.clip(u_vel / 20.0, -1.0, 1.0),                                                                    # surge speed
            np.clip(v_vel / 10.0, -1.0, 1.0),                                                                    # sway speed
            np.clip((yaw % (2 * math.pi)) / math.pi - 1.0, -1.0, 1.0),                                          # absolute heading
            np.clip(dist_to_dest / 7000.0, 0.0, 1.0),                                                            # dist to destination
            np.clip(heading_error / math.pi, -1.0, 1.0),                                                         # heading error (key nav signal)
            np.clip(dist_to_threat / 5000.0, 0.0, 1.0),                                                          # dist to threat
            np.clip(math.atan2(threat_pos[1]-y, threat_pos[0]-x) / math.pi, -1.0, 1.0) if threat_pos else 0.0,  # bearing to threat
            np.clip(self.health / 100.0, 0.0, 1.0),                                                              # health
        ], dtype=np.float32)

        return obs

    def get_action_space(self):
        return gym.spaces.Box(
            low=np.array([-0.2, -0.5]),
            high=np.array([1.0,  0.5]),
            dtype=np.float32
        )