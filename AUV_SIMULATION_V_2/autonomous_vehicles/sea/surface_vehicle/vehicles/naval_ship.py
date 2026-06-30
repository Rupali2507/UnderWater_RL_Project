import torch
import numpy as np
import gymnasium as gym
import math

from autonomous_vehicles.sea.surface_vehicle.surface_vehicle_base import SurfaceVehicle
from autonomous_vehicles.base import VehicleStatus, Message


class NavalShip(SurfaceVehicle):
    """
    Represents a military surface vessel (Zone B Defender).
    Escort / interception / shielding missions.
    """

    def __init__(self, vehicle_id, agent_index, state_tensor, start_position, dest_position, **kwargs):
        kwargs.pop("team", None)

        displacement           = kwargs.pop("displacement", 8000.0)
        water_drag_coefficient = kwargs.pop("water_drag_coefficient", 0.45)
        turn_radius            = kwargs.pop("turn_radius", 300.0)
        draft                  = kwargs.pop("draft", 9.0)
        sea_state_tolerance    = kwargs.pop("sea_state_tolerance", 7)
        visual_range           = kwargs.pop("visual_range", 25000.0)
        weapon_count           = kwargs.pop("weapon_count", 40)
        weapon_range           = kwargs.pop("weapon_range", 15000.0)
        patrol_radius          = kwargs.pop("patrol_radius", 5000.0)

        super().__init__(
            vehicle_id=vehicle_id,
            team="FRIENDLY",
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
            team_id_val=1.0,
            **kwargs
        )

        self._weapon_count      = weapon_count
        self._weapon_range      = weapon_range
        self._patrol_radius     = patrol_radius
        self._formation_position = None
        self._escort_target     = None
        self._current_threat    = None

        self.status = VehicleStatus.PATROL

    # ==========================================
    # COMBAT PROPERTIES & METHODS
    # ==========================================
    @property
    def weapon_count(self): return self._weapon_count

    @property
    def weapon_range(self): return self._weapon_range

    @property
    def patrol_radius(self): return self._patrol_radius

    def fire(self, target_vehicle):
        if self._weapon_count <= 0 or not self.is_alive():
            return False
        if self._get_distance_to(target_vehicle.position) <= self._weapon_range:
            self._weapon_count -= 1
            target_vehicle.take_damage(25.0)
            self.status = VehicleStatus.COMBAT
            return True
        return False

    # ==========================================
    # TACTICAL NAVIGATION METHODS
    # ==========================================
    def assign_escort(self, cargo_ship, formation_offset=(500.0, 500.0, 0.0)):
        self._escort_target      = cargo_ship
        self._formation_position = formation_offset
        if hasattr(cargo_ship, 'add_escort'):
            cargo_ship.add_escort(self.vehicle_id)

    def intercept(self, threat_position):
        if isinstance(threat_position, torch.Tensor):
            threat_position = tuple(threat_position.cpu().numpy())
        self._dest_position = threat_position
        self.status = VehicleStatus.INTERCEPT

    def cover_retreat(self, ally_position, threat_position):
        ally_pos   = np.array(ally_position[:2])
        threat_pos = np.array(threat_position[:2])
        mid        = (ally_pos + threat_pos) / 2.0
        self._dest_position = (float(mid[0]), float(mid[1]), 0.0)
        self.status = VehicleStatus.MOVING

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
            msg = self._inbox.pop(0)
            if msg.msg_type == "DISTRESS_MAYDAY":
                self.intercept(msg.position)
            elif msg.msg_type == "THREAT_DETECTED":
                self._current_threat = msg.payload.get("threat_id")
                self.intercept(msg.position)
        if self.status in [VehicleStatus.COMBAT, VehicleStatus.INTERCEPT]:
            outbound_messages.append(Message(
                sender_id=self.vehicle_id,
                msg_type="TACTICAL_UPDATE",
                position=self.position,
                payload={"status": self.status.name, "ammo": self.weapon_count},
                timestamp=0.0
            ))
        return outbound_messages

    def _on_death(self):
        self.nav_lights_active = False
        self.ais_active = False
        if self.state_tensor is not None:
            self.state_tensor[self.agent_index, self.COL_VELOCITY] = 0.0
        return [Message(
            sender_id=self.vehicle_id,
            msg_type="NAVAL_DESTROYED",
            position=self.position,
            payload={"info": "Defender lost. Zone B compromised."},
            timestamp=0.0
        )]

    # ==========================================
    # REINFORCEMENT LEARNING INTERFACES
    # ==========================================
    def get_observation(self) -> np.ndarray:
        if self.state_tensor is None:
            return np.zeros(9, dtype=np.float32)

        x     = self.state_tensor[self.agent_index, 0].item()
        y     = self.state_tensor[self.agent_index, 1].item()
        u_vel = self.state_tensor[self.agent_index, self.COL_VELOCITY.start].item()
        v_vel = self.state_tensor[self.agent_index, self.COL_VELOCITY.start + 1].item()
        yaw   = self.state_tensor[self.agent_index, self.COL_ORIENTATION.stop - 1].item()

        cargo_pos  = self.sensor_suite.get_nearest_cargo_pos(self.position)
        threat_pos = self.sensor_suite.get_nearest_threat_pos(self.position)

        dist_to_cargo  = np.linalg.norm(np.array(cargo_pos)  - np.array([x, y])) if cargo_pos  else 5000.0
        dist_to_threat = np.linalg.norm(np.array(threat_pos) - np.array([x, y])) if threat_pos else 5000.0

        # Heading error to current _dest_position (the live shield point).
        # Without this the escort has no directional signal toward the shield point.
        dest_x, dest_y  = self._dest_position[0], self._dest_position[1]
        bearing_to_dest = math.atan2(dest_y - y, dest_x - x)
        heading_error   = math.atan2(
            math.sin(bearing_to_dest - yaw),
            math.cos(bearing_to_dest - yaw)
        )  # wrapped to [-pi, pi]

        obs = np.array([
            np.clip(u_vel / 20.0, -1.0, 1.0),                                                                     # surge speed
            np.clip(v_vel / 10.0, -1.0, 1.0),                                                                     # sway speed
            np.clip((yaw % (2 * math.pi)) / math.pi - 1.0, -1.0, 1.0),                                           # absolute heading
            np.clip(heading_error / math.pi, -1.0, 1.0),                                                          # heading error to shield point (key signal)
            np.clip(dist_to_cargo / 5000.0, 0.0, 1.0),                                                            # dist to cargo
            np.clip(dist_to_threat / 5000.0, 0.0, 1.0),                                                           # dist to threat
            np.clip(self.health / 100.0, 0.0, 1.0),                                                               # health
            np.clip(math.atan2(cargo_pos[1]-y,  cargo_pos[0]-x)  / math.pi, -1.0, 1.0) if cargo_pos  else 0.0,   # bearing to cargo
            np.clip(math.atan2(threat_pos[1]-y, threat_pos[0]-x) / math.pi, -1.0, 1.0) if threat_pos else 0.0,   # bearing to threat
        ], dtype=np.float32)

        return obs

    def get_action_space(self):
        return gym.spaces.Box(
            low=np.array([-0.5, -1.0]),
            high=np.array([1.0,  1.0]),
            dtype=np.float32
        )