import math
from abc import ABC, abstractmethod
from collections import namedtuple
from enum import Enum
from sensor_suite.base import SensorSuite


class VehicleStatus(Enum):
    """Generic lifecycle status, primarily for debugging/monitoring across the fleet."""
    IDLE = 0
    PATROL = 1
    INTERCEPT = 2
    NEGOTIATE = 3
    COMBAT = 4
    RETREAT = 5
    MOVING = 6
    DESTROYED = 7


# Fixed, lightweight contract for inter-agent communication.
Message = namedtuple("Message", ["sender_id", "msg_type", "position", "payload", "timestamp"])


class AutonomousVehicle(ABC):
    """
    Abstract Base Class for all autonomous vehicles in the simulation.
    Strictly adheres to encapsulation, preventing external systems from
    modifying internal state without using the proper public interfaces.
    """

    
    # TENSOR SCHEMA (16-Column 6-DOF Layout)
    COL_POSITION = slice(0, 3)          # 0:X, 1:Y, 2:Z
    COL_ORIENTATION = slice(3, 6)       # 3:Roll (phi), 4:Pitch (theta), 5:Yaw (psi)
    COL_VELOCITY = slice(6, 9)          # 6:u (surge), 7:v (sway), 8:w (heave)
    COL_ANGULAR_VEL = slice(9, 12)      # 9:p (roll rate), 10:q (pitch rate), 11:r (yaw rate)
    COL_HEALTH = 12                     # 12:Health
    COL_FUEL = 13                       # 13:Fuel
    COL_TEAM_ID = 14                    # 14:Team ID (e.g., 1=Friendly, -1=Threat, 0=Cargo)
    COL_STATUS = 15                     # 15:VehicleStatus (int cast)

    def __init__(
        self,
        vehicle_id: str,
        team: str,
        agent_index: int,
        state_tensor,
        start_position: tuple,
        dest_position: tuple = (1800.0, 1800.0, 0.0),
        collision_radius: float = 20.0,
        comm_range: float = 5000.0,
        initial_health: float = 100.0,
        initial_fuel: float = 100.0,
        min_operational_fuel: float = 0.0,
        team_id_val: float = 0.0, 
        **kwargs
    ):
        # --- Identity & Engine Attributes (Public) ---
        self.vehicle_id = vehicle_id
        self.team = team
        self.agent_index = agent_index
        self.state_tensor = state_tensor
        self._team_id_val = team_id_val  # Float representation for the tensor

        # --- Mission Attributes (Protected) ---
        self._start_position = start_position
        self._dest_position = dest_position
        self._nav_destination = dest_position  # Immutable original
        # --- Initial Condition Attributes (Protected) ---
        self._initial_health = initial_health
        self._initial_fuel = initial_fuel
        self._min_operational_fuel = min_operational_fuel

        # --- State Attributes (Protected) ---
        self._position = start_position
        self._health = initial_health
        self._fuel = initial_fuel
        self._status = VehicleStatus.IDLE
        self._is_destroyed = False

        # --- Physical Attributes (Protected) ---
        self._collision_radius = collision_radius

        # --- System Attributes (Protected) ---
        self._comm_range = comm_range
        self.sensor_suite = SensorSuite()
        self._perception_memory = {}
        self._inbox = [] 

    # ==========================================
    # PUBLIC PROPERTIES (Tensor-Synchronized Encapsulation)
    # ==========================================
    @property
    def position(self):
        if self.state_tensor is not None:
            return tuple(self.state_tensor[self.agent_index, self.COL_POSITION].cpu().numpy())
        return self._position

    @property
    def health(self):
        if self.state_tensor is not None:
            return float(self.state_tensor[self.agent_index, self.COL_HEALTH].item())
        return self._health

    @property
    def fuel(self):
        if self.state_tensor is not None:
            return float(self.state_tensor[self.agent_index, self.COL_FUEL].item())
        return self._fuel

    @property
    def speed(self):
        """Scalar speed derived from the linear velocity (u, v, w)."""
        if self.state_tensor is not None and self.state_tensor.shape[1] > self.COL_VELOCITY.stop - 1:
            u, v, w = self.state_tensor[self.agent_index, self.COL_VELOCITY].cpu().numpy()
            return float(math.sqrt(u**2 + v**2 + w**2))
        return 0.0

    @property
    def status(self):
        return self._status

    @status.setter
    def status(self, value: VehicleStatus):
        self._status = value
        if self.state_tensor is not None and self.state_tensor.shape[1] > self.COL_STATUS:
            self.state_tensor[self.agent_index, self.COL_STATUS] = float(value.value)

    @property
    def start_position(self): return self._start_position

    @property
    def dest_position(self): return self._dest_position

    @property
    def collision_radius(self): return self._collision_radius

    @property
    def comm_range(self): return self._comm_range

    @property
    def perception_memory(self): return self._perception_memory

    @property
    def sensor_suite(self): return self._sensor_suite

    @sensor_suite.setter
    def sensor_suite(self, suite):
        self._sensor_suite = suite

    # ==========================================
    # CORE METHODS (Public Abstract)
    # ==========================================
    @abstractmethod
    def move(self, action, dt: float):
        """
        Applies action to physics tensor over time step dt.
        Subclasses must update VELOCITY and ANGULAR_VELOCITY indices, 
        and integrate them into POSITION and ORIENTATION.
        """
        pass

    @abstractmethod
    def sense(self, environment=None):
        pass

    @abstractmethod
    def communicate(self):
        pass

    @abstractmethod
    def get_observation(self):
        pass

    @abstractmethod
    def get_action_space(self):
        pass

    # ==========================================
    # CORE METHODS (Public)
    # ==========================================
    def take_damage(self, amount: float):
        """Applies damage directly to the physics tensor to ensure immediate combat awareness."""
        if self.state_tensor is not None:
            current = self.state_tensor[self.agent_index, self.COL_HEALTH].item()
            self.state_tensor[self.agent_index, self.COL_HEALTH] = max(0.0, current - amount)
        self._health = max(0.0, self._health - amount)

        if not self.is_alive() and not self._is_destroyed:
            self._is_destroyed = True
            self.status = VehicleStatus.DESTROYED
            self._on_death()

    def is_alive(self) -> bool:
        """Returns whether the vehicle remains operational: health AND fuel both sufficient."""
        return self.health > 0.0 and self.fuel > self._min_operational_fuel

    def is_goal_reached(self) -> bool:
        """Returns True if the vehicle is within its collision radius of the destination."""
        distance = self._get_distance_to(self._dest_position)
        return distance <= self._collision_radius

    def receive_message(self, message: Message):
        """Queues an incoming Message for this vehicle to process during communicate()."""
        self._inbox.append(message)

    def reset(self):
        """Restores the vehicle to its OWN initial state at the beginning of a new episode."""
        self._position = self._start_position
        self._health = self._initial_health
        self._fuel = self._initial_fuel
        self._perception_memory = {}
        self._inbox = []
        self._is_destroyed = False
        self.status = VehicleStatus.IDLE

        if self.state_tensor is not None:
            # Zero out all movement and rotation
            self.state_tensor[self.agent_index, self.COL_VELOCITY] = 0.0
            self.state_tensor[self.agent_index, self.COL_ANGULAR_VEL] = 0.0
            self.state_tensor[self.agent_index, self.COL_ORIENTATION] = 0.0

            # Set starting coordinates
            self.state_tensor[self.agent_index, 0] = self._start_position[0]
            self.state_tensor[self.agent_index, 1] = self._start_position[1]
            if len(self._start_position) > 2:
                self.state_tensor[self.agent_index, 2] = self._start_position[2]
            else:
                self.state_tensor[self.agent_index, 2] = 0.0

            # Set static properties
            self.state_tensor[self.agent_index, self.COL_HEALTH] = self._initial_health
            self.state_tensor[self.agent_index, self.COL_FUEL] = self._initial_fuel
            
            # Map optional team identifier if tensor supports it
            if self.state_tensor.shape[1] > self.COL_TEAM_ID:
                self.state_tensor[self.agent_index, self.COL_TEAM_ID] = self._team_id_val

    # ==========================================
    # HELPER METHODS (Protected)
    # ==========================================
    def _on_death(self):
        """
        Hook invoked exactly once, the moment this vehicle becomes non-operational.
        """
        pass

    def _get_distance_to(self, target_position: tuple) -> float:
        """Computes the 3D Euclidean distance between current position and target."""
        pos = self.position
        dx = pos[0] - target_position[0]
        dy = pos[1] - target_position[1]
        
        target_z = target_position[2] if len(target_position) > 2 else 0.0
        pos_z = pos[2] if len(pos) > 2 else 0.0
        dz = pos_z - target_z

        return math.sqrt(dx**2 + dy**2 + dz**2)

    def _get_bearing_to(self, target_position: tuple) -> float:
        """Computes the bearing (in radians) from the vehicle toward the target (XY plane)."""
        pos = self.position
        dx = target_position[0] - pos[0]
        dy = target_position[1] - pos[1]
        return math.atan2(dy, dx)