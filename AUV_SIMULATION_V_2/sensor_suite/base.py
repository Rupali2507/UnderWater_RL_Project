# File: sensor_suite/base.py
from dataclasses import dataclass
import time

@dataclass
class Detection:
    object_id: str
    position: tuple
    confidence: float
    source_sensor: str
    timestamp: float

class Sensor:
    def __init__(self, max_range: float, update_frequency: float, noise_level: float):
        self.max_range = max_range
        self.update_frequency = update_frequency
        self._noise_level = noise_level


class SensorSuite:
    """
    Unified container and access point for all sensing and communication components.
    Operates as a bridge between the agent's vehicle state and the environment's global tensor.
    """
    def __init__(
        self, 
        radar=None, 
        sonar=None, 
        visual_sensor=None, 
        ais_receiver=None, 
        communication_module=None
    ):
        self._radar = radar
        self._sonar = sonar
        self._visual_sensor = visual_sensor
        self._ais_receiver = ais_receiver
        self._communication_module = communication_module

    @property
    def communication_module(self):
        return self._communication_module

    def scan(self, vehicle, environment_tensor) -> dict:
        """
        Queries all available sensors and returns a unified dictionary of detections.
        
        Args:
            vehicle: The agent performing the scan.
            environment_tensor: The master state tensor containing all agent positions.
        """
        detections = {}
        
        if self._radar:
            detections['RADAR'] = self._radar.scan(vehicle, environment_tensor)
            
        if self._sonar:
            detections['SONAR'] = self._sonar.scan(vehicle, environment_tensor, mode='passive')
            
        if self._visual_sensor:
            detections['VISUAL'] = self._visual_sensor.scan(vehicle, environment_tensor)
            
        if self._ais_receiver:
            detections['AIS'] = self._ais_receiver.scan(vehicle, environment_tensor)
            
        return detections

    def process_incoming_comms(self, inbox: list, perception_memory: dict):
        """Flushes the communication queue into the vehicle's perception memory."""
        if self._communication_module:
            self._communication_module.process_queue(inbox, perception_memory)    

    def get_equipped_sensors(self) -> list[str]:
        return [name for name, sensor in {
            "RADAR": self._radar,
            "SONAR": self._sonar,
            "VISUAL": self._visual_sensor,
            "AIS": self._ais_receiver,
            "COMMS": self._communication_module
        }.items() if sensor is not None]

    # ------------------------------------------------------------------
    # TACTICAL LOOKUP HELPERS
    # Called by vehicle get_observation() methods to find nearby allies/threats.
    # TacticalNavalEnv calls SensorSuite.set_vehicle_registry(vehicles) once
    # after building the fleet so every sensor suite shares the same reference.
    # ------------------------------------------------------------------
    _vehicle_registry = None   # class-level, shared across all instances

    @classmethod
    def set_vehicle_registry(cls, registry: dict):
        """Called once by TacticalNavalEnv so all sensor suites share the fleet."""
        cls._vehicle_registry = registry

    def _get_nearest_pos_by_class(self, own_pos: tuple, class_name: str):
        """
        Returns the (x, y) of the nearest vehicle whose class name matches.
        Returns None if no matching vehicle is found or registry is empty.
        """
        import math
        if self._vehicle_registry is None:
            return None

        best_pos = None
        best_dist = float('inf')
        for vehicle in self._vehicle_registry.values():
            if type(vehicle).__name__ == class_name:
                pos = vehicle.position       
                dx = pos[0] - own_pos[0]
                dy = pos[1] - own_pos[1]
                dist = math.sqrt(dx * dx + dy * dy)
                if dist < best_dist:
                    best_dist = dist
                    best_pos = (pos[0], pos[1])
        return best_pos

    def get_nearest_threat_pos(self, own_pos: tuple):
        """Returns (x, y) of the nearest ThreatShip, or None."""
        return self._get_nearest_pos_by_class(own_pos, 'ThreatShip')

    def get_nearest_cargo_pos(self, own_pos: tuple):
        """Returns (x, y) of the nearest CargoShip, or None."""
        return self._get_nearest_pos_by_class(own_pos, 'CargoShip')

    def get_nearest_escort_pos(self, own_pos: tuple):
        """Returns (x, y) of the nearest NavalShip (non-threat escort), or None."""
        import math
        if self._vehicle_registry is None:
            return None

        best_pos = None
        best_dist = float('inf')
        for vehicle in self._vehicle_registry.values():
            if type(vehicle).__name__ == 'NavalShip':
                pos = vehicle.position
                dx = pos[0] - own_pos[0]
                dy = pos[1] - own_pos[1]
                dist = math.sqrt(dx * dx + dy * dy)
                if dist < best_dist:
                    best_dist = dist
                    best_pos = (pos[0], pos[1])
        return best_pos