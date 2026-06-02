from abc import ABC, abstractmethod
from typing import List, Optional


class AutonomousVehicle(ABC):
    """
    Abstract base class for all autonomous vehicles.
    All sea-based and aerial vehicle classes must inherit from this.
    """

    def __init__(
        self,
        vehicle_id: str,
        name: str,
        max_speed: float,
        battery_level: float = 1.0,
        environment: str = "unknown"
    ):
        self.vehicle_id: str = vehicle_id
        self.name: str = name
        self.max_speed: float = max_speed          # km/h or knots
        self.speed: float = 0.0                    # current speed
        self.battery_level: float = battery_level  # 0.0 to 1.0
        self.environment: str = environment        # 'sea' or 'aerial'
        self.weather_conditions: str = "clear"
        self.obstacles_detected: List[str] = []
        self.mission_active: bool = False
        self.mission_log: List[str] = []

    # ─── Abstract Methods (MUST be implemented by every subclass) ────────────

    @abstractmethod
    def navigate(self, destination: str) -> str:
        """
        Navigate to a given destination.
        Each vehicle type implements its own navigation logic.
        """
        pass

    @abstractmethod
    def get_vehicle_type(self) -> str:
        """
        Returns a string describing the vehicle type.
        e.g. 'CargoShip', 'Helicopter', etc.
        """
        pass

    # ─── Concrete Methods (shared by all vehicles) ───────────────────────────

    def start_mission(self) -> str:
        if self.battery_level <= 0.1:
            return f"[{self.name}] Cannot start mission — battery too low ({self.battery_level * 100:.0f}%)."
        if self.mission_active:
            return f"[{self.name}] Mission already active."
        self.mission_active = True
        log = f"Mission started for {self.name} (ID: {self.vehicle_id})."
        self.mission_log.append(log)
        return log

    def stop(self) -> str:
        self.speed = 0.0
        self.mission_active = False
        log = f"[{self.name}] Stopped. Mission ended."
        self.mission_log.append(log)
        return log

    def get_status(self) -> dict:
        return {
            "vehicle_id": self.vehicle_id,
            "name": self.name,
            "type": self.get_vehicle_type(),
            "environment": self.environment,
            "speed": self.speed,
            "max_speed": self.max_speed,
            "battery_level": f"{self.battery_level * 100:.0f}%",
            "mission_active": self.mission_active,
            "weather": self.weather_conditions,
            "obstacles": self.obstacles_detected,
        }

    def detect_obstacle(self, obstacle: str) -> str:
        self.obstacles_detected.append(obstacle)
        alert = f"[{self.name}] ALERT: Obstacle detected — '{obstacle}'."
        self.mission_log.append(alert)
        return alert

    def clear_obstacles(self) -> str:
        self.obstacles_detected.clear()
        return f"[{self.name}] Obstacle list cleared."

    def recharge(self, amount: float) -> str:
        if amount <= 0:
            return "Recharge amount must be positive."
        self.battery_level = min(1.0, self.battery_level + amount)
        return f"[{self.name}] Battery recharged to {self.battery_level * 100:.0f}%."

    def set_weather(self, condition: str) -> str:
        self.weather_conditions = condition
        return f"[{self.name}] Weather updated to '{condition}'."

    def accelerate(self, amount: float) -> str:
        if not self.mission_active:
            return f"[{self.name}] Cannot accelerate — no active mission."
        self.speed = min(self.max_speed, self.speed + amount)
        return f"[{self.name}] Speed increased to {self.speed:.1f}."

    def decelerate(self, amount: float) -> str:
        self.speed = max(0.0, self.speed - amount)
        return f"[{self.name}] Speed decreased to {self.speed:.1f}."

    def get_mission_log(self) -> List[str]:
        return self.mission_log


    def __str__(self) -> str:
        status = "ACTIVE" if self.mission_active else "IDLE"
        return (
            f"[{self.get_vehicle_type()}] {self.name} (ID: {self.vehicle_id}) | "
            f"Status: {status} | Speed: {self.speed}/{self.max_speed} | "
            f"Battery: {self.battery_level * 100:.0f}%"
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(vehicle_id={self.vehicle_id!r}, "
            f"name={self.name!r}, max_speed={self.max_speed})"
        )


# Run this file directly to verify the base class works before subclassing.

if __name__ == "__main__":

   
    

    class _TestVehicle(AutonomousVehicle):
        def navigate(self, destination: str) -> str:
            return f"[{self.name}] Navigating to {destination}."
        def get_vehicle_type(self) -> str:
            return "TestVehicle"

    v = _TestVehicle(vehicle_id="T-001", name="Prototype", max_speed=100.0, environment="sea")

    print(v)
    print(v.start_mission())
    print(v.accelerate(60))
    print(v.navigate("Port Alpha"))
    print(v.detect_obstacle("floating debris"))
    print(v.recharge(0.3))
    print(v.stop())
    print()
    print("--- Status ---")
    for key, val in v.get_status().items():
        print(f"  {key}: {val}")
    print()
    print("--- Mission Log ---")
    for entry in v.get_mission_log():
        print(f"  {entry}")
    print()
    print("repr:", repr(v))