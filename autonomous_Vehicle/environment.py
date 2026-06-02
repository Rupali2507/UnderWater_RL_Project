"""
environment.py — Shared environment and obstacle classes
Autonomous Vehicle OOP Project

Commit this to main alongside base.py before either intern branches.

Usage:
    from environment import OceanEnvironment, AerialEnvironment, ObstacleManager
"""

import math
from dataclasses import dataclass, field
from typing import Optional


# Obstacle dataclass (used by ObstacleManager)


@dataclass
class Obstacle:
    name: str
    lat: float
    lon: float
    obstacle_type: str          # e.g. 'reef', 'wreck', 'no-fly-zone', 'terrain'
    severity: str               # 'low' | 'medium' | 'high'
    radius_metres: float        # danger radius around the obstacle centre

    def __str__(self) -> str:
        return (f"Obstacle({self.name!r}, type={self.obstacle_type}, "
                f"severity={self.severity}, radius={self.radius_metres}m)")


# ObstacleManager — shared by both interns, environment-agnostic

class ObstacleManager:
    """
    Manages a runtime list of Obstacle objects.
    Uses the haversine formula for accurate GPS-based distance checks.

    Rules:
    - Never hardcode obstacles in __init__ of any vehicle class.
    - Always add obstacles at runtime via add_obstacle().
    - Intern 1 adds: reefs, wrecks, ice fields, shallow banks.
    - Intern 2 adds: no-fly zones, weather cells, terrain, towers.
    """

    EARTH_RADIUS_M = 6_371_000  # metres

    def __init__(self) -> None:
        self._obstacles: list[Obstacle] = []

    # -- Internals -----------------------------------------------------------

    def _haversine_distance(self, lat1: float, lon1: float,
                             lat2: float, lon2: float) -> float:
        """Return the great-circle distance in metres between two GPS points."""
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlambda = math.radians(lon2 - lon1)
        a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
        return 2 * self.EARTH_RADIUS_M * math.asin(math.sqrt(a))

    # -- Public API ----------------------------------------------------------

    def add_obstacle(self, lat: float, lon: float, name: str,
                     obstacle_type: str = "unknown",
                     severity: str = "medium",
                     radius_metres: float = 100.0) -> Obstacle:
        """Register a new obstacle and return it."""
        obs = Obstacle(
            name=name,
            lat=lat,
            lon=lon,
            obstacle_type=obstacle_type,
            severity=severity,
            radius_metres=radius_metres,
        )
        self._obstacles.append(obs)
        return obs

    def remove_obstacle(self, name: str) -> bool:
        """Remove an obstacle by name. Returns True if found and removed."""
        before = len(self._obstacles)
        self._obstacles = [o for o in self._obstacles if o.name != name]
        return len(self._obstacles) < before

    def check_position(self, lat: float, lon: float) -> list[Obstacle]:
        """Return all obstacles whose radius overlaps the given GPS position."""
        return [
            obs for obs in self._obstacles
            if self._haversine_distance(lat, lon, obs.lat, obs.lon) <= obs.radius_metres
        ]

    def get_nearby_obstacles(self, lat: float, lon: float,
                              radius_metres: float) -> list[Obstacle]:
        """Return all obstacles whose centre is within radius_metres of the position."""
        return [
            obs for obs in self._obstacles
            if self._haversine_distance(lat, lon, obs.lat, obs.lon) <= radius_metres
        ]

    def clear_all(self) -> None:
        """Remove all obstacles (useful for tests)."""
        self._obstacles.clear()

    @property
    def count(self) -> int:
        return len(self._obstacles)

    def __repr__(self) -> str:
        return f"ObstacleManager(obstacles={self.count})"


# OceanEnvironment 
class OceanEnvironment:
    """
    Models the ocean/sea environment for sea-based vehicles.

    Intern 1 usage:
        ocean = OceanEnvironment()
        super().__init__(..., environment=ocean)

        # In dive():
        if not self.environment.is_safe_for_diving(target_depth):
            raise RuntimeError("Unsafe dive conditions")

        # In moor():
        if self.environment.sea_state >= 6:
            raise RuntimeError("Sea state too rough to moor")
    """

    # Beaufort / WMO sea state descriptions (index = state 0-9)
    SEA_STATE_DESCRIPTIONS = [
        "Glassy",       # 0
        "Rippled",      # 1
        "Wavelets",     # 2
        "Slight",       # 3
        "Moderate",     # 4
        "Rough",        # 5
        "Very Rough",   # 6
        "High",         # 7
        "Very High",    # 8
        "Phenomenal",   # 9
    ]

    # Approximate significant wave heights per state (metres)
    _WAVE_HEIGHTS = [0.0, 0.1, 0.3, 0.9, 1.8, 3.0, 4.5, 6.0, 8.5, 12.0]

    def __init__(
        self,
        salinity: float = 35.0,
        temperature: float = 20.0,
        visibility: float = 100.0,
        current_speed: float = 0.0,
        current_direction: str = "N",
        sea_state: int = 0,
        max_depth: float = 500.0,
    ) -> None:
        self.salinity: float = salinity                  # PSU
        self.water_pressure: float = 1.0                 # atm (surface); updated via pressure_at_depth()
        self.temperature: float = temperature            # °C
        self.visibility: float = visibility              # metres
        self.current_speed: float = current_speed        # knots
        self.current_direction: str = current_direction  # compass string
        self._sea_state: int = max(0, min(9, sea_state))
        self.max_depth: float = max_depth                # metres

    # -- Sea state -----------------------------------------------------------

    @property
    def sea_state(self) -> int:
        return self._sea_state

    @sea_state.setter
    def sea_state(self, value: int) -> None:
        self._sea_state = max(0, min(9, value))

    @property
    def sea_state_description(self) -> str:
        return self.SEA_STATE_DESCRIPTIONS[self._sea_state]

    @property
    def wave_height(self) -> float:
        """Approximate significant wave height in metres for current sea state."""
        return self._WAVE_HEIGHTS[self._sea_state]

    # -- Depth & pressure ----------------------------------------------------

    def pressure_at_depth(self, depth_metres: float) -> float:
        """
        Return absolute pressure in atm at a given depth.
        Formula: P = 1 atm (surface) + (depth * seawater density * g) / 101325
        Approximation: ~1 atm per 10 metres of seawater.
        """
        if depth_metres < 0:
            raise ValueError("Depth cannot be negative.")
        return 1.0 + (depth_metres / 10.0)

    def is_safe_for_diving(self, depth_metres: float) -> tuple[bool, str]:
        """
        Check whether diving to depth_metres is safe given current conditions.
        Returns (is_safe: bool, reason: str).
        """
        if depth_metres > self.max_depth:
            return False, f"Depth {depth_metres}m exceeds zone max_depth {self.max_depth}m."
        pressure = self.pressure_at_depth(depth_metres)
        if pressure > 51.0:  # ~500 m
            return False, f"Pressure {pressure:.1f} atm exceeds safe operational limit."
        if self.visibility < 2.0:
            return False, f"Visibility {self.visibility}m is critically low for safe diving."
        return True, "Conditions safe for diving."

    # -- Runtime updates -----------------------------------------------------

    def update_conditions(self, **kwargs) -> None:
        """
        Update any property at runtime.
        Example: ocean.update_conditions(temperature=15.0, sea_state=4)
        """
        for key, value in kwargs.items():
            if not hasattr(self, key) and key != "sea_state":
                raise AttributeError(f"OceanEnvironment has no attribute '{key}'.")
            setattr(self, key, value)

    # -- Representation ------------------------------------------------------

    def get_status(self) -> dict:
        return {
            "salinity_psu": self.salinity,
            "water_pressure_atm": self.water_pressure,
            "temperature_c": self.temperature,
            "visibility_m": self.visibility,
            "current_speed_knots": self.current_speed,
            "current_direction": self.current_direction,
            "sea_state": self._sea_state,
            "sea_state_description": self.sea_state_description,
            "wave_height_m": self.wave_height,
            "max_depth_m": self.max_depth,
        }

    def __str__(self) -> str:
        return (f"OceanEnvironment(sea_state={self._sea_state} '{self.sea_state_description}', "
                f"temp={self.temperature}°C, visibility={self.visibility}m, "
                f"current={self.current_speed}kn {self.current_direction})")

    def __repr__(self) -> str:
        return self.__str__()


# ---------------------------------------------------------------------------
# AerialEnvironment — Intern 2 owns this class
# ---------------------------------------------------------------------------

class AerialEnvironment:
    """
    Models the aerial environment for aerial vehicles.

    Intern 2 usage:
        sky = AerialEnvironment()
        super().__init__(..., environment=sky)

        # In takeoff():
        safe, reason = self.environment.is_safe_to_fly()
        if not safe:
            raise RuntimeError(f"Cannot take off: {reason}")

        # In navigate() at cruise altitude:
        temp = self.environment.temperature_at_altitude(self.altitude)
        if temp < -40:
            log_warning("Extreme cold at cruise altitude")
    """

    # Weather condition danger levels: 0=safe, 1=caution, 2=warning, 3=abort
    WEATHER_DANGER = {
        "clear":        0,
        "partly_cloudy": 0,
        "overcast":     1,
        "rain":         1,
        "drizzle":      1,
        "fog":          2,
        "snow":         2,
        "hail":         3,
        "storm":        3,
        "icing":        3,
    }

    # Standard atmosphere lapse rate: -6.5 °C per 1000 m
    LAPSE_RATE = 0.0065  # °C per metre

    def __init__(
        self,
        wind_speed: float = 0.0,
        wind_direction: str = "N",
        wind_gusts: float = 0.0,
        humidity: float = 0.5,
        temperature: float = 25.0,
        visibility: float = 10_000.0,
        weather: str = "clear",
        cloud_base: float = 1500.0,
        icing_risk: bool = False,
    ) -> None:
        self.wind_speed: float = wind_speed          # km/h
        self.wind_direction: str = wind_direction    # compass string
        self.wind_gusts: float = wind_gusts          # km/h peak gusts
        self.humidity: float = max(0.0, min(1.0, humidity))
        self.temperature: float = temperature        # °C at ground level
        self.visibility: float = visibility          # metres
        self._weather: str = weather
        self.cloud_base: float = cloud_base          # metres AGL
        self.icing_risk: bool = icing_risk

    # -- Weather -------------------------------------------------------------

    @property
    def weather(self) -> str:
        return self._weather

    @weather.setter
    def weather(self, value: str) -> None:
        if value not in self.WEATHER_DANGER:
            raise ValueError(
                f"Unknown weather condition '{value}'. "
                f"Valid options: {list(self.WEATHER_DANGER.keys())}"
            )
        self._weather = value

    @property
    def weather_danger_level(self) -> int:
        """0=safe, 1=caution, 2=warning, 3=abort."""
        return self.WEATHER_DANGER.get(self._weather, 0)

    # -- Safety checks -------------------------------------------------------

    def is_safe_to_fly(self) -> tuple[bool, str]:
        """
        Check whether current conditions are safe for flight.
        Returns (is_safe: bool, reason: str).
        Checks wind, visibility, weather danger level, and icing.
        """
        if self.wind_gusts > 120:
            return False, f"Gusts {self.wind_gusts} km/h exceed safe limit (120 km/h)."
        if self.wind_speed > 90:
            return False, f"Wind speed {self.wind_speed} km/h exceeds safe limit (90 km/h)."
        if self.visibility < 500:
            return False, f"Visibility {self.visibility}m below minimum (500m)."
        if self.weather_danger_level == 3:
            return False, f"Weather condition '{self._weather}' is abort-level dangerous."
        if self.icing_risk:
            return False, "Active icing risk — flight not safe."
        if self.weather_danger_level == 2:
            return True, f"Caution: weather condition '{self._weather}' — proceed carefully."
        return True, "Conditions safe for flight."

    def temperature_at_altitude(self, altitude_metres: float) -> float:
        """
        Return estimated air temperature (°C) at a given altitude using
        the International Standard Atmosphere lapse rate (-6.5°C / 1000m).
        """
        if altitude_metres < 0:
            raise ValueError("Altitude cannot be negative.")
        return self.temperature - (self.LAPSE_RATE * altitude_metres)

    # -- Runtime updates 

    def update_conditions(self, **kwargs) -> None:
        """
        Update any property at runtime.
        Example: sky.update_conditions(wind_speed=45.0, weather='rain')
        """
        for key, value in kwargs.items():
            if not hasattr(self, key) and key != "weather":
                raise AttributeError(f"AerialEnvironment has no attribute '{key}'.")
            setattr(self, key, value)

    # Representation 
    def get_status(self) -> dict:
        return {
            "wind_speed_kmh": self.wind_speed,
            "wind_direction": self.wind_direction,
            "wind_gusts_kmh": self.wind_gusts,
            "humidity": self.humidity,
            "temperature_c": self.temperature,
            "visibility_m": self.visibility,
            "weather": self._weather,
            "weather_danger_level": self.weather_danger_level,
            "cloud_base_m_agl": self.cloud_base,
            "icing_risk": self.icing_risk,
        }

    def __str__(self) -> str:
        return (f"AerialEnvironment(weather='{self._weather}', "
                f"wind={self.wind_speed}km/h {self.wind_direction}, "
                f"visibility={self.visibility}m, temp={self.temperature}°C)")

    def __repr__(self) -> str:
        return self.__str__()


# Quick smoke-test 

if __name__ == "__main__":
    print("=== OceanEnvironment ===")
    ocean = OceanEnvironment(sea_state=3, temperature=18.0, visibility=40.0)
    print(ocean)
    print("Status:", ocean.get_status())
    print("Pressure at 100m:", ocean.pressure_at_depth(100), "atm")
    print("Safe to dive 80m?", ocean.is_safe_for_diving(80))
    print("Safe to dive 600m?", ocean.is_safe_for_diving(600))

    print("\n=== AerialEnvironment ===")
    sky = AerialEnvironment(wind_speed=30.0, wind_gusts=55.0, weather="rain", visibility=2500.0)
    print(sky)
    print("Safe to fly?", sky.is_safe_to_fly())
    print("Temp at 3000m:", sky.temperature_at_altitude(3000), "°C")
    sky.update_conditions(weather="storm")
    print("After storm update — safe to fly?", sky.is_safe_to_fly())

    print("\n=== ObstacleManager ===")
    mgr = ObstacleManager()
    mgr.add_obstacle(18.975, 72.826, "Sunken Wreck", obstacle_type="wreck",
                     severity="high", radius_metres=200)
    mgr.add_obstacle(18.980, 72.830, "Coral Reef", obstacle_type="reef",
                     severity="medium", radius_metres=500)
    print(mgr)
    hits = mgr.check_position(18.975, 72.826)
    print("Obstacles at (18.975, 72.826):", [str(o) for o in hits])
    nearby = mgr.get_nearby_obstacles(18.977, 72.828, radius_metres=1000)
    print("Obstacles within 1km:", [o.name for o in nearby])