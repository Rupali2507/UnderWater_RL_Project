"""
Contains the class implementations for sea-borne autonomous vehicles 
"""

from abc import abstractmethod
from typing import List, Dict, Any, Optional
from autonomous_vehicles.base import AutonomousVehicle
from autonomous_vehicles.environment import OceanEnvironment


class SeaBorneVehicle(AutonomousVehicle):
    """
    Abstract intermediary class representing all vehicles operating in water.
    Extends AutonomousVehicle
    """
    def __init__(
        self, 
        vehicle_id: str, 
        name: str, 
        max_speed: float, 
        hull_type: str, 
        displacement: float, 
        environment: OceanEnvironment,
        sonar_enabled: bool = False,
        max_depth: float = 0.0
    ) -> None:
       
        super().__init__(
            vehicle_id=vehicle_id, 
            name=name, 
            max_speed=max_speed, 
            environment=environment
        )
        self.hull_type: str = hull_type  # 'monohull', 'catamaran', 'submarine_hull'
        self.displacement: float = displacement  # weight of water displaced in tonnes
        self.sonar_enabled: bool = sonar_enabled  # boolean flag 
        self.current_depth: float = 0.0  # depth in metres (0 = surface) 
        self.max_depth: float = max_depth  # maximum safe depth in metres

    def navigate(self, destination: str) -> str:
       
        # Call safe diving environment check prior to planning navigation depth routes 
        if self.current_depth > 0 and not self.environment.is_safe_for_diving(self.current_depth):
            raise RuntimeError(f"Navigation aborted: Depth conditions unsafe at {self.current_depth}m!")
            
        log_msg = f"Vessel '{self.name}' routing a naval path toward {destination}."
        print(log_msg)
        return log_msg

    def check_sonar(self) -> List[Any]:
        
        if not self.sonar_enabled:
            print(f"[{self.name}] Sonar system is currently disabled.")
            return []
        print(f"[{self.name}] Emitting sonar ping. Scanning underwater surroundings...")
        return self.obstacles_detected


class SurfaceVehicle(SeaBorneVehicle):
    """
    Concrete class representing vehicles restricted to the water surface.
    Extends SeaBorneVehicle
    """
    def __init__(
        self, 
        vehicle_id: str, 
        name: str, 
        max_speed: float, 
        hull_type: str, 
        displacement: float, 
        environment: OceanEnvironment,
        wake_pattern: str, 
        draft_depth: float, 
        deck_area: float,
        sonar_enabled: bool = False
    ) -> None:
        super().__init__(
            vehicle_id=vehicle_id, 
            name=name, 
            max_speed=max_speed, 
            hull_type=hull_type, 
            displacement=displacement, 
            environment=environment,
            sonar_enabled=sonar_enabled,
            max_depth=0.0  # Surface vessels cannot plunge below surface depth 
        )
        self.wake_pattern: str = wake_pattern  # 'displacement', 'planing' 
        self.draft_depth: float = draft_depth  # depth of hull below waterline in metres 
        self.deck_area: float = deck_area  # usable deck area in square metres 

    def get_vehicle_type(self) -> str:
        """Contract requirement from base class."""
        return "Surface Vehicle"

    def moor(self, port: str) -> bool:
        """Docks the vessel at a given port if sea conditions are safe[cite: 52]."""
        # Surface vehicles read sea_state to decide whether mooring is safe 
        if self.environment.sea_state > 6:
            print(f"[{self.name}] Mooring aborted! Sea state ({self.environment.sea_state}) is too rough.")
            return False
            
        print(f"[{self.name}] Successfully moored and stabilized at {port}.")
        self.stop()  # Halt movement when docked 
        return True

    def adjust_ballast(self, amount: float) -> None:
        """Modifies water ballast parameters to balance vessel stability[cite: 53]."""
        print(f"[{self.name}] Readjusting ballast systems by {amount} units to adapt to sea state.")


class UnderwaterVehicle(SeaBorneVehicle):
    """
    Concrete class representing submersibles operating deep beneath the ocean.
    Extends SeaBorneVehicle[cite: 54].
    """
    def __init__(
        self, 
        vehicle_id: str, 
        name: str, 
        max_speed: float, 
        hull_type: str, 
        displacement: float, 
        environment: OceanEnvironment,
        dive_rate: float, 
        pressure_resistance: float, 
        oxygen_supply: float,
        max_depth: float,
        sonar_enabled: bool = True
    ) -> None:
        super().__init__(
            vehicle_id=vehicle_id, 
            name=name, 
            max_speed=max_speed, 
            hull_type=hull_type, 
            displacement=displacement, 
            environment=environment,
            sonar_enabled=sonar_enabled,
            max_depth=max_depth
        )
        self.dive_rate: float = dive_rate  # speed of descent (m/min) 
        self.pressure_resistance: float = pressure_resistance  # rated max pressure in atm 
        self.oxygen_supply: float = oxygen_supply  # remaining life support oxygen in hours 

    def get_vehicle_type(self) -> str:
        """Contract requirement from base class[cite: 32]."""
        return "Underwater Submersible"

    def dive(self, target_depth: float) -> bool:
        """Descends safely to a specified target depth[cite: 60]."""
        if target_depth > self.max_depth:
            print(f"[{self.name}] Threat: Target depth {target_depth}m exceeds structural structural threshold!")
            return False

        # Environmental safety validation check before descending 
        if not self.environment.is_safe_for_diving(target_depth):
            print(f"[{self.name}] Aborting dive sequence! Environment conditions unsafe at {target_depth}m.")
            return False

        print(f"[{self.name}] Initiating dive sequence. Descending to {target_depth}m at {self.dive_rate} m/min.")
        self.current_depth = target_depth
        return True

    def surface(self) -> None:
        """Ascends back to baseline sea level (depth = 0)[cite: 61]."""
        print(f"[{self.name}] Blowing ballast tanks. Returning to ocean surface level.")
        self.current_depth = 0.0

    def flood_ballast(self) -> None:
        """Increases negative buoyancy intentionally to speed up submergence[cite: 62]."""
        print(f"[{self.name}] Flooding ballast systems. Generating negative buoyancy vector.")


class CargoShip(SurfaceVehicle):
    """
    Leaf Class modeling large freight cargo transports.
    Extends SurfaceVehicle[cite: 63].
    """
    def __init__(
        self, 
        vehicle_id: str, 
        name: str, 
        max_speed: float, 
        displacement: float, 
        environment: OceanEnvironment,
        draft_depth: float, 
        deck_area: float,
        cargo_capacity: float, 
        num_cranes: int
    ) -> None:
        super().__init__(
            vehicle_id=vehicle_id, 
            name=name, 
            max_speed=max_speed, 
            hull_type='monohull', 
            displacement=displacement, 
            environment=environment,
            wake_pattern='displacement', 
            draft_depth=draft_depth, 
            deck_area=deck_area
        )
        self.cargo_capacity: float = cargo_capacity  # max cargo weight in tonnes 
        self.cargo_manifest: List[str] = []  # list of cargo items currently loaded 
        self.num_cranes: int = num_cranes  # number of loading cranes [cite: 67]

    def get_vehicle_type(self) -> str:
        return "Autonomous Commercial Cargo Ship"

    def load_cargo(self, item: str, weight: float) -> bool:
        """Adds a freight line item to the manifest if payload allows[cite: 69]."""
        # Basic check assuming weight is handled simplistically or mock verified
        print(f"[{self.name}] Operating onboard cranes ({self.num_cranes}) to load: '{item}' ({weight}t).")
        self.cargo_manifest.append(item)
        return True

    def unload_cargo(self, item: str) -> bool:
        """Removes a freight item from the vessel manifest[cite: 70]."""
        if item in self.cargo_manifest:
            self.cargo_manifest.remove(item)
            print(f"[{self.name}] Discharged item: '{item}' off the manifest registry.")
            return True
        print(f"[{self.name}] Item allocation search failed: '{item}' not on manifest.")
        return False


class Rig(SurfaceVehicle):
    """
    Leaf Class representing autonomous stationary offshore drilling installations.
    Extends SurfaceVehicle[cite: 71].
    """
    def __init__(
        self, 
        vehicle_id: str, 
        name: str, 
        displacement: float, 
        environment: OceanEnvironment,
        draft_depth: float, 
        deck_area: float,
        drill_depth: float
    ) -> None:
        # Rigs traditionally have zero operational standard cruising speed while deployed
        super().__init__(
            vehicle_id=vehicle_id, 
            name=name, 
            max_speed=0.0, 
            hull_type='catamaran',  # Semi-submersible or catamaran styling architecture
            displacement=displacement, 
            environment=environment,
            wake_pattern='displacement', 
            draft_depth=draft_depth, 
            deck_area=deck_area
        )
        self.drill_depth: float = drill_depth  # maximum drilling depth in metres 
        self.anchored: bool = False  # structural anchor locking flag 
        self.drilling_active: bool = False  # drilling lifecycle active flag 

    def get_vehicle_type(self) -> str:
        return "Autonomous Offshore Industrial Rig"

    def drop_anchor(self) -> None:
        """Locks the industrial platform in geo-stationary coordinates[cite: 77]."""
        self.anchored = True
        print(f"[{self.name}] Deploying heavy ocean seabed mooring lines. Anchor is SET.")

    def start_drilling(self) -> bool:
        """Begins marine drilling operations[cite: 78]."""
        if not self.anchored:
            print(f"[{self.name}] Refusing drill execution! Platform must be anchored first.")
            return False
        self.drilling_active = True
        print(f"[{self.name}] Rotating drill bits. Commencing downhole excavation up to {self.drill_depth}m.")
        return True

    def stop_drilling(self) -> None:
        """Halts the drilling assembly safely[cite: 79]."""
        self.drilling_active = False
        print(f"[{self.name}] Braking assembly. Industrial drill line operations paused safely.")