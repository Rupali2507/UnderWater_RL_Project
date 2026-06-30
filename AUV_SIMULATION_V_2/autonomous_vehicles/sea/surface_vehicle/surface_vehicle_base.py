import abc
import torch
import numpy as np

# Assuming seaborne_vehicle.py is in the same directory
from autonomous_vehicles.sea.seaborne_vehicle_base import SeaBorneVehicle

class SurfaceVehicle(SeaBorneVehicle, abc.ABC):
    """
    Abstract class representing maritime vehicles that operate on the water surface.
    Provides common surface-navigation, wave-interaction, and maritime-routing functionality.
    """

    # --- Surface Tensor Schema Extension ---
    # Expanding the tensor schema to allow the physics engine to quickly 
    # compute batched Line-of-Sight (LoS) and shallow-water grounding checks.
    COL_DRAFT = 19
    COL_VISUAL_RANGE = 20

    def __init__(
        self, 
        vehicle_id: str, 
        team: str, 
        agent_index: int, 
        state_tensor: torch.Tensor,
        start_position: tuple,
        displacement: float,
        water_drag_coefficient: float,
        turn_radius: float,
        draft: float,
        sea_state_tolerance: int,
        visual_range: float,
        **kwargs  # CRITICAL: Captures health, fuel, dest_position, etc.
    ):
        # Initialize the maritime contract and pass **kwargs up the chain
        super().__init__(
            vehicle_id=vehicle_id, 
            team=team, 
            agent_index=agent_index, 
            state_tensor=state_tensor,
            start_position=start_position,
            displacement=displacement,
            water_drag_coefficient=water_drag_coefficient,
            turn_radius=turn_radius,
            **kwargs
        )

        # --- Surface Physical Attributes ---
        self._draft = draft
        self._sea_state_tolerance = sea_state_tolerance
        
        # --- Surface Operational Attributes ---
        self._sea_lane_waypoints = []
        self._AIS_transponder_active = True
        self._nav_lights_on = True
        self._visual_range = visual_range

        # Push the new surface-specific constants to the tensor
        self._push_surface_constants_to_tensor()

    # ==========================================
    # Read-Only Properties
    # ==========================================
    @property
    def draft(self) -> float:
        """How deep the vessel's hull extends below the waterline."""
        return self._draft

    @property
    def sea_state_tolerance(self) -> int:
        """Maximum sea-state conditions under which the vessel can safely operate."""
        return self._sea_state_tolerance
        
    @property
    def visual_range(self) -> float:
        """Maximum visual observation distance (used for Zone A detection)."""
        return self._visual_range

    # ==========================================
    # State Management Properties
    # ==========================================
    @property
    def ais_active(self) -> bool:
        return self._AIS_transponder_active
        
    @ais_active.setter
    def ais_active(self, state: bool):
        self._AIS_transponder_active = state

    @property
    def nav_lights_active(self) -> bool:
        return self._nav_lights_on
        
    @nav_lights_active.setter
    def nav_lights_active(self, state: bool):
        self._nav_lights_on = state
        
    @property
    def waypoints(self) -> list[tuple]:
        return self._sea_lane_waypoints
        
    def add_waypoint(self, waypoint: tuple):
        self._sea_lane_waypoints.append(waypoint)
        
    def clear_waypoints(self):
        self._sea_lane_waypoints.clear()

    # ==========================================
    # OVERRIDDEN MARITIME METHODS
    # ==========================================
    def get_depth_range(self) -> tuple[float, float]:
        """
        Surface vessels strictly operate at z=0 (or slightly below depending on draft).
        Returns: (min_depth, max_depth)
        """
        return (-self.draft, 0.0)
    
    def reset(self):
        """Ensures surface constants aren't lost when the tensor is zeroed between episodes."""
        super().reset()
        self._push_surface_constants_to_tensor()

    # ==========================================
    # SURFACE METHODS
    # ==========================================
    def _apply_ocean_physics(self, action: np.ndarray | torch.Tensor):
        """Fulfills the SeaBorneVehicle contract and routes to surface physics."""
        self._apply_surface_physics(action)

    @abc.abstractmethod
    def _apply_surface_physics(self, action: np.ndarray | torch.Tensor):
        """
        Applies surface-specific maritime physics such as wave interaction 
        and sea-state effects.
        
        *Architectural Note:* In Phase 1, the PyTorch `physics/surface_3dof.py` 
        handles the actual math. This method serves to bridge specific wave/state 
        parameters into those batched calculations.
        """
        pass
        
    # ==========================================
    # HELPER METHODS (Protected)
    # ==========================================
    def _push_surface_constants_to_tensor(self):
        """Writes the static surface constants into their reserved tensor columns."""
        if self.state_tensor is not None and self.state_tensor.shape[1] > self.COL_VISUAL_RANGE:
            self.state_tensor[self.agent_index, self.COL_DRAFT] = self._draft
            self.state_tensor[self.agent_index, self.COL_VISUAL_RANGE] = self._visual_range