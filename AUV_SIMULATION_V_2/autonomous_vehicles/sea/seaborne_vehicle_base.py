import abc
import torch
import numpy as np
# Assuming base.py is in the parent directory
from autonomous_vehicles.base import AutonomousVehicle


class SeaBorneVehicle(AutonomousVehicle, abc.ABC):
    """
    Abstract class representing vehicles that operate in a maritime environment.
    Provides common maritime-specific properties shared by both surface vessels
    and underwater vehicles.
    """

    # Tensor columns reserved for the static maritime constants below.
    # Shifted to indices 16, 17, and 18 to safely sit after the 16-column 
    # (0-15) 6-DOF kinematic state defined in base.py.
    COL_DISPLACEMENT = 16
    COL_DRAG = 17
    COL_TURN_RADIUS = 18

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
        **kwargs
    ):
        # Initialize the master AutonomousVehicle contract.
        # **kwargs forwards anything a concrete vessel wants to override --
        # dest_position, collision_radius, comm_range, initial_health,
        # initial_fuel, min_operational_fuel -- without SeaBorneVehicle
        # needing to know about every base-level param individually.
        super().__init__(
            vehicle_id=vehicle_id,
            team=team,
            agent_index=agent_index,
            state_tensor=state_tensor,
            start_position=start_position,
            **kwargs
        )

        # --- Validation ---
        # Cheap insurance against a typo'd config value silently producing
        # garbage physics downstream.
        assert displacement > 0, "displacement must be positive"
        assert water_drag_coefficient >= 0, "water_drag_coefficient cannot be negative"
        assert turn_radius >= 0, "turn_radius cannot be negative"

        # --- Maritime Physical Attributes ---
        # Protected attributes as defined in the blueprint. These are static
        # constants (they don't change mid-episode the way health/fuel do),
        # used by the PyTorch engine to calculate batched drag/inertia.
        self._displacement = displacement
        self._water_drag_coefficient = water_drag_coefficient
        self._turn_radius = turn_radius

        # Push the constants into the tensor once at construction so
        # engine.py can pull a whole column across all agents in one
        # vectorized op, instead of looping through Python objects.
        self._push_maritime_constants_to_tensor()

    # ==========================================
    # Read-Only Properties for Physics Engine
    # ==========================================
    @property
    def displacement(self) -> float:
        """Total mass of water displaced by the vehicle."""
        return self._displacement

    @property
    def water_drag_coefficient(self) -> float:
        """Resistance imposed by water on the vehicle."""
        return self._water_drag_coefficient

    @property
    def turn_radius(self) -> float:
        """Minimum turning radius of the vehicle."""
        return self._turn_radius

    # ==========================================
    # MARITIME METHODS (Public Abstract)
    # ==========================================
    @abc.abstractmethod
    def get_depth_range(self) -> tuple[float, float]:
        """
        Returns the valid operating depth range supported by the maritime vehicle.
        Format: (min_depth, max_depth)
        """
        pass

    @abc.abstractmethod
    def _apply_ocean_physics(self, action: np.ndarray | torch.Tensor):
        """
        Applies maritime-specific physical effects during a simulation step.
        *Architectural Note:* Because the PyTorch `physics/engine.py` handles the
        actual batched math, this method will serve as a bridge, passing this vehicle's
        specific drag/displacement constants into the tensor calculations.
        """
        pass

    # ==========================================
    # CORE METHODS (Overrides)
    # ==========================================
    def reset(self):
        """
        Restores base vehicle state, then re-writes the maritime constants
        into the tensor. Needed because many vectorized RL environments
        zero/reallocate the full state tensor between episodes -- without
        this, displacement/drag/turn_radius would silently vanish from the
        tensor on episode 2 even though they never logically changed.
        """
        super().reset()
        self._push_maritime_constants_to_tensor()

    # ==========================================
    # HELPER METHODS (Protected)
    # ==========================================
    def _push_maritime_constants_to_tensor(self):
        """Writes the static maritime constants into their reserved tensor columns."""
        # Ensure the tensor is actually wide enough to hold these constants (needs 19+ columns)
        if self.state_tensor is not None and self.state_tensor.shape[1] > self.COL_TURN_RADIUS:
            self.state_tensor[self.agent_index, self.COL_DISPLACEMENT] = self._displacement
            self.state_tensor[self.agent_index, self.COL_DRAG] = self._water_drag_coefficient
            self.state_tensor[self.agent_index, self.COL_TURN_RADIUS] = self._turn_radius

    def _is_depth_valid(self, depth: float) -> bool:
        """Checks a candidate depth against this vehicle's operating range."""
        min_depth, max_depth = self.get_depth_range()
        return min_depth <= depth <= max_depth