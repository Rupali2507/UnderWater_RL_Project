import torch
from autonomous_vehicles.base import AutonomousVehicle
from movement_physics.surface_3dof import apply_surface_kinematics
from movement_physics.auv_6dof import apply_underwater_kinematics

class PhysicsEngine:
    """
    Vectorized PyTorch physics integrator using shared memory buffers.
    """
    SPEED_MULTIPLIER = 5.0
    def __init__(self, dt: float, device: torch.device):
        self.dt = dt
        self.device = device
        # Pre-allocated buffers to avoid memory fragmentation
        self.surf_idx = None
        self.sub_idx = None

    def refresh_registry(self, agent_registry: dict):
        """
        Call this only when the fleet changes (ships destroyed/spawned).
        Updates the index buffers for the GPU.
        """
        surf = [v.agent_index for v in agent_registry.values() if not hasattr(v, 'max_depth')]
        sub = [v.agent_index for v in agent_registry.values() if hasattr(v, 'max_depth')]
        
        self.surf_idx = torch.tensor(surf, dtype=torch.long, device=self.device)
        self.sub_idx = torch.tensor(sub, dtype=torch.long, device=self.device)

    def step(self, global_state: torch.Tensor, actions: torch.Tensor):
        """
        Executes physics step for the entire fleet in parallel.
        """
        # 1. APPLY BATCHED PHYSICS
        if self.surf_idx is not None and self.surf_idx.numel() > 0:
            apply_surface_kinematics(global_state, actions, self.surf_idx, self.dt)
        
        # if self.sub_idx is not None and self.sub_idx.numel() > 0:
        #     apply_underwater_kinematics(global_state, actions, self.sub_idx, self.dt)
        
        # 2. Vectorized Environment Constraints
        # Clamp Z for surface/underwater logic (0 is water surface)
        global_state[:, 2] = torch.clamp(global_state[:, 2], max=0.0)
        
        # 3. Batched Fuel Consumption
        # Using COL_FUEL from AutonomousVehicle base
        fuel_consumption = torch.abs(actions[:, 0]) * 0.1 * self.dt
        global_state[:, AutonomousVehicle.COL_FUEL] -= fuel_consumption
        global_state[:, AutonomousVehicle.COL_FUEL].clamp_(min=0.0)