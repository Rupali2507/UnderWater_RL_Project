import torch
import numpy as np

class SurfaceOceanEnvironment:
    """
    Orchestrator for the simulation.
    Handles geographic collision masks and orchestrates the physics/sensor steps
    using the global state_tensor.
    """
    def __init__(self, device, location="Open Ocean", map_size=5000.0, resolution=10.0):
        self.device = device
        self.location = location
        self.map_size = map_size
        self.resolution = resolution
        
        # Grid parameters
        self.grid_size = int(map_size / resolution)
        self.landmask = torch.zeros((self.grid_size, self.grid_size), device=self.device)
        
        # Build geography (Default to Open Ocean to simplify training)
        self._build_geography()

    def _build_geography(self):
        """Generates the landmask. Keeps geography logic isolated."""
        if self.location == "Strait of Hormuz":
            g = self.grid_size
            xs = torch.arange(g, device=self.device).float()
            ys = torch.arange(g, device=self.device).float()
            X, Y = torch.meshgrid(xs, ys, indexing="ij")
            self.landmask = ((Y > g * 0.7 + torch.sin(X / 50.0) * 20.0) | 
                             ((Y < g * 0.3 - torch.cos(X / 40.0) * 30.0) & (X < g * 0.6))).float()
        # Additional maps can be added here or via a dedicated TerrainManager
        
    def get_collision_mask(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Vectorized collision check.
        Returns a boolean tensor where True = Collision with land.
        Positions: [N, 2] (x, y) in meters.
        """
        grid_coords = (positions / self.resolution).long()
        # Clamp to bounds to prevent index errors
        grid_coords = torch.clamp(grid_coords, 0, self.grid_size - 1)
        
        # Lookup landmask status
        return self.landmask[grid_coords[:, 0], grid_coords[:, 1]] == 1.0

    def step(self, state_tensor, actions, dt):
        """
        Orchestrates the physics step.
        1. Physics Engine processes movement (taking landmask into account).
        2. SensorSuite processes detections.
        3. CommunicationModule processes messages.
        """
        # Example of how to use the collision mask in the physics step:
        # potential_pos = current_pos + (velocity * dt)
        # collisions = self.get_collision_mask(potential_pos)
        # current_pos[~collisions] = potential_pos[~collisions]
        
        # After physics, we would trigger:
        # self.physics_engine.update(state_tensor, actions, dt)
        # self.sensor_suite.update_all(state_tensor)
        pass

    def reset_map(self, new_location=None):
        """Allows switching geographic scenarios between episodes."""
        if new_location:
            self.location = new_location
            self._build_geography()