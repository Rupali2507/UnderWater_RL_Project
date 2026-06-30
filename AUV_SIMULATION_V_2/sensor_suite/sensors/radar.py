import torch
import time
import math

from ..base import Sensor, Detection

class Radar(Sensor):
    """
    Vectorized electromagnetic detection system for surface and aerial contacts.
    """
    def __init__(
        self, 
        radar_range: float, 
        field_of_view: float = 360.0,
        update_frequency: float = 1.0, 
        noise_level: float = 0.02
    ):
        super().__init__(max_range=radar_range, update_frequency=update_frequency, noise_level=noise_level)
        self.radar_range = radar_range
        self.fov_rad = math.radians(field_of_view)

    def scan(self, vehicle, state_tensor: torch.Tensor) -> list[Detection]:
        """
        Vectorized scan using the master state_tensor instead of iterating through vehicle objects.
        """
        detections = []
        my_idx = vehicle.agent_index
        
        # 1. Slice current vehicle state and all others
        my_pos = state_tensor[my_idx, vehicle.COL_POSITION].unsqueeze(0) # Shape [1, 3]
        all_pos = state_tensor[:, vehicle.COL_POSITION]                   # Shape [N, 3]
        
        # 2. Vectorized Distance Check
        distances = torch.cdist(my_pos, all_pos).squeeze(0) # Shape [N]
        in_range_mask = (distances <= self.radar_range)
        
        # 3. Tactical Rules (Mask out self and underwater targets)
        in_range_mask[my_idx] = False
        underwater_mask = (state_tensor[:, 2] < -2.0)
        in_range_mask &= ~underwater_mask
        
        # 4. Vectorized FOV Check
        delta = all_pos - my_pos # Shape [N, 3]
        angles = torch.atan2(delta[:, 1], delta[:, 0])
        my_yaw = state_tensor[my_idx, 5] # Using yaw index
        
        rel_angles = (angles - my_yaw + math.pi) % (2 * math.pi) - math.pi
        fov_mask = (torch.abs(rel_angles) <= (self.fov_rad / 2.0))
        
        # Combine masks
        final_mask = in_range_mask & fov_mask
        target_indices = torch.nonzero(final_mask).squeeze(1)
        
        # 5. Build Detections
        for idx in target_indices:
            idx = idx.item()
            dist = distances[idx].item()
            
            # Confidence logic
            confidence = max(0.2, 1.0 - (dist / self.radar_range) - self._noise_level)
            
            detections.append(Detection(
                object_id=f"agent_{idx}", # Or mapping to vehicle_id
                position=tuple(state_tensor[idx, vehicle.COL_POSITION].tolist()),
                confidence=confidence,
                source_sensor="RADAR",
                timestamp=time.time()
            ))
                    
        return detections