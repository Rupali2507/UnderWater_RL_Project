import torch
import time

from ..base import Sensor, Detection

class Sonar(Sensor):
    """
    Acoustic detection system for underwater and surface contacts.
    Passive mode: Listens for noise.
    Active mode: Emits a pulse (more accurate, but reveals position).
    """
    def __init__(
        self, 
        sonar_range: float, 
        update_frequency: float = 1.0, 
        noise_level: float = 0.05
    ):
        super().__init__(max_range=sonar_range, update_frequency=update_frequency, noise_level=noise_level)
        self.sonar_range = sonar_range

    def scan(self, vehicle, state_tensor: torch.Tensor, mode: str = 'passive') -> list[Detection]:
        """
        Vectorized sonar scan.
        Passive mode: Detects based on source noise level.
        Active mode: Detects based on sonar range and object reflectivity.
        """
        detections = []
        my_idx = vehicle.agent_index
        my_pos = state_tensor[my_idx, vehicle.COL_POSITION].unsqueeze(0)
        all_pos = state_tensor[:, vehicle.COL_POSITION]
        
        # 1. Vectorized Distance Check
        distances = torch.cdist(my_pos, all_pos).squeeze(0)
        
        # 2. Sonar Logic (Depth Dependency)
        # Sound travels differently based on water density/depth
        if mode == 'passive':
            # Passive: Limited by range and target noise
            in_range_mask = (distances <= self.sonar_range)
        else:
            # Active: Doubled range, but high noise sensitivity
            in_range_mask = (distances <= (self.sonar_range * 2))
            
        in_range_mask[my_idx] = False
        
        target_indices = torch.nonzero(in_range_mask).squeeze(1)
        
        for idx in target_indices:
            idx = idx.item()
            dist = distances[idx].item()
            
            # Sonar confidence increases with proximity and decreases with depth/noise
            confidence = max(0.1, 1.0 - (dist / self.sonar_range) - self._noise_level)
            
            detections.append(Detection(
                object_id=f"agent_{idx}",
                position=tuple(state_tensor[idx, vehicle.COL_POSITION].tolist()),
                confidence=confidence,
                source_sensor=f"SONAR_{mode.upper()}",
                timestamp=time.time()
            ))
                    
        return detections