import torch

def apply_underwater_kinematics(global_state: torch.Tensor, actions: torch.Tensor, indices: torch.Tensor, dt: float):
    """
    Applies 6-DOF kinematics for Submarines.
    actions: [N, 3] -> [throttle, rudder, elevator_pitch]
    """
    if indices.numel() == 0:
        return

    # Extract state slices (Roll:3, Pitch:4, Yaw:5 | u:6, v:7, w:8 | p:9, q:10, r:11)
    ori = global_state[indices, 3:6]
    vel = global_state[indices, 6:9]
    ang = global_state[indices, 9:12]

    # 1. Update Linear Dynamics
    vel[:, 0] += actions[indices, 0] * dt # Surge (u)
    vel[:, 2] += actions[indices, 2] * dt # Heave (w) - Controlled by elevator_pitch

    # 2. Update Angular Dynamics
    ang[:, 2] = actions[indices, 1] * 0.5 # Yaw (r)
    ang[:, 1] = actions[indices, 2] * 0.2 # Pitch (q)

    # 3. Integrate Orientation (Euler angles)
    ori += ang * dt

    # 4. Global Rotation (3D Transformation)
    # Using full rotation matrix components for 6-DOF
    yaw, pitch, roll = ori[:, 2], ori[:, 1], ori[:, 0]
    
    # Calculate global displacement vector
    dx = (vel[:, 0] * torch.cos(yaw) * torch.cos(pitch)) * dt
    dy = (vel[:, 0] * torch.sin(yaw) * torch.cos(pitch)) * dt
    dz = (vel[:, 0] * -torch.sin(pitch) + vel[:, 2] * torch.cos(pitch)) * dt

    # 5. Write back to State Tensor
    global_state[indices, 0:3] += torch.stack([dx, dy, dz], dim=1)
    global_state[indices, 3:6] = ori
    global_state[indices, 6:9] = vel
    global_state[indices, 9:12] = ang