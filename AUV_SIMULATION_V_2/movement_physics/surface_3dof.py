# pyrefly: ignore [missing-import]
import torch
import math

def apply_surface_kinematics(global_state: torch.Tensor, actions: torch.Tensor, indices: torch.Tensor, dt: float):
    MAX_SPEED = 25.0
    DRAG_COEFFICIENT = 0.05
    THRUST_FORCE = 50.0

    if indices.numel() == 0:
        return

    # Views into the shared state tensor
    ori = global_state[indices, 3:6]   # [roll, pitch, yaw]
    vel = global_state[indices, 6:9]   # [u (surge), v (sway), w (heave)]
    ang = global_state[indices, 9:12]  # [p, q, r (yaw rate)]

    # ── 1. SURGE PHYSICS ──────────────────────────────────────────────────────
    thrust = actions[indices, 0] * THRUST_FORCE
    drag   = -DRAG_COEFFICIENT * vel[:, 0]
    vel[:, 0] += (thrust + drag) * dt
    vel[:, 0]  = torch.clamp(vel[:, 0], -MAX_SPEED, MAX_SPEED)

    # ── 2. YAW RATE PHYSICS ───────────────────────────────────────────────────
    # Rudder deflection [-1, 1] creates a yaw torque.
    # Effectiveness is speed-gated (rudder does nothing at zero speed).
    rudder_torque = actions[indices, 1] * 2.0
    speed_factor  = torch.clamp(torch.abs(vel[:, 0]) / MAX_SPEED, 0.0, 1.0)

    ang[:, 2] += (rudder_torque * speed_factor) * dt
    ang[:, 2] *= 0.92   # Angular damping (water resistance)

    # ── 3. INTEGRATE YAW (once, cleanly) ──────────────────────────────────────
    # BUG FIX: removed duplicate `ori += ang * dt` which was double-integrating
    # yaw and also incorrectly updating roll/pitch for surface vessels.
    ori[:, 2] += ang[:, 2] * dt

    # Wrap yaw to [-π, π] to prevent unbounded growth
    ori[:, 2] = (ori[:, 2] + math.pi) % (2 * math.pi) - math.pi

    # ── 4. GLOBAL POSITION UPDATE ─────────────────────────────────────────────
    yaw   = ori[:, 2]
    pitch = ori[:, 1]   # Should stay ~0 for surface vessels

    dx = (vel[:, 0] * torch.cos(yaw) * torch.cos(pitch) - vel[:, 1] * torch.sin(yaw)) * dt
    dy = (vel[:, 0] * torch.sin(yaw) * torch.cos(pitch) + vel[:, 1] * torch.cos(yaw)) * dt
    dz = (vel[:, 0] * -torch.sin(pitch)) * dt

    # ── 5. WRITE BACK ─────────────────────────────────────────────────────────
    global_state[indices, 0] += dx
    global_state[indices, 1] += dy
    global_state[indices, 2] += dz
    global_state[indices, 3:6]  = ori
    global_state[indices, 6:9]  = vel
    global_state[indices, 9:12] = ang