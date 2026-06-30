# pyrefly: ignore [missing-import]
import functools
import gymnasium as gym
import torch
import numpy as np
import math
from pettingzoo.utils.env import ParallelEnv

from environments.surface_ocean_environment import SurfaceOceanEnvironment
from movement_physics.base import PhysicsEngine
from autonomous_vehicles.sea.surface_vehicle.vehicles.naval_ship import NavalShip
from autonomous_vehicles.sea.surface_vehicle.vehicles.cargo_ship import CargoShip
from autonomous_vehicles.base import VehicleStatus
from sensor_suite.base import SensorSuite
from autonomous_vehicles.sea.surface_vehicle.vehicles.threat_ship import ThreatShip


# ZONE RADII  (metres)

ZONE_C_RADIUS   =  300.0   # Cargo formation radius
ZONE_B_RADIUS   =  700.0   # Naval patrol orbit radius around convoy centre
ZONE_A_RADIUS   = 2000.0   # Early-detection perimeter
DEST_THRESHOLD  =  500.0   # Cargo "arrived" distance
COMBAT_RANGE    =  150.0   # Threat close enough to damage cargo / naval
NEG_RANGE       =  400.0   # Range at which negotiation is attempted

NEG_SUCCESS_PROB = 0.4


class TacticalNavalEnv(ParallelEnv):
    metadata = {"render_modes": ["human"], "name": "tactical_naval_v5"}

    def __init__(self, scenario_config, render_mode=None):
        self.device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.render_mode = render_mode
        self.dt          = 0.1

        self.world   = SurfaceOceanEnvironment(
            device=self.device,
            location=scenario_config.get("location", "Open Ocean"))
        self.physics = PhysicsEngine(dt=self.dt, device=self.device)

        self.possible_agents     = list(scenario_config["agents"].keys())
        self.agent_name_mapping  = {n: i for i, n in enumerate(self.possible_agents)}
        self.global_state        = torch.zeros(
            (len(self.possible_agents), 21), device=self.device, dtype=torch.float32)

        self.vehicles = self._initialize_vehicles(scenario_config["agents"])
        self.physics.refresh_registry(self.vehicles)
        self.agents   = self.possible_agents[:]
        SensorSuite.set_vehicle_registry(self.vehicles)

        # Per-agent state machine flags
        self._negotiating   = {}   # escort_id → threat_id being negotiated
        self._neg_countdown = {}   # escort_id → steps remaining in negotiation
        self._alerted       = {}   # escort_id → threat position from a broadcast
        self._prev_dists    = {}

    
    def _initialize_vehicles(self, agent_configs):
        vehicles = {}
        for agent_id, cfg in agent_configs.items():
            idx       = self.agent_name_mapping[agent_id]
            params    = cfg.copy()
            start_pos = tuple(params.pop("start_pos", [0.0, 0.0]))
            dest_pos  = tuple(params.pop("dest_pos",  [5000.0, 5000.0]))
            class_type = params.pop("class_type")

            cls = {"NavalShip": NavalShip,
                   "CargoShip": CargoShip,
                   "ThreatShip": ThreatShip}[class_type]

            vehicles[agent_id] = cls(
                vehicle_id=agent_id, agent_index=idx,
                state_tensor=self.global_state,
                start_position=start_pos, dest_position=dest_pos, **params)

            if not hasattr(vehicles[agent_id], 'sensor_suite'):
                vehicles[agent_id].sensor_suite = SensorSuite()
        return vehicles

    # ──────────────────────────────────────────────────────────────────────────
    # HELPERS
    # ──────────────────────────────────────────────────────────────────────────
    def _is_threat(self, v): return isinstance(v, ThreatShip)
    def _is_escort(self, v): return isinstance(v, NavalShip) and not isinstance(v, ThreatShip)
    def _is_cargo(self, v):  return isinstance(v, CargoShip)

    def _pos(self, vehicle):
        idx = vehicle.agent_index
        return (self.global_state[idx, 0].item(),
                self.global_state[idx, 1].item())

    def _dist(self, pos_a, pos_b):
        return math.sqrt((pos_a[0]-pos_b[0])**2 + (pos_a[1]-pos_b[1])**2)

    def _convoy_centre(self, cargo_agents):
        xs = [self.global_state[self.vehicles[c].agent_index, 0].item() for c in cargo_agents]
        ys = [self.global_state[self.vehicles[c].agent_index, 1].item() for c in cargo_agents]
        return (float(np.mean(xs)), float(np.mean(ys)))

    def _orbit_point(self, centre, angle_rad, radius):
        """Point on Zone-B patrol circle."""
        return (centre[0] + math.cos(angle_rad) * radius,
                centre[1] + math.sin(angle_rad) * radius)

    
    # STEP
    
    def step(self, actions):
        # 1. Physics
        action_tensor = torch.zeros((len(self.possible_agents), 2), device=self.device)
        for agent, act in actions.items():
            act_2d = np.array(act, dtype=np.float32).flatten()[:2]
            action_tensor[self.agent_name_mapping[agent]] = torch.tensor(
                act_2d, device=self.device, dtype=torch.float32)
        self.physics.step(self.global_state, action_tensor)

        # 2. Categorise
        cargo_agents  = [a for a in self.agents if self._is_cargo(self.vehicles[a])]
        threat_agents = [a for a in self.agents if self._is_threat(self.vehicles[a])]
        escort_agents = [a for a in self.agents if self._is_escort(self.vehicles[a])]

        mission_win = mission_loss = False

        if cargo_agents:
            convoy_centre = self._convoy_centre(cargo_agents)

             
            for c in cargo_agents:
                cv = self.vehicles[c]
                if cv._get_distance_to(cv._nav_destination) < DEST_THRESHOLD:
                    mission_win = True

            
            for t in threat_agents:
                tv   = self.vehicles[t]
                t_pos = self._pos(tv)

                
                nearest_cargo, min_cd = None, float('inf')
                for c in cargo_agents:
                    d = self._dist(t_pos, self._pos(self.vehicles[c]))
                    if d < min_cd:
                        min_cd, nearest_cargo = d, self.vehicles[c]

                if nearest_cargo is None:
                    continue

                
                tv._dest_position = self._pos(nearest_cargo)

                
                if min_cd < COMBAT_RANGE:
                    mission_loss = True

            # ── ESCORT LOGIC ─────────────────────────────────────────────
            for i, e in enumerate(escort_agents):
                ev    = self.vehicles[e]
                e_pos = self._pos(ev)

                # Default: patrol orbit around convoy centre on Zone B circle
                # Each escort gets a unique angle slice
                n_escorts    = len(escort_agents)
                patrol_angle = (2 * math.pi * i / n_escorts) + \
                               (self.global_state[ev.agent_index, 5].item() * 0.01)  # slow drift
                orbit_pt     = self._orbit_point(convoy_centre, patrol_angle, ZONE_B_RADIUS)

                # Check if any threat is inside Zone A
                nearest_threat, min_td = None, float('inf')
                for t in threat_agents:
                    d = self._dist(e_pos, self._pos(self.vehicles[t]))
                    if d < min_td:
                        min_td, nearest_threat = d, self.vehicles[t]

                dist_threat_to_convoy = (self._dist(self._pos(nearest_threat), convoy_centre)
                                         if nearest_threat else float('inf'))

                # ── STATE MACHINE ─────────────────────────────────────────
                if nearest_threat and dist_threat_to_convoy < ZONE_A_RADIUS:
                    threat_pos = self._pos(nearest_threat)
                    threat_id  = id(nearest_threat)

                    if e in self._negotiating and self._negotiating[e] == threat_id:
                        # Already in negotiation countdown
                        self._neg_countdown[e] -= 1
                        if self._neg_countdown[e] <= 0:
                            # Negotiation resolves
                            if np.random.random() < NEG_SUCCESS_PROB:
                                # SUCCESS: threat retreats outside Zone A
                                dx = threat_pos[0] - convoy_centre[0]
                                dy = threat_pos[1] - convoy_centre[1]
                                d  = math.sqrt(dx**2 + dy**2) or 1.0
                                retreat_x = convoy_centre[0] + (dx/d) * (ZONE_A_RADIUS * 1.5)
                                retreat_y = convoy_centre[1] + (dy/d) * (ZONE_A_RADIUS * 1.5)
                                nearest_threat._dest_position = (retreat_x, retreat_y)
                                ev.status = VehicleStatus.PATROL
                            else:
                                # FAILED: combat — both take damage
                                ev.take_damage(15.0)
                                nearest_threat.take_damage(20.0)
                                ev.status = VehicleStatus.COMBAT
                            del self._negotiating[e]
                            del self._neg_countdown[e]

                    elif min_td < NEG_RANGE and e not in self._negotiating:
                        # Start negotiation: escort moves to intercept point
                        self._negotiating[e]   = threat_id
                        self._neg_countdown[e] = 30   # 3 seconds at dt=0.1
                        ev.status = VehicleStatus.NEGOTIATE
                        ev._dest_position = threat_pos

                    else:
                        # Intercept: move toward threat
                        ev._dest_position = threat_pos
                        ev.status = VehicleStatus.INTERCEPT

                elif e in self._alerted:
                    # Received broadcast: go to last known threat position
                    ev._dest_position = self._alerted[e]
                    ev.status = VehicleStatus.INTERCEPT
                    # Clear alert once close enough
                    if self._dist(e_pos, self._alerted[e]) < 200:
                        del self._alerted[e]

                else:
                    # Default: orbit patrol
                    ev._dest_position = orbit_pt
                    if ev.status not in (VehicleStatus.COMBAT,):
                        ev.status = VehicleStatus.PATROL

                # ── ESCORT DEATH BROADCAST ────────────────────────────────
                if ev.health <= 0 and nearest_threat:
                    threat_pos = self._pos(nearest_threat)
                    for other_e in escort_agents:
                        if other_e != e and other_e not in self._alerted:
                            self._alerted[other_e] = threat_pos

        # 3. OOB check
        OOB = 20000.0
        is_oob = any(
            math.sqrt(self.global_state[self.vehicles[a].agent_index, 0].item()**2 +
                      self.global_state[self.vehicles[a].agent_index, 1].item()**2) > OOB
            for a in self.agents)

        is_done = mission_win or mission_loss

        # 4. Observations
        observations = {}
        for agent in self.agents:
            v = self.vehicles[agent]
            v.sensor_suite.scan(v, self.global_state)
            observations[agent] = v.get_observation()

        # 5. Rewards
        rewards = self._compute_rewards(mission_win, mission_loss, is_oob)

        terminations = {a: is_done for a in self.agents}
        truncations  = {a: is_oob  for a in self.agents}
        infos        = {a: {"mission_win": mission_win, "mission_loss": mission_loss}
                        for a in self.agents}

        return observations, rewards, terminations, truncations, infos

    
    # REWARDS
    def _compute_rewards(self, mission_win=False, mission_loss=False, is_oob=False):
        rewards = {}
        for agent in self.agents:
            v = self.vehicles[agent]

            curr_dist = v._get_distance_to(v._dest_position)
            prev_dist = self._prev_dists.get(agent, curr_dist)
            progress  = (prev_dist - curr_dist) / 100.0
            self._prev_dists[agent] = curr_dist

            r = progress - 0.001   # dense progress + tiny time penalty

            if self._is_cargo(v):
                r += 0.005   # small survival bonus per step

            elif self._is_escort(v):
                # Penalise straying far outside Zone B
                cargo_key = next((k for k in self.vehicles if self._is_cargo(self.vehicles[k])), None)
                if cargo_key:
                    dist_to_cargo = v._get_distance_to(self._pos(self.vehicles[cargo_key]))
                    excess = max(0.0, dist_to_cargo - ZONE_B_RADIUS * 1.5)
                    r -= 0.00002 * excess
                # Small bonus for being in negotiation/intercept (doing its job)
                if v.status in (VehicleStatus.NEGOTIATE, VehicleStatus.INTERCEPT):
                    r += 0.01

            if is_oob:
                r -= 5.0

            if mission_win:
                r += 10.0 if self._is_cargo(v) or self._is_escort(v) else -10.0
            elif mission_loss:
                r += 10.0 if self._is_threat(v) else -10.0

            if v.health <= 0:
                r -= 0.5

            rewards[agent] = r
        return rewards

    # ──────────────────────────────────────────────────────────────────────────
    # RESET
    # ──────────────────────────────────────────────────────────────────────────
    def reset(self, seed=None, options=None):
        self.agents         = self.possible_agents[:]
        self._negotiating   = {}
        self._neg_countdown = {}
        self._alerted       = {}
        self._prev_dists    = {}
        self.global_state.zero_()

        for agent in self.agents:
            v   = self.vehicles[agent]
            idx = v.agent_index

            self.global_state[idx, 0] = v._start_position[0]
            self.global_state[idx, 1] = v._start_position[1]

            v._dest_position = v._nav_destination

            dx  = v._nav_destination[0] - v._start_position[0]
            dy  = v._nav_destination[1] - v._start_position[1]
            yaw = math.atan2(dy, dx)
            self.global_state[idx, 5] = yaw

            if hasattr(v, '_health'):
                v._health = getattr(v, 'initial_health', 100.0)
            if hasattr(v, 'COL_FUEL'):
                self.global_state[idx, v.COL_FUEL] = 100.0
            if hasattr(v, 'reset'):
                v.reset()
            self.global_state[idx, 5] = yaw

        return {a: self.vehicles[a].get_observation() for a in self.agents}, \
               {a: {} for a in self.agents}

    # ──────────────────────────────────────────────────────────────────────────
    # SPACES
    # ──────────────────────────────────────────────────────────────────────────
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        if "escort" in agent:
            return gym.spaces.Box(low=-1.0, high=1.0, shape=(9,), dtype=np.float32)
        elif "threat" in agent:
            return gym.spaces.Box(low=-1.0, high=1.0, shape=(8,), dtype=np.float32)
        else:
            return gym.spaces.Box(low=-1.0, high=1.0, shape=(8,), dtype=np.float32)

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return gym.spaces.Box(
            low=np.array([-0.5, -1.0]),
            high=np.array([1.0,  1.0]),
            dtype=np.float32)