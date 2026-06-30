import pygame
import numpy as np
import sys, os, math

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pettingzoo_env.tactical_naval_env import (
    TacticalNavalEnv, ZONE_A_RADIUS, ZONE_B_RADIUS, ZONE_C_RADIUS)


class TacticalVisualizer:
    def __init__(self, width=900, height=900, world_size=7000):
        pygame.init()
        pygame.display.set_caption("DRDO Naval Simulator")
        self.screen   = pygame.display.set_mode((width, height))
        self.clock    = pygame.time.Clock()
        self.W, self.H = width, height
        self.scale    = width / world_size
        self.font     = pygame.font.SysFont("monospace", 11)
        self.font_big = pygame.font.SysFont("monospace", 26, bold=True)

    def ws(self, x, y):   # world → screen
        return int(x * self.scale), int(self.H - y * self.scale)

    def draw(self, env, step, outcome=None):
        self.screen.fill((8, 12, 22))

        # Grid
        for g in range(0, 8000, 1000):
            px = int(g * self.scale); py = int(self.H - g * self.scale)
            pygame.draw.line(self.screen, (18, 35, 55), (px, 0), (px, self.H))
            pygame.draw.line(self.screen, (18, 35, 55), (0, py), (self.W, py))

        # Destination
        dx, dy = self.ws(5000, 5000)
        pygame.draw.circle(self.screen, (0, 220, 90), (dx, dy), 10, 2)
        self.screen.blit(self.font.render("DEST", True, (0, 220, 90)), (dx+12, dy-6))

        # Compute convoy centre for zone circles
        cargo_agents = [a for a in env.agents if "cargo" in a]
        if cargo_agents:
            cx = np.mean([env.global_state[env.vehicles[c].agent_index, 0].item() for c in cargo_agents])
            cy = np.mean([env.global_state[env.vehicles[c].agent_index, 1].item() for c in cargo_agents])
            scx, scy = self.ws(cx, cy)
            # Zone C (gold, innermost)
            pygame.draw.circle(self.screen, (120, 90, 0),
                (scx, scy), int(ZONE_C_RADIUS * self.scale), 1)
            # Zone B (blue)
            pygame.draw.circle(self.screen, (0, 60, 120),
                (scx, scy), int(ZONE_B_RADIUS * self.scale), 1)
            # Zone A (red, outermost)
            pygame.draw.circle(self.screen, (80, 0, 0),
                (scx, scy), int(ZONE_A_RADIUS * self.scale), 1)
            # Labels
            self.screen.blit(self.font.render("C", True, (160, 120, 0)),
                (scx + int(ZONE_C_RADIUS*self.scale)+3, scy))
            self.screen.blit(self.font.render("B", True, (0, 100, 180)),
                (scx + int(ZONE_B_RADIUS*self.scale)+3, scy))
            self.screen.blit(self.font.render("A", True, (160, 0, 0)),
                (scx + int(ZONE_A_RADIUS*self.scale)+3, scy))

        # Agents
        for name, v in env.vehicles.items():
            x   = env.global_state[v.agent_index, 0].item()
            y   = env.global_state[v.agent_index, 1].item()
            yaw = env.global_state[v.agent_index, 5].item()
            sx, sy = self.ws(x, y)

            is_cargo  = "cargo"  in name
            is_threat = "threat" in name
            color = (255, 215, 0) if is_cargo else (255, 55, 55) if is_threat else (70, 170, 255)

            # Dest line
            ddx, ddy = self.ws(*v._dest_position[:2])
            pygame.draw.line(self.screen, tuple(c//5 for c in color), (sx, sy), (ddx, ddy))

            # Icon
            if is_cargo:
                pygame.draw.rect(self.screen, color, (sx-8, sy-8, 16, 16), 2)
            elif is_threat:
                tip  = (sx+int(math.cos(yaw)*13), sy-int(math.sin(yaw)*13))
                l    = (sx+int(math.cos(yaw+2.4)*7), sy-int(math.sin(yaw+2.4)*7))
                r    = (sx+int(math.cos(yaw-2.4)*7), sy-int(math.sin(yaw-2.4)*7))
                pygame.draw.polygon(self.screen, color, [tip, l, r], 2)
            else:
                pygame.draw.circle(self.screen, color, (sx, sy), 7, 2)
                # Show negotiation ring
                if name in env._negotiating:
                    pygame.draw.circle(self.screen, (255, 255, 0), (sx, sy), 14, 1)

            # Heading
            pygame.draw.line(self.screen, (160, 160, 160),
                (sx, sy), (sx+int(math.cos(yaw)*20), sy-int(math.sin(yaw)*20)))

            # Label: name + status + dist
            status = v.status.name if hasattr(v, 'status') else ""
            dist   = v._get_distance_to(v._dest_position)
            label  = f"{name}  {status}  {dist:.0f}m"
            self.screen.blit(self.font.render(label, True, color), (sx+10, sy-6))

        # HUD
        self.screen.blit(
            self.font.render(f"Step:{step}  [R]reset  [ESC]quit", True, (90, 90, 110)), (6, 6))

        # Outcome banner
        if outcome:
            clr = {"MISSION WIN": (0,220,90), "MISSION LOSS": (255,55,55),
                   "OUT OF BOUNDS": (255,165,0)}.get(outcome, (200,200,200))
            banner = self.font_big.render(f"  {outcome} — press R  ", True, (8,12,22), clr)
            self.screen.blit(banner,
                (self.W//2 - banner.get_width()//2, self.H//2 - banner.get_height()//2))

        pygame.display.flip()

    def handle_events(self):
        for e in pygame.event.get():
            if e.type == pygame.QUIT: return 'quit'
            if e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE: return 'quit'
                if e.key == pygame.K_r:      return 'reset'
        return 'run'


def make_config():
    return {
        "location": "Open Ocean",
        "agents": {
            "escort_1": {"class_type": "NavalShip",  "team": "BLUE",
                         "start_pos": [500, 500],   "dest_pos": [5000, 5000]},
            "cargo_1":  {"class_type": "CargoShip",  "team": "BLUE",
                         "start_pos": [500, 600],   "dest_pos": [5000, 5000]},
            "threat_1": {"class_type": "ThreatShip", "team": "RED",
                         "start_pos": [3500, 2000], "dest_pos": [5000, 5000]},
        }
    }


def main():
    env     = TacticalNavalEnv(scenario_config=make_config())
    viz     = TacticalVisualizer()
    env.reset()
    step, outcome = 0, None

    while True:
        sig = viz.handle_events()
        if sig == 'quit': break
        if sig == 'reset':
            env = TacticalNavalEnv(scenario_config=make_config())
            env.reset(); step = 0; outcome = None

        if outcome is None:
            actions = {
                "escort_1": np.array([0.6,  0.05], dtype=np.float32),
                "cargo_1":  np.array([0.4,  0.0],  dtype=np.float32),
                "threat_1": np.array([0.8, -0.1],  dtype=np.float32),
            }
            _, _, terms, truncs, infos = env.step(actions)

            if any(truncs.values()):
                outcome = "OUT OF BOUNDS"
            elif any(terms.values()):
                i = infos[env.agents[0]]
                outcome = "MISSION WIN" if i.get("mission_win") else "MISSION LOSS"

            step += 1

        viz.draw(env, step, outcome)
        viz.clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    main()