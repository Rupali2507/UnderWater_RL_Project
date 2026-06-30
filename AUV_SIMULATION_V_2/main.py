from pettingzoo_env.tactical_naval_env import TacticalNavalEnv
import numpy as np

# 1. Configuration (Load your scenario)
scenario_config = {
    "location": "Open Ocean",
    "agents": {
        "ship_1": {"class_type": "NavalShip", "team": "BLUE", "start_pos": [100, 100]},
        "cargo_1": {"class_type": "CargoShip", "team": "BLUE", "start_pos": [200, 200]}
    }
}

# 2. Init Environment
env = TacticalNavalEnv(scenario_config=scenario_config, render_mode="human")
obs, _ = env.reset()

# 3. Manual Loop
# Replace this with your own input logic (e.g., keyboard library)
for _ in range(1000):
    actions = {}
    for agent in env.agents:
        # Example Manual Logic: Move forward (throttle 0.5) and turn slightly (rudder 0.1)
        actions[agent] = np.array([0.5, 0.1, 0.0])
    
    # Step the environment
    obs, rewards, terminations, truncations, infos = env.step(actions)
    
    # Check if any agents crashed
    if any(terminations.values()):
        break

env.close()