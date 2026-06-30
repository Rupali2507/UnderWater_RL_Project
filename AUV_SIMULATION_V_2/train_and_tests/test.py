import sys
import os
# pyrefly: ignore [missing-import]
import numpy as np
from stable_baselines3 import PPO

# Ensure project root is in system path for clean imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pettingzoo_env.tactical_naval_env import TacticalNavalEnv

def test_naval_env():
    # 1. Configuration (Must match training environment)
    config = {
        "location": "Open Ocean",
        "agents": {
            "escort_1": {"class_type": "NavalShip",  "team": "BLUE", "start_pos": [500, 500],  "dest_pos": [5000, 5000]},
            "cargo_1":  {"class_type": "CargoShip",  "team": "BLUE", "start_pos": [500, 600],  "dest_pos": [5000, 5000]},
            "threat_1": {"class_type": "ThreatShip", "team": "RED",  "start_pos": [3000, 2000], "dest_pos": [500, 600]}
        }
    }

    # 2. Initialize the raw environment (NO Supersuit Wrappers!)
    env = TacticalNavalEnv(scenario_config=config)

    # 3. Load the trained model
    model_path = os.path.join("models", "naval_ppo_model")
    print(f"Loading model from {model_path}.zip...")
    
    try:
        model = PPO.load(model_path)
    except FileNotFoundError:
        print(f"Error: Could not find model at {model_path}.zip")
        sys.exit(1)

    # Detect model's expected obs size (set by pad_observations_v0 during training).
    # cargo/threat return 8-dim obs; escort returns 9-dim; model expects the padded max.
    model_obs_size = model.observation_space.shape[0]
    print(f"Model expects obs size: {model_obs_size}")

    # 4. Run Evaluation Episode
    obs_dict, _ = env.reset()
    print("\n--- Starting Tactical Evaluation ---")
    
    for step in range(5000):
        actions_dict = {}
        last_agent_name = None
        last_action = None
        
        # Predict the action for each live agent.
        # Zero-pad obs to model_obs_size to match what pad_observations_v0 did at training time.
        for agent_id, obs in obs_dict.items():
            if len(obs) < model_obs_size:
                padded = np.zeros(model_obs_size, dtype=np.float32)
                padded[:len(obs)] = obs
                obs = padded
            action, _states = model.predict(obs, deterministic=True)
            actions_dict[agent_id] = action
            last_agent_name = agent_id
            last_action = action

        # Step the raw environment forward
        obs_dict, rewards, terminations, truncations, infos = env.step(actions_dict)
        
        # Print telemetry every 50 steps
        if step % 50 == 0:
            print(f"\n[Step {step}]")
            print(f"  > {last_agent_name.upper()} | PPO Action: {last_action}")
           
            for agent_name in env.agents:
                vehicle = env.vehicles[agent_name]
                idx = vehicle.agent_index
                
                # Read coordinates directly from the live tensor
                x = env.global_state[idx, 0].item()
                y = env.global_state[idx, 1].item()
                
                # Get current distance to destination/target
                dist = vehicle._get_distance_to(vehicle._dest_position)
                
                print(f"  > {agent_name.upper():<9} | Pos: ({x:6.1f}, {y:6.1f}) | Distance to Target: {dist:6.1f}m")
            
        # Stop early if the environment signals a Game Over (Win/Loss)
        if any(terminations.values()):
            print(f"\n TACTICAL EVENT TRIGGERED! Episode finished early at step {step}.")
            break

    print("\n--- Evaluation Complete ---")

if __name__ == "__main__":
    test_naval_env()