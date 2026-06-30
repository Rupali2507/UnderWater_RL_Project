import sys
import os
import numpy as np

# Ensure project root is in system path for clean imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pettingzoo_env.tactical_naval_env import TacticalNavalEnv

def run_pure_manual_test():
    # 1. Define scenario
    config = {
        "location": "Open Ocean",
        "agents": {
            "escort_1": {"class_type": "NavalShip",  "team": "BLUE", "start_pos": [500, 500],  "dest_pos": [5000, 5000]},
            "cargo_1":  {"class_type": "CargoShip",  "team": "BLUE", "start_pos": [500, 600],  "dest_pos": [5000, 5000]},
            "threat_1": {"class_type": "ThreatShip", "team": "RED",  "start_pos": [3000, 2000], "dest_pos": [500, 600]}
        }
    }

    # 2. Initialize environment
    env = TacticalNavalEnv(scenario_config=config)
    env.reset()

    print("--- Starting Pure Manual Physics Test ---")
    print("Action format: [Throttle, Rudder]")
    
    # 3. Manual Step Loop
    for step in range(200):
        actions = {}
        for agent in env.agents:
            # MANUAL CONTROL: Set specific behaviors for different ship types
            if "cargo" in agent:
                # Cargo moves straight at constant speed
                actions[agent] = np.array([0.4, 0.0], dtype=np.float32)
            elif "escort" in agent:
                # Escort turns hard to verify yaw integration
                actions[agent] = np.array([0.6, 0.5], dtype=np.float32)
            elif "threat" in agent:
                # Threat maneuvers aggressively
                actions[agent] = np.array([0.8, -0.3], dtype=np.float32)
            
        # 4. Step the environment
        obs, rewards, terminations, truncations, infos = env.step(actions)
        
        # 5. Telemetry output to verify physics
        if step % 20 == 0:
            print(f"\n[Step {step}]")
            for agent in env.agents:
                vehicle = env.vehicles[agent]
                idx = vehicle.agent_index
                
                # Access raw tensor state
                x = env.global_state[idx, 0].item()
                y = env.global_state[idx, 1].item()
                yaw = env.global_state[idx, 5].item()
                
                # Calculate distance to target
                # Accessing the destination position set in the vehicle
                dx = vehicle._dest_position[0] - x
                dy = vehicle._dest_position[1] - y
                dist = np.sqrt(dx**2 + dy**2)
                
                print(f"  > {agent:<9} | Pos: ({x:6.1f}, {y:6.1f}) | Yaw: {yaw:6.2f}rad | Dist to Dest: {dist:6.1f}m")
            
    print("\n--- Manual Test Complete ---")
    env.close()

if __name__ == "__main__":
    run_pure_manual_test()