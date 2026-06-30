import sys
import os
import numpy as np
import supersuit as ss
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pettingzoo_env.tactical_naval_env import TacticalNavalEnv


def make_env():
    """
    Builds one TacticalNavalEnv with randomised start positions.
    Called once per worker — supersuit cloudpickles this instance to clone it.
    """
    ex = float(np.random.randint(200, 2000))
    ey = float(np.random.randint(200, 2000))
    cx = float(np.clip(ex + np.random.randint(-200, 200), 0, 4800))
    cy = float(np.clip(ey + np.random.randint(0,    200), 0, 4800))
    tx = float(np.random.randint(2500, 4000))
    ty = float(np.random.randint(1500, 3500))

    config = {
        "location": "Open Ocean",
        "agents": {
            "escort_1": {"class_type": "NavalShip",  "team": "BLUE",
                         "start_pos": [ex, ey], "dest_pos": [5000, 5000]},
            "cargo_1":  {"class_type": "CargoShip",  "team": "BLUE",
                         "start_pos": [cx, cy], "dest_pos": [5000, 5000]},
            "threat_1": {"class_type": "ThreatShip", "team": "RED",
                         "start_pos": [tx, ty], "dest_pos": [5000, 5000]},
        }
    }
    env = TacticalNavalEnv(scenario_config=config)

    # Step 1: equalise obs/action spaces across agents (must happen on the PettingZoo env)
    env = ss.pad_observations_v0(env)
    env = ss.pad_action_space_v0(env)

    # Step 2: convert PettingZoo ParallelEnv → single gymnasium-style VecEnv
    # (one "slot" per agent — 3 agents = 3 sub-envs inside this one vec env)
    env = ss.pettingzoo_env_to_vec_env_v1(env)

    return env


def train_naval_env():
    MODEL_DIR = "models"
    LOG_DIR   = "logs"
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR,   exist_ok=True)

    N_ENVS = 4  # parallel copies; each has 3 agent slots → 12 total streams

    # Step 3: concat_vec_envs_v1 expects a VecEnv instance (not a factory).
    # It cloudpickles the instance to spawn N_ENVS workers.
    # We build one reference env and let supersuit clone it.
    ref_env = make_env()
    env = ss.concat_vec_envs_v1(ref_env, N_ENVS,
                                 num_cpus=1,
                                 base_class='stable_baselines3')

    model = PPO(
        MlpPolicy,
        env,
        verbose=1,
        tensorboard_log=LOG_DIR,
        learning_rate=3e-4,
        n_steps=512,        # per-worker steps before update
        batch_size=64,
        n_epochs=10,
        gamma=0.99,         # 0.995 made terminal rewards near-invisible over 1000-step episodes
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.02,      # slightly higher for 3-agent diversity
        vf_coef=0.5,
        max_grad_norm=0.5,
    )

    print("--- Starting Training ---")
    print(f"Parallel envs  : {N_ENVS}  x  3 agents  =  {N_ENVS*3} total streams")
    print(f"TensorBoard    : tensorboard --logdir {LOG_DIR}")

    model.learn(
        total_timesteps=2_000_000,
        reset_num_timesteps=True,
        progress_bar=True,
    )

    save_path = os.path.join(MODEL_DIR, "naval_ppo_model")
    model.save(save_path)
    print(f"\n--- Training Complete. Model saved to: {save_path}.zip ---")


if __name__ == "__main__":
    train_naval_env()