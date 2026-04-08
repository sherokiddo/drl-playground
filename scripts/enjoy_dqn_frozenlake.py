import time
import torch
import numpy as np
import gymnasium as gym

from envs.frozen_lake_env import FrozenLakeCustom
from agents.dqn_agent import DQNAgent


def state_to_vec(s: int, n_states: int) -> np.ndarray:
    vec = np.zeros(n_states, dtype=np.float32)
    vec[s] = 1.0
    return vec


def main():
    weights_path = "runs/dqn_frozenlake_weights.pth"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    env = gym.make(
        "FrozenLake-v1",
        is_slippery=True,
        map_name="4x4",
        render_mode="human",
    )
    n_states  = env.observation_space.n
    n_actions = env.action_space.n

    agent = DQNAgent(state_dim=n_states, action_dim=n_actions, device=device)
    agent.q_net.load_state_dict(torch.load(weights_path, map_location=device))
    agent.q_net.eval()

    # Выключаем exploration: только argmax
    agent.epsilon_start = 0.0
    agent.epsilon_end   = 0.0

    num_episodes = 5
    max_steps    = 100

    for ep in range(num_episodes):
        s, _ = env.reset()
        state_vec = state_to_vec(s, n_states)
        done = False
        total_reward = 0.0
        step = 0

        print(f"\n=== Episode {ep+1} ===")
        while not done and step < max_steps:
            step += 1
            action = agent.select_action(state_vec)
            ns, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            state_vec = state_to_vec(ns, n_states)
            total_reward += reward

            print(f"Step {step}: action={action}, reward={reward}")
            time.sleep(0.3)

        print(f"Episode {ep+1} finished with total_reward={total_reward}")

    env.close()


if __name__ == "__main__":
    main()