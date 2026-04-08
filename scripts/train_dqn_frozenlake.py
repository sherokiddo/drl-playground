import csv
import os

import gymnasium as gym
import numpy as np
import torch

from agents.dqn_agent import DQNAgent
from utils.replay_buffer import ReplayBuffer


def state_to_vec(s: int, n_states: int) -> np.ndarray:
    vec = np.zeros(n_states, dtype=np.float32)
    vec[s] = 1.0
    return vec


def save_metrics_to_csv(path: str, episode_rewards: list, success_flags: list):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["episode", "episode_reward", "success"])
        for i, (r, s) in enumerate(zip(episode_rewards, success_flags)):
            w.writerow([i, r, int(s)])


def main():
    os.makedirs("runs", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    env = gym.make(
        "FrozenLake-v1",
        is_slippery=True,
        map_name="4x4",
    )
    
    n_states  = env.observation_space.n
    n_actions = env.action_space.n

    agent = DQNAgent(state_dim=n_states, action_dim=n_actions, device=device)
    buffer = ReplayBuffer(capacity=50000)

    num_episodes       = 5000
    batch_size         = 64
    learning_starts    = 1000
    target_update_freq = 500
    max_steps_per_ep   = 100

    episode_rewards = []
    success_flags   = []   # 1 если достигли цели в эпизоде

    global_step = 0

    for episode in range(num_episodes):
        s, _ = env.reset()
        state_vec = state_to_vec(s, n_states)

        done         = False
        total_reward = 0.0
        steps_in_ep  = 0
        success      = 0

        while not done and steps_in_ep < max_steps_per_ep:
            steps_in_ep += 1
            global_step += 1

            action = agent.select_action(state_vec)

            ns, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            next_state_vec = state_to_vec(ns, n_states)
            buffer.push(state_vec, action, reward, next_state_vec, done)

            state_vec = next_state_vec
            total_reward += reward

            if reward > 0:
                success = 1

            if len(buffer) >= learning_starts:
                batch = buffer.sample(batch_size)
                agent.train_step(batch)

            if global_step % target_update_freq == 0:
                agent.target_q_net.load_state_dict(agent.q_net.state_dict())

        episode_rewards.append(total_reward)
        success_flags.append(success)

        if (episode + 1) % 100 == 0:
            avg_reward  = np.mean(episode_rewards[-100:])
            avg_success = np.mean(success_flags[-100:]) * 100.0
            eps         = agent._get_epsilon()
            print(
                f"Episode {episode+1:>4} | "
                f"avg_reward(last 100): {avg_reward:>5.2f} | "
                f"success_rate(last 100): {avg_success:>5.1f}% | "
                f"epsilon: {eps:.3f}"
            )

    env.close()

    csv_path = "runs/dqn_frozenlake.csv"
    save_metrics_to_csv(csv_path, episode_rewards, success_flags)
    print(f"Метрики сохранены: {csv_path}")

    weights_path = "runs/dqn_frozenlake_weights.pth"
    torch.save(agent.q_net.state_dict(), weights_path)
    print(f"Веса сохранены: {weights_path}")
    print("Для графика: python -m scripts.plot_results_frozenlake")


if __name__ == "__main__":
    main()