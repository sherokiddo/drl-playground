"""
Визуализация обученного PPO-агента на CartPole-v1.

Использование:
    python -m scripts.enjoy_ppo runs/ppo_cartpole_weights.pth
"""

import sys
import torch
import gymnasium as gym

from agents.ppo_agent import PPOAgent


def main():
    if len(sys.argv) < 2:
        print("Использование: python -m scripts.enjoy_ppo <path_to_weights>")
        sys.exit(1)

    weights_path = sys.argv[1]

    env        = gym.make("CartPole-v1", render_mode="human")
    state_dim  = env.observation_space.shape[0]
    action_dim = env.action_space.n
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    agent = PPOAgent(state_dim, action_dim, device=device)
    agent.ac.load_state_dict(torch.load(weights_path, map_location=device))
    agent.ac.eval()

    num_episodes = 5
    print(f"Запуск {num_episodes} эпизодов.")

    for episode in range(num_episodes):
        state, _     = env.reset()
        done         = False
        total_reward = 0.0

        while not done:
            action, _, _ = agent.select_action(state)
            state, reward, terminated, truncated, _ = env.step(action)
            done          = terminated or truncated
            total_reward += reward

        print(f"Episode {episode + 1}: total reward = {total_reward}")

    env.close()


if __name__ == "__main__":
    main()