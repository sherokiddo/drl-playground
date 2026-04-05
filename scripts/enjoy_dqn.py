"""
Визуализация обученного DQN-агента на CartPole-v1.

Использование:
    python -m scripts.enjoy_dqn runs/dqn_cartpole_weights.pth

Агент будет играть несколько эпизодов с анимацией в окне.
"""

import sys
import torch
import gymnasium as gym

from agents.dqn_agent import DQNAgent


def main():
    if len(sys.argv) < 2:
        print("Использование: python -m scripts.enjoy_dqn <path_to_weights>")
        sys.exit(1)

    weights_path = sys.argv[1]

    # Создаём среду с визуализацией
    # render_mode="human" открывает окно с анимацией
    env = gym.make("CartPole-v1", render_mode="human")

    state_dim  = env.observation_space.shape[0]
    action_dim = env.action_space.n

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Создаём агента и загружаем обученные веса
    agent = DQNAgent(state_dim, action_dim, device=device)
    agent.q_net.load_state_dict(torch.load(weights_path, map_location=device))
    agent.q_net.eval()  # переводим сеть в режим inference (без градиентов)

    agent.epsilon_end   = 0.0
    agent.epsilon_start = 0.0

    num_episodes = 5
    print(f"Запуск {num_episodes} эпизодов. Закрой окно, чтобы остановить.")

    for episode in range(num_episodes):
        state, _ = env.reset()
        done         = False
        total_reward = 0.0

        while not done:
            # render вызывается автоматически при render_mode="human"
            action = agent.select_action(state)
            state, reward, terminated, truncated, _ = env.step(action)
            done          = terminated or truncated
            total_reward += reward

        print(f"Episode {episode + 1}: total reward = {total_reward}")

    env.close() 


if __name__ == "__main__":
    main()