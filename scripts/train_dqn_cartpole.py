import csv
import os
import random

import gymnasium as gym
import numpy as np
import torch

from agents.dqn_agent import DQNAgent
from utils.replay_buffer import ReplayBuffer

def save_metrics_to_csv(path: str, episode_rewards: list, losses: list):
    """Сохраняет метрики обучения в CSV файл."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "episode_reward", "train_step", "loss"])

        max_len = max(len(episode_rewards), len(losses))
        for i in range(max_len):
            episode        = i if i < len(episode_rewards) else ""
            episode_reward = episode_rewards[i] if i < len(episode_rewards) else ""
            train_step     = i if i < len(losses) else ""
            loss           = losses[i] if i < len(losses) else ""
            writer.writerow([episode, episode_reward, train_step, loss])

def main():
    os.makedirs("runs", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    env         = gym.make("CartPole-v1")
    state, _    = env.reset(seed=0)
    state_dim   = env.observation_space.shape[0]
    action_dim  = env.action_space.n

    agent   = DQNAgent(state_dim, action_dim, device=device)
    buffer  = ReplayBuffer(capacity=10000)

    num_episodes        = 1000
    batch_size          = 64
    learning_starts     = 500
    target_update_freq  = 500
    episode_rewards     = []
    losses              = []
    global_step         = 0

    for episode in range(num_episodes):
        state, _        = env.reset()
        done            = False
        total_reward    = 0.0

        while not done:

            # Выбираем действие
            global_step     += 1
            action          = agent.select_action(state)

            # Делаем шаг в среде
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Кладем переход в буфер
            buffer.push(state, action, reward, next_state, done)

            state           = next_state
            total_reward    += reward

            # Обучение DQN если достаточно данных
            if len(buffer) >= learning_starts and global_step % 4 == 0:
                batch = buffer.sample(batch_size)
                loss  = agent.train_step(batch)
                losses.append(loss)

            # Периодически обновляем таргет-сеть
            if global_step % target_update_freq == 0:
                agent.target_q_net.load_state_dict(agent.q_net.state_dict())

        episode_rewards.append(total_reward)

        if (episode + 1) % 10 == 0:
            avg_reward  = np.mean(episode_rewards[-10:])
            current_eps = agent._get_epsilon()
            print(
                f"Episode {episode+1:>3} | "
                f"avg_reward (last 10): {avg_reward:>6.1f} | "
                f"epsilon: {current_eps:.3f}"
            )

    # Сохраняем результаты
    csv_path = "runs/dqn_cartpole.csv"
    save_metrics_to_csv(csv_path, episode_rewards, losses)
    print(f"Метрики сохранены: {csv_path}")
    weights_path = "runs/dqn_cartpole_weights.pth"
    torch.save(agent.q_net.state_dict(), weights_path)
    print(f"Веса модели сохранены: {weights_path}")
    print("Обучение DQN на CartPole завершено (базовая версия).")
    print("Для построения графиков запусти: python -m scripts.plot_results runs/dqn_cartpole.csv")

if __name__ == "__main__":
    main()