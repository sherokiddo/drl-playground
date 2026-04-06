import csv
import os

import gymnasium as gym
import numpy as np
import torch

from agents.ppo_agent import PPOAgent
from utils.rollout_buffer import RolloutBuffer


def save_metrics_to_csv(path: str, episode_rewards: list, losses: list):
    """Сохраняет метрики обучения в CSV."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "episode_reward", "train_step", "loss"])
        max_len = max(len(episode_rewards), len(losses))
        for i in range(max_len):
            episode        = i           if i < len(episode_rewards) else ""
            episode_reward = episode_rewards[i] if i < len(episode_rewards) else ""
            train_step     = i           if i < len(losses)          else ""
            loss           = losses[i]   if i < len(losses)          else ""
            writer.writerow([episode, episode_reward, train_step, loss])


def main():
    os.makedirs("runs", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    env        = gym.make("CartPole-v1")
    state_dim  = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent  = PPOAgent(state_dim, action_dim, device=device)
    buffer = RolloutBuffer()

    # --- Гиперпараметры цикла ---
    # rollout_steps: сколько шагов среды собираем перед каждым update()
    # Типичные значения: 128–2048. Больше → стабильнее, но медленнее реакция.
    rollout_steps  = 128
    total_steps    = 80_000  
    log_interval   = 10

    episode_rewards = []
    losses          = []

    state, _ = env.reset(seed=0)
    episode_reward   = 0.0
    episode_count    = 0
    global_step      = 0

    print(f"Начинаем обучение: total_steps={total_steps}, rollout_steps={rollout_steps}")

    while global_step < total_steps:

        # ────────────────────────────────────────────
        # Фаза 1: сбор роллаута (rollout_steps шагов)
        # ────────────────────────────────────────────
        buffer.clear()

        for _ in range(rollout_steps):
            global_step += 1

            # Выбираем действие — получаем action, log_prob, value
            action, log_prob, value = agent.select_action(state)

            # Делаем шаг в среде
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Сохраняем переход в буфер роллаута
            buffer.push(state, action, reward, done, log_prob, value)

            state          = next_state
            episode_reward += reward

            if done:
                # Эпизод закончился — сохраняем награду и сбрасываем среду
                episode_rewards.append(episode_reward)
                episode_count  += 1
                episode_reward  = 0.0
                state, _        = env.reset()

                if episode_count % log_interval == 0:
                    avg_reward = np.mean(episode_rewards[-log_interval:])
                    print(
                        f"Episode {episode_count:>4} | "
                        f"avg_reward (last {log_interval}): {avg_reward:>6.1f} | "
                        f"global_step: {global_step:>7}"
                    )

        # ────────────────────────────────────────────
        # Фаза 2: считаем last_value для незавершённого эпизода
        # ────────────────────────────────────────────
        # После роллаута текущий state может быть посередине эпизода.
        # Если эпизод не закончился — критик оценивает это состояние,
        # Если закончился (done=True в последнем шаге) — last_value = 0.
        last_done = buffer.dones[-1]
        if last_done:
            last_value = torch.zeros(1, device=device)
        else:
            state_tensor = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                _, last_value_tensor = agent.ac(state_tensor)
            last_value = last_value_tensor.squeeze(-1)

        # ────────────────────────────────────────────
        # Фаза 3: обновление политики
        # ────────────────────────────────────────────
        loss = agent.update(buffer, last_value)
        losses.append(loss)

    # ────────────────────────────────────────────
    # После обучения: сохраняем метрики и веса
    # ────────────────────────────────────────────
    csv_path = "runs/ppo_cartpole.csv"
    save_metrics_to_csv(csv_path, episode_rewards, losses)
    print(f"Метрики сохранены: {csv_path}")

    weights_path = "runs/ppo_cartpole_weights.pth"
    torch.save(agent.ac.state_dict(), weights_path)
    print(f"Веса модели сохранены: {weights_path}")

    print(f"Для графиков: python -m scripts.plot_results runs/ppo_cartpole.csv")
    print(f"Для анимации: python -m scripts.enjoy_ppo runs/ppo_cartpole_weights.pth")


if __name__ == "__main__":
    main()