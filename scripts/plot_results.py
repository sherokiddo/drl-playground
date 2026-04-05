"""
Скрипт для визуализации метрик обучения из CSV.

Использование:
    python -m scripts.plot_results runs/dqn_cartpole.csv

CSV должен содержать столбцы:
    episode, episode_reward, train_step, loss
"""

import sys
import os
import csv

import numpy as np
import matplotlib.pyplot as plt


def load_metrics(path: str):
    """Читаем CSV и возвращаем два списка: episode_rewards и losses."""
    episode_rewards = []
    losses          = []

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["episode_reward"] != "":
                episode_rewards.append(float(row["episode_reward"]))
            if row["loss"] != "":
                losses.append(float(row["loss"]))

    return episode_rewards, losses


def rolling_mean(values: list, window: int):
    """Скользящее среднее через np.convolve."""
    if len(values) < window:
        return [], []
    smoothed = np.convolve(values, np.ones(window) / window, mode="valid")
    # Индексы по оси X: первая точка rolling среднего соответствует индексу (window-1)
    x = list(range(window - 1, len(values)))
    return x, smoothed.tolist()


def plot(episode_rewards: list, losses: list, title: str, save_path: str):
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    fig.suptitle(title, fontsize=14)

    # --- График 1: Episode Reward ---
    ax1 = axes[0]
    ax1.plot(episode_rewards, alpha=0.35, color="steelblue", label="Reward per episode")

    x_roll, y_roll = rolling_mean(episode_rewards, window=20)
    if x_roll:
        ax1.plot(x_roll, y_roll, color="steelblue", linewidth=2, label="Rolling avg (window=20)")

    ax1.axhline(y=195, color="green", linestyle="--", linewidth=1, label="Solved threshold (195)")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Total Reward")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # --- График 2: TD Loss ---
    ax2 = axes[1]
    ax2.plot(losses, alpha=0.25, color="tomato", label="TD Loss")

    x_roll_l, y_roll_l = rolling_mean(losses, window=200)
    if x_roll_l:
        ax2.plot(x_roll_l, y_roll_l, color="tomato", linewidth=2, label="Rolling avg (window=200)")

    ax2.set_xlabel("Training step")
    ax2.set_ylabel("MSE Loss")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"График сохранён: {save_path}")


def main():
    if len(sys.argv) < 2:
        print("Использование: python -m scripts.plot_results <path_to_csv>")
        sys.exit(1)

    csv_path = sys.argv[1]
    if not os.path.exists(csv_path):
        print(f"Файл не найден: {csv_path}")
        sys.exit(1)

    # Имя графика = имя CSV без расширения + .png
    base_name  = os.path.splitext(csv_path)[0]
    save_path  = base_name + ".png"

    # Заголовок графика из имени файла
    title = os.path.basename(base_name).replace("_", " ").title()

    episode_rewards, losses = load_metrics(csv_path)
    print(f"Загружено: {len(episode_rewards)} эпизодов, {len(losses)} шагов обучения")

    plot(episode_rewards, losses, title=title, save_path=save_path)


if __name__ == "__main__":
    main()