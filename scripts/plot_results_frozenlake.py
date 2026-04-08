import csv
import os

import numpy as np
import matplotlib.pyplot as plt


def load_metrics(path: str):
    episodes = []
    rewards  = []
    success  = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            episodes.append(int(row["episode"]))
            rewards.append(float(row["episode_reward"]))
            success.append(int(row["success"]))
    return np.array(episodes), np.array(rewards), np.array(success)


def rolling_mean(values, window):
    if len(values) < window:
        return None, None
    conv = np.convolve(values, np.ones(window) / window, mode="valid")
    x    = np.arange(window - 1, window - 1 + len(conv))
    return x, conv


def main():
    csv_path = "runs/dqn_frozenlake.csv"
    if not os.path.exists(csv_path):
        print(f"Файл не найден: {csv_path}")
        return

    episodes, rewards, success = load_metrics(csv_path)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    fig.suptitle("DQN on FrozenLakeCustom (4x4, slippery=True)")

    # 1) Reward per episode
    ax1 = axes[0]
    ax1.plot(episodes, rewards, alpha=0.3, label="Reward per episode")
    x_r, y_r = rolling_mean(rewards, window=100)
    if x_r is not None:
        ax1.plot(x_r, y_r, color="steelblue", label="Rolling avg reward (100)")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Reward")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2) Success rate (per episode)
    ax2 = axes[1]
    x_s, y_s = rolling_mean(success, window=100)
    ax2.plot(episodes, success, alpha=0.1, label="Success (0/1)")
    if x_s is not None:
        ax2.plot(x_s, y_s * 100.0, color="green", label="Success rate % (100)")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Success %")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    os.makedirs("runs", exist_ok=True)
    out_path = "runs/dqn_frozenlake.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"График сохранён: {out_path}")


if __name__ == "__main__":
    main()