import csv
import os

import numpy as np
import torch

from agents.lte_dqn_agent import LTEDQNAgent, MaskedReplayBuffer
from envs.lte_padded_env import PaddedLTESchedulerEnv
from envs.lte_scheduler_env import LTESchedulerEnv


TRAIN_N_UE  = 3
MAX_N_UE    = 40


def make_base_env(n_ue: int, seed: int | None = None) -> LTESchedulerEnv:
    return LTESchedulerEnv(
        n_ue=n_ue,
        n_rb_dl=50,
        episode_len=300,
        reward_mode="per_tti",
        reward_window=1,
        alpha=1.0,
        beta=2.0,
        wb_cqi_report_period_tti=5,
        traffic_lambda=5000.0,
        traffic_profile=("full_buffer", "on_off", "bursty") if n_ue == 3 else "on_off",
        rate_scale_bps=1e6,
        jfi_target=0.70,
        lambda_jfi=2.0,
        seed=seed,
    )


def make_env(n_ue: int, max_n_ue: int, seed: int | None = None) -> PaddedLTESchedulerEnv:
    return PaddedLTESchedulerEnv(make_base_env(n_ue=n_ue, seed=seed), max_n_ue=max_n_ue)


def evaluate_agent(agent: LTEDQNAgent, env: PaddedLTESchedulerEnv) -> dict[str, float]:
    obs, info   = env.reset()
    done        = False
    total_steps = 0
    invalid_action_count = 0

    while not done:
        action = agent.predict(obs, info["action_mask"], deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        invalid_action_count += int(info.get("invalid_action", False) or info.get("padded_invalid_action", False))
        total_steps += 1
        done = terminated or truncated

    summary = env.unwrapped.get_episode_summary()
    summary["invalid_action_rate"] = invalid_action_count / max(total_steps, 1)
    return summary


def save_metrics_csv(path: str, episode_metrics: list[dict[str, float]]) -> None:
    if not episode_metrics:
        return

    fieldnames = list(episode_metrics[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(episode_metrics)


def main():
    os.makedirs("runs/lte_dqn", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    env = make_env(n_ue=TRAIN_N_UE, max_n_ue=MAX_N_UE, seed=0)
    obs, info = env.reset()

    agent = LTEDQNAgent(
        max_n_ue=env.max_n_ue,
        ue_feature_dim=env.ue_feature_dim,
        context_dim=env.context_dim,
        hidden_dim=64,
        device=device,
        epsilon_start=1.0,
        epsilon_end=0.10,
        epsilon_decay_steps=200_000,
        gamma=0.995,
        lr=3e-4,
    )
    replay_buffer = MaskedReplayBuffer(capacity=100_000)

    total_env_steps     = 200_000
    learning_starts     = 5_000
    batch_size          = 128
    train_freq          = env.unwrapped.n_rbg
    gradient_steps      = 1
    target_update_freq  = 2_000
    log_interval_tti    = 100

    global_step = 0
    episode_idx = 0
    episode_metrics: list[dict[str, float]] = []
    losses: list[float] = []

    while global_step < total_env_steps:
        action = agent.select_action(obs, info["action_mask"], deterministic=False)
        next_obs, reward, terminated, truncated, next_info = env.step(action)
        done = terminated or truncated

        replay_buffer.push(
            obs,
            info["action_mask"],
            action,
            float(reward),
            next_obs,
            next_info["action_mask"],
            done,
        )

        obs = next_obs
        info = next_info
        global_step += 1

        if len(replay_buffer) >= learning_starts and global_step % train_freq == 0:
            for _ in range(gradient_steps):
                batch = replay_buffer.sample(batch_size)
                losses.append(agent.train_step(batch))

        if global_step % target_update_freq == 0:
            agent.update_target()

        if done:
            episode_idx += 1
            summary = env.unwrapped.get_episode_summary()
            summary["episode"] = episode_idx
            summary["epsilon"] = agent._get_epsilon()
            summary["global_step"] = global_step
            summary["mean_loss_last_100"] = float(np.mean(losses[-100:])) if losses else 0.0
            episode_metrics.append(summary)

            if episode_idx % 5 == 0:
                print(
                    f"Episode {episode_idx:>3} | "
                    f"step {global_step:>7} | "
                    f"mean_reward: {summary.get('mean_reward', 0.0):.4f} | "
                    f"mean_se: {summary.get('mean_se_bps_hz', 0.0):.4f} | "
                    f"mean_jfi: {summary.get('mean_jfi', 0.0):.4f} | "
                    f"eps: {summary['epsilon']:.3f}"
                )

            obs, info = env.reset()

    weights_path = "runs/lte_dqn/lte_dqn_shared_q.pt"
    agent.save(weights_path)
    print(f"Weights saved: {weights_path}")

    metrics_path = "runs/lte_dqn/train_metrics.csv"
    save_metrics_csv(metrics_path, episode_metrics)
    print(f"Metrics saved: {metrics_path}")

    eval_env_train = make_env(n_ue=TRAIN_N_UE, max_n_ue=MAX_N_UE, seed=123)
    train_summary = evaluate_agent(agent, eval_env_train)
    print("\n=== Eval: train scenario ===")
    for key, value in train_summary.items():
        print(f"{key}: {value:.4f}")

    eval_env_40 = make_env(n_ue=40, max_n_ue=MAX_N_UE, seed=321)
    eval_40_summary = evaluate_agent(agent, eval_env_40)
    print("\n=== Eval: 40 UE scenario ===")
    for key, value in eval_40_summary.items():
        print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    main()
