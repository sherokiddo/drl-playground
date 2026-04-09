import numpy as np
import matplotlib.pyplot as plt

from envs.lte_scheduler_env import LTESchedulerEnv
from stable_baselines3 import DQN
from utils.lte_transfer import predict_transfer_action


MODEL_N_UE = 3
EVAL_N_UE = 40


def make_env():
    return LTESchedulerEnv(
        n_ue=EVAL_N_UE,
        n_rbg=16,
        episode_len=500,
        reward_mode="per_tti",
        reward_window=1,
        alpha=1.0,
        beta=2.0,
        bandwidth_hz=10e6,
        traffic_lambda=5000.0,
        rate_scale_bps=1e6,
        jfi_target=0.70,
        lambda_jfi=2.0,
        seed=123,
    )


if __name__ == "__main__":
    model_path = "runs/dqn_scheduler/dqn_lte_scheduler.zip"

    env = make_env()
    model = DQN.load(model_path)

    obs, info = env.reset()
    done = False
    tti_to_show = 500
    shown_tti = 0

    transfer_enabled = EVAL_N_UE != MODEL_N_UE
    mode_label = f"top-{MODEL_N_UE} PF transfer" if transfer_enabled else "direct"
    print(f"[INFO] Evaluation mode: {mode_label} (model {MODEL_N_UE} UE -> env {EVAL_N_UE} UE)")

    plt.ion()
    fig_grid, ax_grid = plt.subplots(figsize=(9, 5))

    window_tti = 20

    while not done and shown_tti < tti_to_show:
        action, candidates, _, _ = predict_transfer_action(
            model=model,
            env=env,
            model_n_ue=MODEL_N_UE,
            deterministic=True,
        )
        obs, reward, terminated, truncated, info = env.step(action)

        if env._current_rbg == 0:
            tti = env._current_tti - 1
            env.render(mode="human")
            if candidates is not None:
                print(f"  Transfer shortlist (last RBG): {candidates.tolist()}")

            ax_grid.clear()

            if len(env.grid_history) > 0:
                grid = np.array(env.grid_history[-window_tti:])
                T, _ = grid.shape

                grid_vis = grid.copy()
                grid_vis[grid_vis < 0] = env.n_ue

                cmap = plt.get_cmap("nipy_spectral", env.n_ue + 1)
                ax_grid.imshow(
                    grid_vis.T,
                    aspect="auto",
                    origin="lower",
                    cmap=cmap,
                    vmin=0,
                    vmax=env.n_ue,
                )

                ax_grid.set_xlabel("TTI")
                ax_grid.set_ylabel("RBG index")
                ax_grid.set_title(f"Resource grid ({mode_label})")

                if env.n_ue <= 10:
                    for x in range(T):
                        for y in range(env.n_rbg):
                            ue = grid[x, y]
                            if ue >= 0:
                                ax_grid.text(
                                    x,
                                    y,
                                    str(ue),
                                    ha="center",
                                    va="center",
                                    fontsize=6,
                                    color="white",
                                )

                start_tti = max(0, tti - T + 1)
                xticks = np.arange(T)
                xticklabels = [str(start_tti + i) for i in range(T)]
                ax_grid.set_xticks(xticks)
                ax_grid.set_xticklabels(xticklabels, rotation=45)
                ax_grid.set_yticks(range(env.n_rbg))
                ax_grid.set_yticklabels(range(env.n_rbg))

            plt.pause(0.2)
            shown_tti += 1

        done = terminated or truncated

    plt.ioff()
    plt.show()
    env.close()
