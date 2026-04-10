import numpy as np
import matplotlib.pyplot as plt

from agents.lte_dqn_agent import LTEDQNAgent
from envs.lte_padded_env import PaddedLTESchedulerEnv
from envs.lte_scheduler_env import LTESchedulerEnv


MODEL_PATH = "runs/lte_dqn/lte_dqn_shared_q.pt"
MAX_N_UE = 40
EVAL_N_UE = 3


def make_base_env(n_ue: int, seed: int | None = None) -> LTESchedulerEnv:
    return LTESchedulerEnv(
        n_ue=n_ue,
        n_rb_dl=50,
        episode_len=500,
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


if __name__ == "__main__":
    env = make_env(n_ue=EVAL_N_UE, max_n_ue=MAX_N_UE, seed=123)
    agent = LTEDQNAgent.load(MODEL_PATH)
    base_env = env.unwrapped

    obs, info = env.reset()
    done = False
    tti_to_show = 500
    shown_tti = 0

    plt.ion()
    fig_grid, ax_grid = plt.subplots(figsize=(8, 5))
    window_tti = 20

    print(f"[INFO] Animate mode: custom LTE DQN ({EVAL_N_UE} UE env, padded to {MAX_N_UE})")

    while not done and shown_tti < tti_to_show:
        action = agent.predict(obs, info["action_mask"], deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        if info["rbg_step"] == 0:
            tti = info["tti"] - 1
            base_env.render(mode="human")

            ax_grid.clear()

            if len(base_env.grid_history) > 0:
                grid = np.array(base_env.grid_history[-window_tti:])
                T, _ = grid.shape

                grid_vis = grid.copy()
                grid_vis[grid_vis < 0] = base_env.n_ue
                cmap = plt.get_cmap("tab10", base_env.n_ue + 1)

                ax_grid.imshow(
                    grid_vis.T,
                    aspect="auto",
                    origin="lower",
                    cmap=cmap,
                    vmin=0,
                    vmax=base_env.n_ue,
                )

                ax_grid.set_xlabel("TTI (последние)")
                ax_grid.set_ylabel("RBG index")
                ax_grid.set_title("Resource grid: time (X) vs RBG index (Y)")

                for x in range(T):
                    for y in range(base_env.n_rbg):
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
                ax_grid.set_yticks(range(base_env.n_rbg))
                ax_grid.set_yticklabels(range(base_env.n_rbg))

            plt.pause(0.2)
            shown_tti += 1

        done = terminated or truncated

    plt.ioff()
    plt.show()
    env.close()
