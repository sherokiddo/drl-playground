import numpy as np
import matplotlib.pyplot as plt

from envs.lte_scheduler_env import LTESchedulerEnv
from stable_baselines3 import DQN


def make_env():
    return LTESchedulerEnv(
        n_ue=3,
        n_rb_dl=50,
        episode_len=500,
        reward_mode="per_tti",
        reward_window=1,
        alpha=1.0,
        beta=2.0,
        wb_cqi_report_period_tti=5,
        traffic_lambda=5000.0,
        traffic_profile=("full_buffer", "on_off", "bursty"),
        rate_scale_bps=1e6,
        jfi_target=0.70,
        lambda_jfi=2.0,
        seed=123,
    )


if __name__ == "__main__":
    model_path = "runs/dqn_scheduler/dqn_lte_scheduler.zip"

    env = make_env()
    model = DQN.load(model_path, env=env)

    obs, info = env.reset()
    done = False
    tti_to_show = 500   # сколько TTI анимировать
    shown_tti = 0

    plt.ion()
    fig_grid, ax_grid = plt.subplots(figsize=(8, 5))

    window_tti = 20    # ширина окна по времени для сетки

    while not done and shown_tti < tti_to_show:
        # один step = одно решение по одному RBG
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        # когда закончился TTI, env._current_rbg сброшен в 0
        if env._current_rbg == 0:
            tti = env._current_tti - 1

            env.render(mode="human")
        
            ax_grid.clear()

            if len(env.grid_history) > 0:
                grid = np.array(env.grid_history[-window_tti:]) 
                T, R = grid.shape 

                grid_vis = grid.copy()
                grid_vis[grid_vis < 0] = env.n_ue 

                cmap = plt.get_cmap("tab10", env.n_ue + 1)

                im = ax_grid.imshow(
                    grid_vis.T,     
                    aspect="auto",
                    origin="lower",
                    cmap=cmap,
                    vmin=0,
                    vmax=env.n_ue,
                )

                ax_grid.set_xlabel("TTI (последние)")
                ax_grid.set_ylabel("RBG index")
                ax_grid.set_title("Resource grid: time (X) vs RBG index (Y)")

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
