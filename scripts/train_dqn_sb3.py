from envs.lte_scheduler_env import LTESchedulerEnv
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
import os
import csv

def make_env():
    env = LTESchedulerEnv(
        n_ue            =3,
        n_rb_dl         =50,
        episode_len     =500,
        reward_mode     ="per_tti",
        reward_window   =1,
        alpha           =1.0,
        beta            =2.0,
        wb_cqi_report_period_tti = 5,
        traffic_lambda  =5000.0,
        rate_scale_bps  =1e6,
        jfi_target      =0.70,
        lambda_jfi      =2.0,
        seed            =0,
        traffic_profile =("full_buffer", "on_off", "bursty"),
    )
    return env

if __name__ == "__main__":
    log_dir = "runs/dqn_scheduler"
    os.makedirs(log_dir, exist_ok=True)

    env = make_env()
    env = Monitor(env)

    logger = configure(log_dir, ["stdout", "tensorboard"])

    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        buffer_size=100_000,
        learning_starts=5_000,
        batch_size=128,
        train_freq=17,
        gamma=0.995,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.10,
        exploration_fraction=0.5,
        target_update_interval=2_000,
        verbose=1,
        tensorboard_log=log_dir,
    )
    model.set_logger(logger)

    total_timesteps = 400_000 
    model.learn(total_timesteps=total_timesteps)

    model.save(os.path.join(log_dir, "dqn_lte_scheduler"))
    env.close()

    # Прогон evaluation-эпизода
    eval_env = make_env()
    obs, info = eval_env.reset(seed=123)
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = eval_env.step(action)
        done = terminated or truncated

    summary = eval_env.get_episode_summary()
    print("\n=== Evaluation (DQN) ===")
    for k, v in summary.items():
        print(f"{k}: {v:.4f}")

    history = eval_env.history 
    with open("runs/dqn_scheduler/eval_history.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["tti", "throughput_mbps", "se_bps_hz", "jfi"])
        for tti, (tput, se, jfi) in enumerate(
            zip(history["throughput"], history["se"], history["jfi"])
        ):
            writer.writerow([tti, tput, se, jfi])
