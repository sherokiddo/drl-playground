from envs.lte_scheduler_env import LTESchedulerEnv
import numpy as np

env = LTESchedulerEnv(
    n_ue=3,
    n_rb_dl=50,
    reward_mode="per_tti",
    wb_cqi_report_period_tti=5,
    traffic_profile=("full_buffer", "on_off", "bursty"),
    seed=42,
)
obs, info = env.reset()

print(f"Observation shape: {obs.shape}")  # ожидаем (21,)
print(f"Action space: {env.action_space}")  # Discrete(3)
print(f"Reported CQI:      {info['reported_wb_cqi'].astype(int).tolist()}")
print(f"CQI age:           {info['wb_cqi_age'].astype(int).tolist()}")
print(f"Action mask:       {info['action_mask'].astype(int).tolist()}")
print(f"Alloc frac TTI:    {info['alloc_rbg_frac_tti'].round(3).tolist()}")

done = False
rewards = []
while not done:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if reward != 0.0:
        rewards.append(reward)
    done = terminated or truncated

summary = env.get_episode_summary()
print(f"Средний Throughput: {summary['mean_throughput_mbps']:.3f} Мбит/с")
print(f"Средняя SE:         {summary['mean_se_bps_hz']:.4f} бит/с/Гц")
print(f"Средний JFI:        {summary['mean_jfi']:.4f}")
print(f"Средняя награда:    {summary['mean_reward']:.4f}")
print(f"Min JFI:            {summary['min_jfi']:.4f}")
print(f"Max SE:             {summary['max_se_bps_hz']:.4f} бит/с/Гц")
