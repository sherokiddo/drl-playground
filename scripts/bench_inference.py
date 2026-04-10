import time
from envs.lte_scheduler_env import LTESchedulerEnv
from stable_baselines3 import DQN

def make_env():
    return LTESchedulerEnv(
        n_ue=3,
        n_rbg=16,
        episode_len=200,
        reward_mode="per_tti",
        reward_window=10,
        bandwidth_hz=10e6,
        traffic_lambda=5000.0,
        seed=123,
    )

if __name__ == "__main__":
    model_path = "runs/dqn_scheduler/dqn_lte_scheduler.zip"

    env = make_env()
    model = DQN.load(model_path, env=env)

    n_tti = 100

    # 0) Прогрев
    obs, info = env.reset()
    for _ in range(10):
        for _ in range(env.n_rbg):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                obs, info = env.reset()

    # 1) ONLY PREDICT:
    obs, info = env.reset()
    start = time.perf_counter()
    for _ in range(n_tti):
        for _ in range(env.n_rbg):
            action, _ = model.predict(obs, deterministic=True)
            # env.step 
    end = time.perf_counter()
    total_predict = end - start
    print(f"Only predict - total: {total_predict*1000:.2f} ms")
    print(f"Only predict - per TTI: {(total_predict/n_tti)*1000:.4f} ms")
    print(f"Only predict - per RBG: {(total_predict/(n_tti*env.n_rbg))*1000:.4f} ms\n")

    # 2) ONLY ENV.STEP: 
    actions = [env.action_space.sample() for _ in range(n_tti * env.n_rbg)]
    obs, info = env.reset()
    k = 0
    start = time.perf_counter()
    for _ in range(n_tti):
        for _ in range(env.n_rbg):
            obs, _, terminated, truncated, info = env.step(actions[k])
            k += 1
            if terminated or truncated:
                obs, info = env.reset()
    end = time.perf_counter()
    total_step = end - start
    print(f"Only env.step - total: {total_step*1000:.2f} ms")
    print(f"Only env.step - per TTI: {(total_step/n_tti)*1000:.4f} ms")
    print(f"Only env.step - per RBG: {(total_step/(n_tti*env.n_rbg))*1000:.4f} ms\n")

    # 3) PREDICT + STEP: 
    obs, info = env.reset()
    start = time.perf_counter()
    for _ in range(n_tti):
        for _ in range(env.n_rbg):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                obs, info = env.reset()
    end = time.perf_counter()
    total_both = end - start
    print(f"Predict+step - total: {total_both*1000:.2f} ms")
    print(f"Predict+step - per TTI: {(total_both/n_tti)*1000:.4f} ms")
    print(f"Predict+step - per RBG: {(total_both/(n_tti*env.n_rbg))*1000:.4f} ms")