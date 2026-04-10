import time
import numpy as np

from agents.lte_dqn_agent import LTEDQNAgent
from envs.lte_padded_env import PaddedLTESchedulerEnv
from envs.lte_scheduler_env import LTESchedulerEnv


MODEL_PATH = "runs/lte_dqn/lte_dqn_shared_q.pt"
MAX_N_UE = 40
EVAL_N_UE = 40


def make_base_env(n_ue: int, seed: int | None = None) -> LTESchedulerEnv:
    return LTESchedulerEnv(
        n_ue=n_ue,
        n_rb_dl=50,
        episode_len=200,
        reward_mode="per_tti",
        reward_window=10,
        wb_cqi_report_period_tti=5,
        traffic_lambda=5000.0,
        traffic_profile="on_off",
        seed=seed,
    )


def make_env(n_ue: int, max_n_ue: int, seed: int | None = None) -> PaddedLTESchedulerEnv:
    return PaddedLTESchedulerEnv(make_base_env(n_ue=n_ue, seed=seed), max_n_ue=max_n_ue)


def sample_valid_action(action_mask: np.ndarray) -> int:
    valid = np.flatnonzero(np.asarray(action_mask, dtype=bool))
    if len(valid) == 0:
        return 0
    return int(np.random.choice(valid))


if __name__ == "__main__":
    env = make_env(n_ue=EVAL_N_UE, max_n_ue=MAX_N_UE, seed=123)
    agent = LTEDQNAgent.load(MODEL_PATH)

    print(
        f"[INFO] Benchmark mode: custom LTE DQN "
        f"(env {EVAL_N_UE} UE, padded to {MAX_N_UE})"
    )

    n_tti = 100

    obs, info = env.reset()
    for _ in range(10):
        for _ in range(env.n_rbg):
            action = agent.predict(obs, info["action_mask"], deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                obs, info = env.reset()

    obs, info = env.reset()
    start = time.perf_counter()
    for _ in range(n_tti):
        for _ in range(env.n_rbg):
            _ = agent.predict(obs, info["action_mask"], deterministic=True)
    end = time.perf_counter()
    total_predict = end - start
    print(f"Only predict - total: {total_predict*1000:.2f} ms")
    print(f"Only predict - per TTI: {(total_predict/n_tti)*1000:.4f} ms")
    print(f"Only predict - per RBG: {(total_predict/(n_tti*env.n_rbg))*1000:.4f} ms\n")

    obs, info = env.reset()
    start = time.perf_counter()
    for _ in range(n_tti):
        for _ in range(env.n_rbg):
            action = sample_valid_action(info["action_mask"])
            obs, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                obs, info = env.reset()
    end = time.perf_counter()
    total_step = end - start
    print(f"Only env.step - total: {total_step*1000:.2f} ms")
    print(f"Only env.step - per TTI: {(total_step/n_tti)*1000:.4f} ms")
    print(f"Only env.step - per RBG: {(total_step/(n_tti*env.n_rbg))*1000:.4f} ms\n")

    obs, info = env.reset()
    start = time.perf_counter()
    for _ in range(n_tti):
        for _ in range(env.n_rbg):
            action = agent.predict(obs, info["action_mask"], deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                obs, info = env.reset()
    end = time.perf_counter()
    total_both = end - start
    print(f"Predict+step - total: {total_both*1000:.2f} ms")
    print(f"Predict+step - per TTI: {(total_both/n_tti)*1000:.4f} ms")
    print(f"Predict+step - per RBG: {(total_both/(n_tti*env.n_rbg))*1000:.4f} ms")
