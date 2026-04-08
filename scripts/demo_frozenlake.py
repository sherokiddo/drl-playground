# scripts/demo_frozenlake_random.py
import time

from envs.frozen_lake_env import FrozenLakeCustom


def main():
    env = FrozenLakeCustom(is_slippery=True, render_mode="human")
    state, info = env.reset(seed=0)

    done = False
    step = 0

    while not done:
        step += 1
        action = env.action_space.sample()

        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        print(f"Step {step}: action={action}, reward={reward}, next_state={next_state}")
        time.sleep(0.3)

    env.close()
    print("Episode finished.")

if __name__ == "__main__":
    print("Demo starting...")
    main()
