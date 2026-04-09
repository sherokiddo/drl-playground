import time
import gymnasium as gym


def main():
    env = gym.make(
        "FrozenLake-v1",
        is_slippery=True,
        render_mode="human",
        map_name="4x4",
    )

    for episode in range(3):
        state, info = env.reset()
        done = False
        step = 0
        print(f"\n=== Episode {episode+1} ===")

        while not done:
            step += 1
            action = env.action_space.sample()
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            print(f"step={step}, action={action}, reward={reward}")
            time.sleep(0.2)  # замедляем, чтобы видеть анимацию

    env.close()


if __name__ == "__main__":
    main()