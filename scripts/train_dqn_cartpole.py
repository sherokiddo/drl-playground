import gymnasium as gym
import torch

from agents.dqn_agent import DQNAgent
from utils.replay_buffer import ReplayBuffer

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    env = gym.make("CartPole-v1")
    state, _    = env.reset(seed=0)
    state_dim   = env.observation_space.shape[0]
    action_dim  = env.action_space.n

    agent   = DQNAgent(state_dim, action_dim, device=device)
    buffer  = ReplayBuffer(capacity=10000)

    num_episodes = 5

    for episode in range(num_episodes):
        state, _        = env.reset()
        done            = False
        total_reward    = 0.0

        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            buffer.push(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward

        print(f"Episode {episode}, total reward: {total_reward}")

    print("Проверка цикла с epsilon-greedy завершена. Без обучения.")

if __name__ == "__main__":
    main()