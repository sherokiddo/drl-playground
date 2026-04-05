import gymnasium as gym
import torch

from agents.dqn_agent import DQNAgent
from utils.replay_buffer import ReplayBuffer

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    env = gym.make("CartPole-v1")
    state, _ = env.reset()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DQNAgent(state_dim, action_dim, device=device)
    buffer = ReplayBuffer(capacity=10000)

    print("Скелет DQN-обучения готов, дальше добавляем логику.")

if __name__ == "__main__":
    main()