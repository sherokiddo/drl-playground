import random

import gymnasium as gym
import torch

from agents.dqn_agent import DQNAgent
from utils.replay_buffer import ReplayBuffer

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    env         = gym.make("CartPole-v1")
    state, _    = env.reset(seed=0)
    state_dim   = env.observation_space.shape[0]
    action_dim  = env.action_space.n

    agent   = DQNAgent(state_dim, action_dim, device=device)
    buffer  = ReplayBuffer(capacity=10000)

    num_episodes        = 300
    batch_size          = 64
    learning_starts     = 1000
    target_update_freq  = 1000
    episode_rewards    = []
    global_step         = 0

    for episode in range(num_episodes):
        state, _        = env.reset()
        done            = False
        total_reward    = 0.0

        while not done:

            # Выбираем действие
            global_step     += 1
            action          = agent.select_action(state)

            # Делаем шаг в среде
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Кладем переход в буфер
            buffer.push(state, action, reward, next_state, done)

            state           = next_state
            total_reward    += reward

            # Обучение DQN если достаточно данных
            if len(buffer) >= learning_starts:
                batch   = buffer.sample(batch_size)
                loss    = agent.train_step(batch)

            # Периодически обновляем таргет-сеть
            if global_step % target_update_freq == 0:
                agent.target_q_net.load_state_dict(agent.q_net.state_dict())

        episode_rewards.append(total_reward)

        if (episode + 1) % 10 == 0:
            avg_reward = sum(episode_rewards[-10:]) / 10
            print(f"Episode {episode+1}, avg_reward (last 10): {avg_reward:.1f}")

    print("Обучение DQN на CartPole завершено (базовая версия).")

if __name__ == "__main__":
    main()