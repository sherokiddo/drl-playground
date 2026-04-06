import numpy as np
import torch

class RolloutBuffer:
    """
    Буфер для хранения одного батча траекторий (роллаута) в PPO.

    В отличие от ReplayBuffer (DQN), этот буфер:
      - НЕ случайный: данные хранятся в порядке сбора
      - Очищается полностью после каждого обновления политики
      - Хранит дополнительно log_prob и value — нужны для PPO loss

    Поля на каждый шаг t:
      states    — состояние s_t
      actions   — действие a_t
      rewards   — награда r_t
      dones     — флаг окончания эпизода
      log_probs — log π_old(a_t|s_t) на момент сбора
      values    — V(s_t) от критика на момент сбора
    """

    def __init__(self):
        self.states    = []
        self.actions   = []
        self.rewards   = []
        self.dones     = []
        self.log_probs = []
        self.values    = []

    def push(self, state, action, reward, done, log_prob, value):
        """Добавляем один шаг в буфер."""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)   

    def clear(self):
        """Очищаем буфер после обновления политики."""
        self.states    = []
        self.actions   = []
        self.rewards   = []
        self.dones     = []
        self.log_probs = []
        self.values    = []

    def __len__(self):
        return len(self.states)

    def get_tensors(self, device):
        """
        Преобразуем накопленные данные в тензоры для обучения.
        Возвращает все поля как тензоры на нужном устройстве.
        """
        states    = torch.as_tensor(np.array(self.states),   dtype=torch.float32, device=device)
        actions   = torch.as_tensor(np.array(self.actions),  dtype=torch.int64,   device=device)
        rewards   = torch.as_tensor(np.array(self.rewards),  dtype=torch.float32, device=device)
        dones     = torch.as_tensor(np.array(self.dones),    dtype=torch.float32, device=device)
        log_probs = torch.stack(self.log_probs).to(device) 
        values    = torch.stack(self.values).to(device)

        return states, actions, rewards, dones, log_probs, values