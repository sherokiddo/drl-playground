import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class DQN(nn.Module):
    """Простая сеть для DQN."""

    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, x):
        return self.net(x)


class DQNAgent:
    """Обёртка над сетью + логика выбора действий и обучения (заглушка)."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        device=None,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay_steps: int = 30_000,
        gamma: float = 0.99,
        lr: float = 1e-3,
    ):
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.q_net = DQN(state_dim, action_dim).to(self.device)
        self.target_q_net = DQN(state_dim, action_dim).to(self.device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.target_q_net.eval()

        self.optimizer  = optim.Adam(self.q_net.parameters(), lr=lr)
        self.gamma      = gamma

        self.epsilon_start          = epsilon_start
        self.epsilon_end            = epsilon_end
        self.epsilon_decay_steps    = epsilon_decay_steps
        self.total_steps            = 0

        self.action_dim             = action_dim

    def _get_epsilon(self) -> float:
        # Линейный decay от epsilon_start до epsilon_end по total_steps

        fraction    = min(self.total_steps / self.epsilon_decay_steps, 1.0)
        epsilon     = self.epsilon_start + fraction * (self.epsilon_end - self.epsilon_start)

        return epsilon
    
    def select_action(self, state):
        """Возвратит действие по ε-greedy политике."""

        epsilon = self._get_epsilon()
        self.total_steps += 1

        # Исследование: с вероятностью epsilon выбираем случайное действие
        if random.random() < epsilon:
            return random.randint(0, self.action_dim - 1)

        # Эксплуатация: действуем для достижения максимальной награды
        state_tensor = torch.as_tensor(state, 
                                       dtype=torch.float32, 
                                       device=self.device
                                       ).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_net(state_tensor)
        action = int(torch.argmax(q_values, dim=1).item())
        
        return action

    def train_step(self, batch):
        """Один шаг обучения по батчу из replay buffer."""
        raise NotImplementedError("DQNAgent.train_step ещё не реализован")