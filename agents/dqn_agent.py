import torch
import torch.nn as nn
import torch.optim as optim

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

    def __init__(self, state_dim: int, action_dim: int, device=None):
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.q_net = DQN(state_dim, action_dim).to(self.device)
        self.target_q_net = DQN(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=1e-3)

    def select_action(self, state):
        """Возвратит действие по ε-greedy (пока заглушка)."""
        raise NotImplementedError("DQNAgent.select_action ещё не реализован")

    def train_step(self, batch):
        """Один шаг обучения по батчу из replay buffer."""
        raise NotImplementedError("DQNAgent.train_step ещё не реализован")