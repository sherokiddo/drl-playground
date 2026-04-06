import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class ActorCritic(nn.Module):
    """
    Единая сеть с общим backbone и двумя головами:
      - actor  → логиты для распределения над действиями
      - critic → скалярная оценка V(s)
    """

    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
        )
        self.actor  = nn.Linear(64, action_dim)
        self.critic = nn.Linear(64, 1)

        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)

    def forward(self, x: torch.Tensor):
        """
        Прямой проход: state → (logits, value)

        x: тензор состояния, shape [batch, state_dim]
        Возвращает:
          logits: shape [batch, action_dim]  — для актора
          value:  shape [batch, 1]           — для критика
        """
        features    = self.shared(x)
        logits      = self.actor(features)
        value       = self.critic(features)
        return logits, value

    def get_action(self, state: torch.Tensor):
        """
        Семплируем действие из текущей политики.

        Возвращает:
          action:   int — выбранное действие
          log_prob: тензор [1] — логарифм вероятности этого действия
          value:    тензор [1] — оценка текущего состояния критиком
          entropy:  тензор [1] — энтропия распределения (для лосса)
        """
        logits, value = self.forward(state)

        dist = torch.distributions.Categorical(logits=logits)

        action   = dist.sample()          # сэмплируем одно действие
        log_prob = dist.log_prob(action)  # log π(a|s) — нужен для PPO loss
        entropy  = dist.entropy()         # H[π] — нужна для entropy bonus

        return action.item(), log_prob, value.squeeze(-1), entropy

class PPOAgent:
    """
    PPO агент: управляет сбором траекторий и обновлением политики.

    Гиперпараметры:
      lr            — learning rate для обоих компонентов
      gamma         — дисконт-фактор для будущих наград
      gae_lambda    — λ для Generalized Advantage Estimation
      clip_epsilon  — ε для PPO clipping (обычно 0.1–0.2)
      value_coef    — вес critic loss в суммарном лоссе
      entropy_coef  — вес entropy bonus (поощряет исследование)
      n_epochs      — сколько эпох градиентных шагов по одному роллауту
      batch_size    — размер мини-батча внутри эпохи
    """

    def __init__(
        self,
        state_dim:    int,
        action_dim:   int,
        device=None,
        lr:           float = 3e-4,
        gamma:        float = 0.95,
        gae_lambda:   float = 0.95,
        clip_epsilon: float = 0.2,
        value_coef:   float = 1.0,
        entropy_coef: float = 0.01,
        n_epochs:     int   = 10,
        batch_size:   int   = 64,
    ):
        self.device       = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma        = gamma
        self.gae_lambda   = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef   = value_coef
        self.entropy_coef = entropy_coef
        self.n_epochs     = n_epochs
        self.batch_size   = batch_size

        # Одна сеть — и актор, и критик
        self.ac        = ActorCritic(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.ac.parameters(), lr=lr, eps=1e-5)

    def select_action(self, state: np.ndarray):
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action, log_prob, value, _ = self.ac.get_action(state_tensor)
        return action, log_prob.squeeze(), value.squeeze()

    def compute_gae(
        self,
        rewards:    torch.Tensor,
        values:     torch.Tensor,
        dones:      torch.Tensor,
        last_value: torch.Tensor,
    ):
        """
        Generalized Advantage Estimation (GAE).

        GAE сглаживает advantage между двумя крайностями:
          λ=0 → TD(0): A_t = r_t + γV(s_{t+1}) - V(s_t)  (низкая дисперсия, высокое смещение)
          λ=1 → MC:    A_t = G_t - V(s_t)                 (высокая дисперсия, нет смещения)
        λ=0.95 — хороший компромисс.

        Алгоритм считает advantages с конца траектории (backward pass):
          δ_t     = r_t + γ * V(s_{t+1}) * (1 - done_t) - V(s_t)
          A_t     = δ_t + γ * λ * (1 - done_t) * A_{t+1}
        """
        T          = len(rewards)
        advantages = torch.zeros(T, device=self.device)
        last_gae   = 0.0

        for t in reversed(range(T)):
            next_value  = last_value if t == T - 1 else values[t + 1]
            next_non_terminal = 1.0 - dones[t]

            # TD-ошибка: насколько оценка критика ошиблась
            delta    = rewards[t] + self.gamma * next_value * next_non_terminal - values[t]

            # GAE: рекуррентно накапливаем
            last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
            advantages[t] = last_gae

        # Returns = advantages + values (это таргеты для критика)
        returns = advantages + values

        # Нормализуем returns для стабильности critic loss
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        return advantages, returns

    def update(self, buffer, last_value: torch.Tensor):
        """
        Обновление политики по собранному роллауту.

        Шаги:
          1. Достаём данные из буфера
          2. Считаем GAE advantages и returns
          3. Нормализуем advantages
          4. n_epochs раз проходим по мини-батчам, считаем лосс, делаем шаг
        """
        states, actions, rewards, dones, old_log_probs, values = buffer.get_tensors(self.device)

        values       = values.squeeze(-1)
        old_log_probs = old_log_probs.squeeze(-1)

        # Считаем advantages и returns
        with torch.no_grad():
            advantages, returns = self.compute_gae(rewards, values, dones, last_value.squeeze())

        # Нормализуем advantages: mean=0, std=1
        # стабилизирует обучение
        # убирает зависимость от масштаба наград
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_size = len(states)

        for _ in range(self.n_epochs):

            # Перемешиваем индексы — делаем мини-батчи случайными
            indices = torch.randperm(total_size, device=self.device)

            for start in range(0, total_size, self.batch_size):
                end  = start + self.batch_size
                idx  = indices[start:end]

                mb_states     = states[idx]
                mb_actions    = actions[idx]
                mb_advantages = advantages[idx]
                mb_returns    = returns[idx]
                mb_old_lp     = old_log_probs[idx]

                # Forward pass через текущую (обновляемую) политику
                logits, values_new = self.ac(mb_states)
                dist               = torch.distributions.Categorical(logits=logits)
                new_log_probs      = dist.log_prob(mb_actions)
                entropy            = dist.entropy()

                # PPO ratio: π_new(a|s) / π_old(a|s)
                # В логарифмах: exp(log π_new - log π_old)
                ratio = torch.exp(new_log_probs - mb_old_lp)

                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * mb_advantages
                actor_loss = -torch.min(surr1, surr2).mean()

                critic_loss = nn.functional.mse_loss(values_new.squeeze(-1), mb_returns)
                entropy_loss = -entropy.mean()
                loss = actor_loss + self.value_coef * critic_loss + self.entropy_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                # ограничиваем норму градиентов
                # предотвращает взрывной градиент
                nn.utils.clip_grad_norm_(self.ac.parameters(), max_norm=0.5)
                self.optimizer.step()

        return loss.item()