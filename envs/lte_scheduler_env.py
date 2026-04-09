"""
#------------------------------------------------------------------------------
# Модуль: RL_ENV - Gymnasium-среда для RL-планировщика LTE
#------------------------------------------------------------------------------
# Описание:
#   Реализует среду Gymnasium для обучения Direct RL-агента (Direct Scheduler Agent),
#   который последовательно назначает каждый RBG одному из N_UE пользователей.
#
#   Архитектура:
#   - Direct Agent: за каждый TTI принимает N_RBG последовательных решений
#     (кому отдать текущий RBG). Действие: Discrete(N_UE).
#   - Observation: вектор признаков per-UE (CQI, буфер, avg_throughput)
#     + контекст текущего RBG + уже выданные в этом TTI ресурсы.
#   - Награда: alpha * log(1 + SE/SE_scale) + beta * JFI
#     Режим: "per_tti" (каждый TTI) или накапливается для "delayed".
#
# Параметры среды:
#   n_ue          : int   = 3       — число пользователей
#   n_rbg         : int   = 16      — число RBG (10 МГц)
#   episode_len   : int   = 200     — длина эпизода в TTI
#   reward_mode   : str   = "delayed" | "per_tti"
#   reward_window : int   = 10      — окно отложенной награды (TTI)
#   alpha         : float = 0.5     — вес SE в награде
#   beta          : float = 0.5     — вес JFI в награде
#   se_scale      : float = 1e6     — масштаб SE (бит/с -> нормировка)
#   cqi_markov    : bool  = True    — использовать марков. цепь CQI
#   traffic_lambda: float = 5000.0  — средний трафик Пуассона (байт/TTI на UE)
#   seed          : int   = None    — зерно генератора
#
# Observation space: Box(shape=(N_UE * 3 + 3,))
#   Per-UE признаки (x3 на UE):
#     [0] cqi_norm        — нормированный CQI [0,1]  (CQI / 15)
#     [1] buffer_norm     — нормированный буфер [0,1] (байты / MAX_BUFFER)
#     [2] avg_tput_norm   — нормированный средний throughput [0,1]
#   Контекст (x3):
#     [N_UE*3 + 0] rbg_idx_norm    — индекс текущего RBG [0,1]
#     [N_UE*3 + 1] tti_norm        — нормированный TTI [0,1]
#     [N_UE*3 + 2] rbg_alloc_norm  — доля уже выданных RBG в TTI [0,1]
#
# Action space: Discrete(N_UE) — индекс UE, которому выдаётся текущий RBG
#
# Версия: 0.1.0
# Дата: 2026-04-08
# Автор: Брагин Кирилл
# Зависимости: gymnasium, numpy
#------------------------------------------------------------------------------
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Dict, Tuple, Any


# ---------------------------------------------------------------------------
# Вспомогательные таблицы LTE
# ---------------------------------------------------------------------------

# CQI -> bits_per_RE: приближение из 3GPP TS 36.213, Table 7.2.3-1
CQI_BITS_PER_RE = {
    1:  0.1523,  2:  0.2344,  3:  0.3770,  4:  0.6016,  5:  0.8770,
    6:  1.1758,  7:  1.4766,  8:  1.9141,  9:  2.4063,  10: 2.7305,
    11: 3.3223,  12: 3.9023,  13: 4.5234,  14: 5.1152,  15: 5.5547,
}

RE_PER_RB_TTI = 132   # RE на RB за 1 TTI (нормальный CP, без управляющих)
RB_PER_RBG    = 3     # 10 МГц: 50 RB -> 16 RBG по 3 RB

# Марковская цепь CQI: P_STAY=0.7, P_STEP=0.15 (вверх/вниз)
_P_STAY, _P_STEP = 0.7, 0.15

def _build_cqi_transition_matrix() -> np.ndarray:
    n = 15
    T = np.zeros((n, n))
    for i in range(n):
        T[i, i] = _P_STAY
        if i > 0:      T[i, i-1] = _P_STEP
        else:          T[i, i]  += _P_STEP   # граница: отражение
        if i < n - 1:  T[i, i+1] = _P_STEP
        else:          T[i, i]  += _P_STEP   # граница: отражение
    return T / T.sum(axis=1, keepdims=True)

CQI_TRANSITION  = _build_cqi_transition_matrix()
MAX_BUFFER_BYTES = 1_000_000   # 1 МБ
MAX_AVG_TPUT     = 100e6       # 100 Мбит/с


# ---------------------------------------------------------------------------
# Среда Gymnasium
# ---------------------------------------------------------------------------

class LTESchedulerEnv(gym.Env):
    """
    Gymnasium-среда для обучения Direct RL-планировщика LTE.

    Каждый step() — одно решение агента: кому выдать текущий RBG.
    После N_RBG шагов завершается TTI: обновляются буферы, CQI, считается награда.
    После episode_len TTI — эпизод завершён.

    Пример использования:
    ---------------------
    >>> env = LTESchedulerEnv(n_ue=3, reward_mode="delayed")
    >>> obs, info = env.reset(seed=42)
    >>> done = False
    >>> while not done:
    ...     action = env.action_space.sample()
    ...     obs, reward, terminated, truncated, info = env.step(action)
    ...     done = terminated or truncated
    >>> print(env.get_episode_summary())
    """

    metadata = {"render_modes": []}
    N_UE_FEATURES      = 3
    N_CONTEXT_FEATURES = 3

    def __init__(
        self,
        n_ue:           int   = 40,
        n_rbg:          int   = 16,
        episode_len:    int   = 500,
        reward_mode:    str   = "delayed",
        reward_window:  int   = 10,
        alpha:          float = 0.5,
        beta:           float = 0.5,
        bandwidth_hz:   float = 10e6,
        cqi_markov:     bool  = True,
        traffic_lambda: float = 5000.0,
        rate_scale_bps: float = 1e6,
        jfi_target:     float = 0.7,   
        lambda_jfi:     float = 1.0, 
        seed:           Optional[int] = None,
        ):
        super().__init__()

        assert reward_mode in ("per_tti", "delayed"), \
            "reward_mode должен быть 'per_tti' или 'delayed'"
        assert n_ue >= 1, "n_ue должен быть >= 1"

        self.n_ue           = n_ue
        self.n_rbg          = n_rbg
        self.episode_len    = episode_len
        self.reward_mode    = reward_mode
        self.reward_window  = reward_window
        self.alpha          = alpha
        self.beta           = beta
        self.bandwidth_hz   = bandwidth_hz
        self.cqi_markov     = cqi_markov
        self.traffic_lambda = traffic_lambda
        self.rate_scale_bps = rate_scale_bps
        self.jfi_target     = jfi_target 
        self.lambda_jfi     = lambda_jfi
        self._last_rbg_alloc = None
        self.grid_history   = [] 

        obs_dim = n_ue * self.N_UE_FEATURES + self.N_CONTEXT_FEATURES
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(n_ue)

        self._rng: np.random.Generator = np.random.default_rng(seed)
        self._cqi:         np.ndarray  = None
        self._buffer:      np.ndarray  = None
        self._avg_tput:    np.ndarray  = None
        self._rbg_alloc:   np.ndarray  = None
        self._current_rbg: int         = 0
        self._current_tti: int         = 0
        self._reward_accum: float      = 0.0
        self._window_count: int        = 0
        self._ema_alpha = 0.01  # EMA ~100 TTI

        self.history: Dict[str, list] = {"throughput": [], "se": [], "jfi": [], "reward": []}

    # ------------------------------------------------------------------
    # reset
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._cqi      = self._rng.integers(4, 13, size=self.n_ue)
        self._buffer   = self._rng.uniform(10_000, 50_000, size=self.n_ue)
        self._avg_tput = np.zeros(self.n_ue, dtype=np.float64)

        self._rbg_alloc   = np.full(self.n_rbg, -1, dtype=np.int32)
        self._current_rbg = 0
        self._current_tti = 0
        self._reward_accum = 0.0
        self._window_count = 0

        self._last_rbg_alloc = None
        self.grid_history = []
        self.history = {"throughput": [], "se": [], "jfi": [], "reward": []}
        return self._get_obs(), self._get_info()

    # ------------------------------------------------------------------
    # step
    # ------------------------------------------------------------------

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        assert self.action_space.contains(action), \
            f"Недопустимое действие: {action}"

        self._rbg_alloc[self._current_rbg] = action
        self._current_rbg += 1

        reward     = 0.0
        terminated = False
        truncated  = False

        if self._current_rbg >= self.n_rbg:
            reward = self._process_tti()
            self._current_tti += 1

            self._last_rbg_alloc = self._rbg_alloc.copy()
            self.grid_history.append(self._last_rbg_alloc.copy())

            self._current_rbg = 0
            self._rbg_alloc[:] = -1

            if self._current_tti >= self.episode_len:
                terminated = True

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    # ------------------------------------------------------------------
    # Обработка TTI
    # ------------------------------------------------------------------

    def _process_tti(self) -> float:
        bits = self._compute_bits_delivered()
        self._update_buffers(bits)

        prev_avg_tput = self._avg_tput.copy()
        tti_tput_bps = bits * 1000.0
        self._avg_tput = (
            (1.0 - self._ema_alpha) * self._avg_tput
            + self._ema_alpha * tti_tput_bps
        )

        if self.cqi_markov:
            self._update_cqi()

        throughput_bps = float(np.sum(tti_tput_bps))
        se_bps_hz      = throughput_bps / self.bandwidth_hz
        jfi            = self._compute_jfi(self._avg_tput)

        self.history["throughput"].append(throughput_bps / 1e6)  # Мбит/с
        self.history["se"].append(se_bps_hz)
        self.history["jfi"].append(jfi)

        tti_reward = self._compute_reward(
            se_bps_hz=se_bps_hz,
            jfi=jfi,
            prev_avg_tput=prev_avg_tput,
            tti_tput_bps=tti_tput_bps,
        )
        if self.reward_mode == "per_tti":
            self.history["reward"].append(tti_reward)
            return tti_reward

        # delayed: накапливаем, отдаём раз в reward_window TTI
        self._reward_accum += tti_reward
        self._window_count += 1
        if self._window_count >= self.reward_window:
            avg_r = self._reward_accum / self.reward_window
            self.history["reward"].append(avg_r)
            self._reward_accum = 0.0
            self._window_count = 0
            return avg_r
        return 0.0

    # ------------------------------------------------------------------
    # Вычисление переданных бит
    # ------------------------------------------------------------------

    def _compute_bits_delivered(self) -> np.ndarray:
        bits = np.zeros(self.n_ue, dtype=np.float64)
        for i in range(self.n_ue):
            n_rbg = int(np.sum(self._rbg_alloc == i))
            if n_rbg == 0:
                continue
            bpre     = CQI_BITS_PER_RE.get(int(self._cqi[i]), 0.0)
            raw_bits = n_rbg * RB_PER_RBG * RE_PER_RB_TTI * bpre
            buf_bits = self._buffer[i] * 8.0
            bits[i]  = min(raw_bits, buf_bits)  # ограничение буфером
        return bits

    # ------------------------------------------------------------------
    # Обновление буферов (Пуассон)
    # ------------------------------------------------------------------

    def _update_buffers(self, bits_delivered: np.ndarray) -> None:
        bytes_out = bits_delivered / 8.0
        new_bytes  = self._rng.poisson(
            self.traffic_lambda, size=self.n_ue
        ).astype(np.float64)
        self._buffer = np.clip(
            self._buffer - bytes_out + new_bytes,
            0.0, MAX_BUFFER_BYTES,
        )

    # ------------------------------------------------------------------
    # Обновление CQI (Марков)
    # ------------------------------------------------------------------

    def _update_cqi(self) -> None:
        for i in range(self.n_ue):
            cqi_idx     = int(self._cqi[i]) - 1
            self._cqi[i] = self._rng.choice(15, p=CQI_TRANSITION[cqi_idx]) + 1

    # ------------------------------------------------------------------
    # Метрики
    # ------------------------------------------------------------------

    def _compute_jfi(self, bits: np.ndarray) -> float:
        s  = float(np.sum(bits))
        ss = float(np.sum(bits ** 2))
        if ss < 1e-12:
            return 0.0
        return (s ** 2) / (self.n_ue * ss)

    def _compute_reward(
        self,
        se_bps_hz: float,
        jfi: float,
        prev_avg_tput: np.ndarray,
        tti_tput_bps: np.ndarray,
    ) -> float:
        rate_scale_bps = max(float(self.rate_scale_bps), 1.0)
        prev_u = np.mean(np.log1p(prev_avg_tput / rate_scale_bps))
        curr_u = np.mean(np.log1p(self._avg_tput / rate_scale_bps))
        pf_delta = curr_u - prev_u

        se_term = np.log1p(se_bps_hz)
        jfi_penalty = self.lambda_jfi * max(0.0, self.jfi_target - jfi) ** 2

        return float(self.alpha * se_term + self.beta * pf_delta - jfi_penalty)

    # ------------------------------------------------------------------
    # Observation / Info
    # ------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        obs = np.zeros(
            self.n_ue * self.N_UE_FEATURES + self.N_CONTEXT_FEATURES,
            dtype=np.float32,
        )
        for i in range(self.n_ue):
            b = i * self.N_UE_FEATURES
            obs[b + 0] = float(self._cqi[i])      / 15.0
            obs[b + 1] = float(self._buffer[i])   / MAX_BUFFER_BYTES
            obs[b + 2] = float(self._avg_tput[i]) / MAX_AVG_TPUT

        c = self.n_ue * self.N_UE_FEATURES
        obs[c + 0] = self._current_rbg / self.n_rbg
        obs[c + 1] = self._current_tti / max(self.episode_len, 1)
        obs[c + 2] = np.sum(self._rbg_alloc >= 0) / self.n_rbg

        return np.clip(obs, 0.0, 1.0)

    def _get_info(self) -> Dict[str, Any]:
        return {
            "tti":          self._current_tti,
            "rbg_step":     self._current_rbg,
            "cqi":          self._cqi.copy(),
            "buffer_bytes": self._buffer.copy(),
            "avg_tput_bps": self._avg_tput.copy(),
            "rbg_alloc":    self._rbg_alloc.copy(),
        }

    # ------------------------------------------------------------------
    # Сводка по эпизоду
    # ------------------------------------------------------------------

    def get_episode_summary(self) -> Dict[str, float]:
        """Возвращает агрегированные метрики за эпизод."""
        if not self.history["se"]:
            return {}
        tput = np.array(self.history["throughput"])
        se   = np.array(self.history["se"])
        jfi  = np.array(self.history["jfi"])
        rwd  = np.array(self.history["reward"])

        return {
            "mean_throughput_mbps": float(np.mean(tput))      if len(tput) > 0 else 0.0,
            "mean_se_bps_hz":       float(np.mean(se))        if len(se)   > 0 else 0.0,
            "mean_jfi":             float(np.mean(jfi))       if len(jfi)  > 0 else 0.0,
            "mean_reward":          float(np.mean(rwd))       if len(rwd)  > 0 else 0.0,
            "min_jfi":              float(np.min(jfi))        if len(jfi)  > 0 else 0.0,
            "max_se_bps_hz":        float(np.max(se))         if len(se)   > 0 else 0.0,
        }
    
    def render(self, mode: str = "human") -> None:
        """
        Простая визуализация по TTI:
        - печатает, сколько RBG получил каждый UE в последнем TTI,
        - выводит текущие avg_tput и JFI.
        Вызывается ПОСЛЕ завершения TTI (когда _current_rbg снова 0).
        """
        if mode != "human":
            return

        tti = self._current_tti - 1

        print(f"\n[RENDER] TTI {tti}")
        print(f"  CQI:      {self._cqi.astype(int).tolist()}")
        print(f"  BufferKB: {[round(b/1024, 1) for b in self._buffer]}")
        print(f"  AvgTput:  {[round(r/1e6, 3) for r in self._avg_tput]} Мбит/с")
        if self.history["jfi"]:
            print(f"  JFI(avg): {self.history['jfi'][-1]:.3f}")
        if self.history["se"]:
            print(f"  SE:       {self.history['se'][-1]:.3f} бит/с/Гц")
        if self._last_rbg_alloc is not None:
            counts = [int(np.sum(self._last_rbg_alloc == i)) for i in range(self.n_ue)]
            print(f"  RBG per UE: {counts} (из {self.n_rbg})")     
