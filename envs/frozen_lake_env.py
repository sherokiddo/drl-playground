from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Tuple, Dict, Any


LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

ACTION_NAMES = {
    LEFT:  "Left",
    DOWN:  "Down",
    RIGHT: "Right",
    UP:    "Up",
}


class FrozenLakeCustom(gym.Env):
    """
    Минимальная реализация FrozenLake-4x4 под Gymnasium.

    Карта:
        S F F F
        F H F H
        F F F H
        H F F G

    Легенда:
        S – старт
        F – безопасный лёд (reward=0)
        H – провал (reward=0, terminated=True)
        G – цель (reward=1, terminated=True)

    observation_space: Discrete(16) – индекс клетки 0..15
    action_space:      Discrete(4)  – LEFT, DOWN, RIGHT, UP
    """

    metadata = {"render_modes": ["human", "ansi"], "render_fps": 4}

    def __init__(self, is_slippery: bool = True, render_mode: Optional[str] = None):
        super().__init__()

        self.is_slippery = is_slippery
        self.render_mode = render_mode

        # 4x4 карта в виде списка строк
        self.desc = np.asarray(
            [
                list("SFFF"),
                list("FHFH"),
                list("FFFH"),
                list("HFFG"),
            ],
            dtype="c",
        )

        self.nrow, self.ncol = self.desc.shape
        self.observation_space = spaces.Discrete(self.nrow * self.ncol)
        self.action_space      = spaces.Discrete(4)
        self.s: int = 0

    # -------------------------
    # Вспомогательные функции
    # -------------------------

    def _to_row_col(self, s: int) -> Tuple[int, int]:
        """Преобразование индекса состояния в (row, col)."""
        row = s // self.ncol
        col = s % self.ncol
        return int(row), int(col)

    def _to_state(self, row: int, col: int) -> int:
        """Преобразование (row, col) обратно в индекс состояния."""
        return int(row * self.ncol + col)

    def _is_terminal_cell(self, row: int, col: int) -> bool:
        """Проверяем, является ли клетка ямой (H) или целью (G)."""
        cell = self.desc[row, col].decode("utf-8")
        return cell in ("H", "G")

    # -------------------------
    # API Gymnasium
    # -------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[int, Dict[str, Any]]:
        """
        Сбрасываем эпизод.

        Возвращаем:
          obs:  индекс стартовой клетки (S)
          info: пустой словарь (на будущее)
        """
        super().reset(seed=seed)

        # Ищем клетку 'S' на карте
        start_positions = np.argwhere(self.desc == b"S")
        assert len(start_positions) == 1, "Карта должна иметь ровно одну S"
        row, col = start_positions[0]
        self.s   = self._to_state(int(row), int(col))

        return self.s, {}

    def step(self, action: int):
        """
        Один шаг среды.

        Логика:
          1. Выбираем фактическое направление (учитывая is_slippery).
          2. Двигаемся по решётке, не выходя за границы.
          3. Смотрим в какую клетку пришли: F / H / G.
          4. Возвращаем (next_state, reward, terminated, truncated, info).
        """
        assert self.action_space.contains(action), "Неверное действие"

        row, col = self._to_row_col(self.s)

        # 1. Выбор направления с учётом скользкости
        if self.is_slippery:
            # На льду можно "промахнуться" влево/вправо от желаемого направления
            # Возможные действия: {action-1, action, action+1} по модулю 4
            actual_action = self.np_random.choice(
                [ (action - 1) % 4, action, (action + 1) % 4 ]
            )
        else:
            actual_action = action

        # 2. Двигаемся по фактическому направлению
        new_row, new_col = row, col
        if actual_action == LEFT:
            new_col = max(col - 1, 0)
        elif actual_action == RIGHT:
            new_col = min(col + 1, self.ncol - 1)
        elif actual_action == UP:
            new_row = max(row - 1, 0)
        elif actual_action == DOWN:
            new_row = min(row + 1, self.nrow - 1)

        # 3. Смотрим, куда попали
        new_state = self._to_state(new_row, new_col)
        cell      = self.desc[new_row, new_col].decode("utf-8")

        terminated = cell in ("H", "G")
        truncated  = False   # здесь не вводим лимит по шагам

        if cell == "G":
            reward = 1.0
        else:
            reward = 0.0

        self.s = new_state

        info = {}

        if self.render_mode == "human":
            self.render()

        return new_state, reward, terminated, truncated, info

    # -------------------------
    # Рендеринг (текстовый)
    # -------------------------

    def render(self):
        """Простейший текстовый рендер."""
        row, col = self._to_row_col(self.s)

        desc = self.desc.astype("U1")
        grid = desc.tolist()
        grid[row][col] = "A" 

        lines = [" ".join(r) for r in grid]
        text  = "\n".join(lines)

        if self.render_mode == "human":
            print(text)
        else: 
            return text