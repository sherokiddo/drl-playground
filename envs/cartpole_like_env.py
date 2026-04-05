import gymnasium as gym
from .base_env import BaseEnv

class CartPoleLikeEnv(BaseEnv):
    """Учебная CartPole-подобная среда (пока заглушка)."""

    def __init__(self):
        super().__init__()

    def reset(self, *, seed=None, options=None):
        raise NotImplementedError("CartPoleLikeEnv.reset ещё не реализован")

    def step(self, action):
        raise NotImplementedError("CartPoleLikeEnv.step ещё не реализован")