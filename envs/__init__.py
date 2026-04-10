from gymnasium.envs.registration import register

register(
    id="FrozenLakeCustom-v0",
    entry_point="envs.frozen_lake_env:FrozenLakeCustom",
)