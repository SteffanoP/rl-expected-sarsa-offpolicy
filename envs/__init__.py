from ._implementations.racetrack_env import RacetrackEnv

import gymnasium as gym

gym.envs.registration.register(
    id="RaceTrack-v0",
    #entry_point="envs:RacetrackEnv",  # Caminho para a classe
    entry_point="envs._implementations.racetrack_env:create_wrapped_racetrack_env",
    max_episode_steps=150,
)