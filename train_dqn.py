"""
Entraîne un agent DQN sur SimpleRobotEnv.
Architecture minuscule (4 -> 6 -> 4) pour être compatible avec le FPGA.
"""

import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
import torch

from env_simple import SimpleRobotEnv

if __name__ == "__main__":
    # Créer un environnement vectorisé (plus stable)
    env = make_vec_env(SimpleRobotEnv, n_envs=1)

    model = DQN(
        "MlpPolicy",
        env,
        policy_kwargs=dict(net_arch=[6]),  # 1 couche cachée 6 neurones
        learning_rate=1e-3,
        buffer_size=10000,
        learning_starts=100,
        batch_size=64,
        gamma=0.95,
        verbose=1,
        tensorboard_log="./logs/"
    )

    print(" Début de l'entraînement...")
    model.learn(total_timesteps=30000)
    model.save("models/dqn_robot")

    print("Entraînement terminé — modèle sauvegardé.")
    env.close()
