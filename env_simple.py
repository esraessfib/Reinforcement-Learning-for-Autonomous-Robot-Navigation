"""
Environnement Gym simplifié pour un robot à 4 capteurs.
Observation : [front, left, right, rear]  (valeurs normalisées 0..1)
Action space : 0=STOP, 1=FORWARD, 2=LEFT, 3=RIGHT
Reward :
 - +1 si le robot avance sans collision
 - -1 si obstacle détecté à l'avant
 - petite pénalité par pas pour encourager efficacité
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np

class SimpleRobotEnv(gym.Env):
    def __init__(self):
        super().__init__()
        # Observation : 4 distances normalisées (0..1)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(4,), dtype=np.float32)
        # Action : 4 actions discrètes
        self.action_space = spaces.Discrete(4)
        self.state = np.zeros(4, dtype=np.float32)
        self.step_count = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Génère un état aléatoire (distances initiales)
        self.state = np.random.rand(4).astype(np.float32)
        self.step_count = 0
        return self.state, {}

    def step(self, action):
        front, left, right, rear = self.state
        reward = 0
        terminated = False

        # logiques simples :
        if action == 0:  # STOP
            reward = -0.1
        elif action == 1:  # FORWARD
            if front > 0.3:
                reward = +1.0
                # simulation : on avance, obstacle plus proche
                self.state[0] = max(0.0, front - 0.1)
            else:
                reward = -1.0
                terminated = True
        elif action == 2:  # LEFT
            reward = 0.2
            self.state = np.roll(self.state, 1)  # rotation gauche
        elif action == 3:  # RIGHT
            reward = 0.2
            self.state = np.roll(self.state, -1)  # rotation droite

        # bruit et limite d’épisodes
        self.state = np.clip(self.state + np.random.normal(0, 0.02, 4), 0, 1)
        self.step_count += 1
        if self.step_count > 50:
            terminated = True

        return self.state, reward, terminated, False, {}

    def render(self):
        print(f"State: {self.state}")

if __name__ == "__main__":
    env = SimpleRobotEnv()
    obs, _ = env.reset()
    for _ in range(5):
        a = env.action_space.sample()
        obs, r, t, _, _ = env.step(a)
        print(obs, r)
        if t: break
