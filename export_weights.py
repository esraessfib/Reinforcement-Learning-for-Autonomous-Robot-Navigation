"""
Charge le modèle DQN et exporte les matrices de poids
et biais sous forme numpy (.npy, .csv).
"""

import torch
import numpy as np
from stable_baselines3 import DQN
import os

# Charger le modèle entraîné
model = DQN.load("models/dqn_robot.zip")
sdict = model.policy.q_net.state_dict()

os.makedirs("exported_weights", exist_ok=True)

for name, param in sdict.items():
    arr = param.cpu().numpy()
    np.save(f"exported_weights/{name}.npy", arr)
    np.savetxt(f"exported_weights/{name}.csv", arr, delimiter=",")
    print(f"Exported {name}: shape {arr.shape}")

print(" Poids exportés dans exported_weights/")
