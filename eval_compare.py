# eval_compare.py
"""
Évaluation comparative : modèle SB3 (float) vs modèle quantifié (.coe -> int8 -> reconstitué)
Usage:
  python eval_compare.py --episodes 50 --render False
"""
"""
Le script eval_compare.py  :

  charge le modèle DQN original (float32) ;

  charge les poids quantifiés int8 depuis tes fichiers .coe ;

  exécute les deux modèles sur le même environnement de test pendant 30 épisodes ;

  compare :

     les récompenses totales (reward) de chaque modèle,

     et le pourcentage d’accord des actions (combien de fois les deux modèles ont choisi la même action).
"""

import argparse
import numpy as np
import os
from stable_baselines3 import DQN
import gymnasium as gym
from env_simple import SimpleRobotEnv

# ---------- Helpers pour charger poids quantifiés ----------
def load_scales_from_coe_dir(coe_dir="coe"):
    scales = {}
    for fname in os.listdir(coe_dir):
        if fname.endswith("_scale.txt"):
            key = fname.replace("_scale.txt","")
            with open(os.path.join(coe_dir,fname), "r") as f:
                scales[key] = float(f.read().strip())
    return scales

def load_int8_weights_from_coe_npy(export_dir="coe"):
    """
    Préfère charger les fichiers numpy int8 si présents (coe/*_int8.npy),
    sinon cherche dans exported_weights/*.npy et quantifie.
    Retour : dict name -> ndarray (int8)
    """
    weights = {}
    for fname in os.listdir(export_dir):
        if fname.endswith("_int8.npy"):
            name = fname.replace("_int8.npy","")
            arr = np.load(os.path.join(export_dir,fname))
            weights[name] = arr.astype(np.int8)
    # fallback: no int8 npy found -> try exported_weights float and quantize using corresponding scale file
    if not weights:
        ew_dir = "exported_weights"
        if os.path.isdir(ew_dir):
            scales = load_scales_from_coe_dir(export_dir)
            for fname in os.listdir(ew_dir):
                if fname.endswith(".npy"):
                    name = fname.replace(".npy","")
                    arr = np.load(os.path.join(ew_dir, fname))
                    if name in scales:
                        scale = scales[name]
                        q = np.clip(np.round(arr / scale), -128, 127).astype(np.int8)
                        weights[name] = q
    return weights

# ---------- Forward pass reconstitué depuis poids quantifiés ----------
def forward_from_quant(weights_int8, scales, obs):
    """
    Reconstruit les poids en float via W = q * scale,
    puis effectue forward simple : relu(W1*x + b1) -> W2*h + b2
    Hypothèse noms : q_net.0.weight, q_net.0.bias, q_net.2.weight, q_net.2.bias
    obs : 1D numpy array shape (4,)
    Retour : logits numpy (shape N_out,), action(int)
    """
    # fetch arrays
    W1_q = weights_int8.get("q_net.0.weight")
    b1_q = weights_int8.get("q_net.0.bias")
    W2_q = weights_int8.get("q_net.2.weight")
    b2_q = weights_int8.get("q_net.2.bias")
    if any(x is None for x in [W1_q,b1_q,W2_q,b2_q]):
        raise RuntimeError("Manque des poids quantifiés (q_net.*) dans coe/ ou exported_weights/")
    # scales
    sW1 = scales.get("q_net.0.weight", 1.0)
    sb1 = scales.get("q_net.0.bias", 1.0)
    sW2 = scales.get("q_net.2.weight", 1.0)
    sb2 = scales.get("q_net.2.bias", 1.0)
    # dequantize to float
    W1 = W1_q.astype(np.float32) * sW1
    b1 = b1_q.astype(np.float32) * sb1
    W2 = W2_q.astype(np.float32) * sW2
    b2 = b2_q.astype(np.float32) * sb2
    # forward
    x = obs.astype(np.float32)
    h = np.maximum(0.0, W1.dot(x) + b1)   # ReLU
    logits = W2.dot(h) + b2
    action = int(np.argmax(logits))
    return logits, action

# ---------- Main evaluation loop ----------
def evaluate(modelsb3_path=r"F:\Users\msi\Desktop\Robot_essaims\Python_Sim\rl_pipeline\models\dqn_robot.zip", episodes=50, render=False):
    # load SB3 model (float)
    model = DQN.load(modelsb3_path)
    env = SimpleRobotEnv()
    scales = load_scales_from_coe_dir("coe")
    weights_int8 = load_int8_weights_from_coe_npy("coe")
    if not weights_int8:
        print("Avertissement : aucun q_*_int8.npy trouvé. Tentative de quantification à partir de exported_weights...")
        weights_int8 = load_int8_weights_from_coe_npy("exported_weights")
    # metrics
    total_rewards_float = []
    total_rewards_quant = []
    action_agreements = []  # per step 1/0
    step_counts = []
    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        ep_reward_f = 0.0
        ep_reward_q = 0.0
        steps = 0
        while not done:
            if render:
                env.render()
            # action from SB3 model (float)
            action_f, _ = model.predict(obs, deterministic=True)
            # action from quantized (reconstructed) forward
            try:
                _, action_q = forward_from_quant(weights_int8, scales, obs)
            except Exception as e:
                print("Erreur forward quantifié:", e)
                raise
            # apply action_f in environment to get next state and reward_f
            obs_next, reward, done, _, _ = env.step(int(action_f))
            ep_reward_f += reward
            # For fair comparison, step environment again from same initial obs for quantized? 
            # Simpler: we apply the same action_f to env; to compare rewards of policies we need separate env copies.
            # So create a separate env for quantized policy evaluation:
            break  # break to switch to separate evals below
        # --- Evaluate float policy on a fresh run ---
        envf = SimpleRobotEnv()
        obs_f, _ = envf.reset()
        done_f = False
        rsum_f = 0.0
        while not done_f:
            a_f, _ = model.predict(obs_f, deterministic=True)
            obs_f, r, done_f, _, _ = envf.step(int(a_f))
            rsum_f += r
        # --- Evaluate quantized policy on a fresh run ---
        envq = SimpleRobotEnv()
        obs_q, _ = envq.reset()
        done_q = False
        rsum_q = 0.0
        agree_steps = 0
        tot_steps = 0
        while not done_q:
            # float action for comparison
            a_f, _ = model.predict(obs_q, deterministic=True)
            _, a_q = forward_from_quant(weights_int8, scales, obs_q)
            # step env with quantized action
            obs_q, r_q, done_q, _, _ = envq.step(int(a_q))
            rsum_q += r_q
            tot_steps += 1
            if int(a_f) == int(a_q):
                agree_steps += 1
        total_rewards_float.append(rsum_f)
        total_rewards_quant.append(rsum_q)
        action_agreements.append(agree_steps / max(1, tot_steps))
        step_counts.append(tot_steps)
        print(f"Ep {ep+1}/{episodes}: R_float={rsum_f:.3f}, R_quant={rsum_q:.3f}, agree={action_agreements[-1]*100:.1f}%")
    # summary
    import statistics
    print("\n=== Résumé ===")
    print(f"Episodes: {episodes}")
    print(f"Reward float  avg={statistics.mean(total_rewards_float):.3f} std={statistics.pstdev(total_rewards_float):.3f}")
    print(f"Reward quant  avg={statistics.mean(total_rewards_quant):.3f} std={statistics.pstdev(total_rewards_quant):.3f}")
    print(f"Action agreement avg={statistics.mean(action_agreements)*100:.2f}% (par-episode)")
    print(f"Mean steps per episode: {statistics.mean(step_counts):.1f}")
    return {
        "r_float": total_rewards_float,
        "r_quant": total_rewards_quant,
        "agree": action_agreements
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    evaluate(episodes=args.episodes, render=args.render)
