"""
Quantifie les poids exportés en int8 et crée des fichiers .coe
(compatible Xilinx BlockRAM).
Chaque fichier .coe contient les valeurs hex signées.
"""

import numpy as np
import glob, os

def quantize_to_int8(array: np.ndarray):
    max_abs = np.max(np.abs(array))
    scale = max_abs / 127 if max_abs != 0 else 1.0
    q = np.clip(np.round(array / scale), -128, 127).astype(np.int8)
    return q, scale

def save_as_coe(qarray: np.ndarray, filename: str):
    # flatten et convertir en hex
    flat = qarray.flatten()
    hexvals = [f"{(int(val) & 0xFF):02X}" for val in flat]
    with open(filename, "w") as f:
        f.write("memory_initialization_radix=16;\n")
        f.write("memory_initialization_vector=\n")
        f.write(",\n".join(hexvals))
        f.write(";\n")

if __name__ == "__main__":
    os.makedirs("coe", exist_ok=True)
    npy_files = glob.glob("exported_weights/*.npy")

    scales = {}
    for f in npy_files:
        name = os.path.splitext(os.path.basename(f))[0]
        arr = np.load(f)
        q, scale = quantize_to_int8(arr)
        scales[name] = scale
        np.save(f"coe/{name}_int8.npy", q)
        save_as_coe(q, f"coe/{name}.coe")
        with open(f"coe/{name}_scale.txt", "w") as sf:
            sf.write(str(scale))
        print(f"Quantized {name}, scale={scale:.6f}")

    # Sauvegarde globale
    np.save("coe/scales_dict.npy", scales)
    print("Tous les fichiers .coe et échelles générés dans ./coe/")
