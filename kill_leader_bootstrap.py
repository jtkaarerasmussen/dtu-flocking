import numpy as np
import json
import os

def create_default_params_file(filename="evolution_params.json", leader_kill_prob=0.0):
    """Create a default parameters JSON file"""
    default_params = {
        "world_size": 2.0,
        "num_agents": 320,
        "density": 0.0277,
        "pg": 1.0,
        "ps": 0.0,
        "omega_gc": 4.0,
        "omega_sc": 4.0,
        "sigma_mu": 0.01,
        "tau_tr": 2000,
        "tau_fit": 500,
        "nr": 30,
        "checkpoint_interval": 10,
        "num_generations": 100,
        "selection_method": "tournament",
        "leader_kill_prob": leader_kill_prob
    }
    with open(filename, 'w') as f:
        json.dump(default_params, f, indent=2)
    print(f"Default parameters saved to {filename}")
    return filename

prob_points = np.linspace(0,1,50)

for i,p in enumerate(prob_points):
    print(i,p)
    create_default_params_file(leader_kill_prob=p)
    os.system("python3 evolution_checkpoint.py evolution_params.json")
