#!/usr/bin/env python3

import math
from cupy_simulation import CuPySimulation

def test_simple():
    """Simple test to verify CuPy simulation works"""
    
    print("=== SIMPLE CUPY TEST ===")
    
    # Small test case
    WORLD_SIZE = 2.0
    NUM_AGENTS = 640
    DENSITY = 0.0277
    
    r_a = WORLD_SIZE * math.sqrt(DENSITY / NUM_AGENTS)
    r_s = 6.0 * r_a
    
    print(f"Testing {NUM_AGENTS} agents")
    print(f"Parameters: r_a={r_a:.4f}, r_s={r_s:.4f}")
    
    # Create simulation
    sim = CuPySimulation(
        world_size=WORLD_SIZE,
        num_agents=NUM_AGENTS,
        r_a=r_a, r_s=r_s, s=r_a,
        sigma_r=1.0, theta_max=2.0, dt=0.2,
        block_size=256
    )
    
    # Test a few timesteps
    print("\nRunning 10 timesteps...")
    for i in range(10):
        sim.timestep()
        if i % 5 == 0:
            positions = sim.get_positions()
            print(f"  Step {i}: Agent 0 at ({positions[0,0]:.3f}, {positions[0,1]:.3f})")
    
    # Quick benchmark
    print(f"\nQuick benchmark (100 timesteps)...")
    perf = sim.benchmark(100)
    
    print(f"Success! CuPy simulation working at {perf:.1f} timesteps/sec")
    return True

if __name__ == "__main__":
    test_simple()