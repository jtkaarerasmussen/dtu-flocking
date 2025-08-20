#!/usr/bin/env python3

import math
from cupy_simulation import CuPySimulation

def benchmark_cupy():
    """Benchmark CuPy simulation vs OpenGL version"""
    
    WORLD_SIZE = 2.0
    DENSITY = 0.0277
    NUM_TIMESTEPS = 1000
    
    # Test different agent counts and block sizes
    agent_counts = [64, 256, 640, 1024, 2560, 6400]
    block_sizes = [256, 512, 1024]
    
    print("=== CUPY SIMULATION BENCHMARK ===")
    print(f"Testing {NUM_TIMESTEPS} timesteps per configuration")
    print(f"World size: {WORLD_SIZE}, Density: {DENSITY}")
    print()
    
    results = []
    
    for num_agents in agent_counts:
        # Calculate parameters
        r_a = WORLD_SIZE * math.sqrt(DENSITY / num_agents)
        r_s = 6.0 * r_a
        
        print(f"=== {num_agents} agents (r_a={r_a:.4f}, r_s={r_s:.4f}) ===")
        
        best_perf = 0
        best_block_size = 0
        
        for block_size in block_sizes:
            try:
                sim = CuPySimulation(
                    world_size=WORLD_SIZE,
                    num_agents=num_agents,
                    r_a=r_a, r_s=r_s, s=r_a,
                    sigma_r=1.0, theta_max=2.0, dt=0.2,
                    block_size=block_size
                )
                
                timesteps_per_sec = sim.benchmark(NUM_TIMESTEPS)
                
                if timesteps_per_sec > best_perf:
                    best_perf = timesteps_per_sec
                    best_block_size = block_size
                
                print(f"  Block size {block_size}: {timesteps_per_sec:.1f} timesteps/sec")
                
            except Exception as e:
                print(f"  Block size {block_size}: ERROR - {e}")
        
        agent_timesteps_per_sec = num_agents * best_perf
        
        results.append({
            'agents': num_agents,
            'best_timesteps_per_sec': best_perf,
            'best_block_size': best_block_size,
            'agent_timesteps_per_sec': agent_timesteps_per_sec
        })
        
        print(f"  BEST: {best_perf:.1f} timesteps/sec (block_size={best_block_size})")
        print(f"        {agent_timesteps_per_sec:.0f} agent-timesteps/sec")
        print()
    
    # Summary
    print("=== SUMMARY ===")
    print(f"{'Agents':<8} {'Best t/s':<12} {'Block Size':<12} {'Agent-t/s':<15}")
    print("-" * 55)
    
    for r in results:
        print(f"{r['agents']:<8} {r['best_timesteps_per_sec']:<12.1f} "
              f"{r['best_block_size']:<12} {r['agent_timesteps_per_sec']:<15.0f}")
    
    return results

if __name__ == "__main__":
    benchmark_cupy()