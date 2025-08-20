#!/usr/bin/env python3

import time
import math
from run_simple import SimplifiedSimulation

def benchmark_timesteps():
    """Benchmark timesteps/sec for different agent counts and methods"""
    
    WORLD_SIZE = 2.0
    DENSITY = 0.0277
    NUM_TIMESTEPS = 100  # Test duration
    
    # Different agent counts to test
    agent_counts = [64, 256, 640, 1024, 2560, 6400,16000,30000]
    
    results = []
    
    print("=== GPU TIMESTEP BENCHMARK ===")
    print(f"Testing {NUM_TIMESTEPS} timesteps per configuration")
    print(f"World size: {WORLD_SIZE}, Density: {DENSITY}")
    print()
    
    for num_agents in agent_counts:
        # Calculate parameters
        r_a = WORLD_SIZE * math.sqrt(DENSITY / num_agents)
        r_s = 6.0 * r_a
        
        print(f"Testing {num_agents} agents (r_a={r_a:.4f}, r_s={r_s:.4f})")
        
        # Test O(N²) method
        try:
            sim = SimplifiedSimulation(
                world_size=WORLD_SIZE,
                num_agents=num_agents,
                r_a=r_a, r_s=r_s, s=r_a,
                sigma_g=0.447, sigma_r=1.0,
                theta_max=2.0, dt=0.2, tp=100.0,
                compute_size_x=256
            )
            
            # Warm up
            for _ in range(5):
                sim.timestep_gpu_only()
            
            # Benchmark O(N²) method
            start_time = time.time()
            for _ in range(NUM_TIMESTEPS):
                sim.timestep_gpu_only()
            sim.ctx.finish()  # FORCE GPU to complete all work before measuring time
            end_time = time.time()
            
            total_time = end_time - start_time
            timesteps_per_sec = NUM_TIMESTEPS / total_time
            agent_timesteps_per_sec = num_agents * timesteps_per_sec
            
            print(f"  O(N²) method:  {timesteps_per_sec:.1f} timesteps/sec, {agent_timesteps_per_sec:.0f} agent-timesteps/sec")
            
            sim.cleanup()
            
        except Exception as e:
            print(f"  O(N²) method:  ERROR - {e}")
            timesteps_per_sec = 0
            agent_timesteps_per_sec = 0
        
        # Test branchless method (A100 optimized)
        try:
            sim = SimplifiedSimulation(
                world_size=WORLD_SIZE,
                num_agents=num_agents,
                r_a=r_a, r_s=r_s, s=r_a,
                sigma_g=0.447, sigma_r=1.0,
                theta_max=2.0, dt=0.2, tp=100.0,
                compute_size_x=256
            )
            
            # Warm up
            for _ in range(5):
                sim.timestep_gpu_branchless()
            
            # Benchmark branchless method
            start_time = time.time()
            for _ in range(NUM_TIMESTEPS):
                sim.timestep_gpu_branchless()
            sim.ctx.finish()  # FORCE GPU to complete all work before measuring time
            end_time = time.time()
            
            total_time = end_time - start_time
            branchless_timesteps_per_sec = NUM_TIMESTEPS / total_time
            branchless_agent_timesteps_per_sec = num_agents * branchless_timesteps_per_sec
            
            print(f"  Branchless:    {branchless_timesteps_per_sec:.1f} timesteps/sec, {branchless_agent_timesteps_per_sec:.0f} agent-timesteps/sec")
            
            sim.cleanup()
            
        except Exception as e:
            print(f"  Branchless:    ERROR - {e}")
            branchless_timesteps_per_sec = 0
            branchless_agent_timesteps_per_sec = 0
        
        # Test grid method
        try:
            sim = SimplifiedSimulation(
                world_size=WORLD_SIZE,
                num_agents=num_agents,
                r_a=r_a, r_s=r_s, s=r_a,
                sigma_g=0.447, sigma_r=1.0,
                theta_max=2.0, dt=0.2, tp=100.0,
                compute_size_x=256
            )
            
            # Warm up
            for _ in range(5):
                sim.timestep_gpu_grid()
            
            # Benchmark grid method
            start_time = time.time()
            for _ in range(NUM_TIMESTEPS):
                sim.timestep_gpu_grid()
            sim.ctx.finish()  # FORCE GPU to complete all work before measuring time
            end_time = time.time()
            
            total_time = end_time - start_time
            grid_timesteps_per_sec = NUM_TIMESTEPS / total_time
            grid_agent_timesteps_per_sec = num_agents * grid_timesteps_per_sec
            
            print(f"  Grid method:   {grid_timesteps_per_sec:.1f} timesteps/sec, {grid_agent_timesteps_per_sec:.0f} agent-timesteps/sec")
            
            # Calculate speedup
            if timesteps_per_sec > 0:
                speedup = grid_timesteps_per_sec / timesteps_per_sec
                print(f"  Grid speedup:  {speedup:.2f}x")
            
            sim.cleanup()
            
        except Exception as e:
            print(f"  Grid method:   ERROR - {e}")
            grid_timesteps_per_sec = 0
            grid_agent_timesteps_per_sec = 0
        
        results.append({
            'agents': num_agents,
            'o2_timesteps_per_sec': timesteps_per_sec,
            'o2_agent_timesteps_per_sec': agent_timesteps_per_sec,
            'grid_timesteps_per_sec': grid_timesteps_per_sec,
            'grid_agent_timesteps_per_sec': grid_agent_timesteps_per_sec
        })
        
        print()
    
    # Summary
    print("=== SUMMARY ===")
    print(f"{'Agents':<8} {'O(N²) t/s':<12} {'Grid t/s':<12} {'O(N²) at/s':<15} {'Grid at/s':<15} {'Speedup':<8}")
    print("-" * 80)
    
    for r in results:
        speedup = r['grid_timesteps_per_sec'] / r['o2_timesteps_per_sec'] if r['o2_timesteps_per_sec'] > 0 else 0
        print(f"{r['agents']:<8} {r['o2_timesteps_per_sec']:<12.1f} {r['grid_timesteps_per_sec']:<12.1f} "
              f"{r['o2_agent_timesteps_per_sec']:<15.0f} {r['grid_agent_timesteps_per_sec']:<15.0f} {speedup:<8.2f}x")
    
    return results

if __name__ == "__main__":
    benchmark_timesteps()