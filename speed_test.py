#!/usr/bin/env python3
"""
Speed test for flocking simulation - runs for approximately 1 minute
"""

import time
from run_simple import SimplifiedSimulation

def run_speed_test():
    """Run speed tests with different configurations for about 1 minute"""
    
    print("Starting flocking simulation speed test...")
    print("=" * 50)
    
    # Track performance data
    performance_data = {}
    
    # Test configurations: (num_agents, timesteps, description)
    test_configs = [
        (640, 5000, "Fast evolutionary scale"),
        (1024, 4000, "Small scale"),
        (4096, 3000, "Medium scale"), 
        (8192, 2000, "Large scale"),
        (16384, 1000, "Extra large scale"),
    ]
    
    total_start = time.time()
    
    for num_agents, timesteps, description in test_configs:
        print(f"\nTesting {description}: {num_agents} agents, {timesteps} timesteps")
        
        # Create simulation with world size 100 and optimized compute size for A100
        sim = SimplifiedSimulation(world_size=100.0, num_agents=num_agents, compute_size_x=256)
        
        start_time = time.time()
        
        # Test both GPU-only and batched modes with manual synchronization
        print(f"  Testing GPU-only mode (with sync)...")
        start_gpu_only = time.time()
        for step in range(timesteps // 2):
            sim.timestep_gpu_only()
        sim.ctx.finish()  # Force GPU completion for accurate timing
        gpu_only_time = time.time() - start_gpu_only
        gpu_only_rate = (timesteps // 2) / gpu_only_time
        
        print(f"  GPU-only: {gpu_only_rate:.1f} timesteps/sec")
        
        print(f"  Testing batched mode (with sync)...")
        start_batched = time.time()
        batch_size = 10
        remaining_steps = timesteps - (timesteps // 2)
        full_batches = remaining_steps // batch_size
        for _ in range(full_batches):
            sim.timestep_gpu_batched(batch_size=batch_size)
        if remaining_steps % batch_size > 0:
            sim.timestep_gpu_batched(batch_size=remaining_steps % batch_size)
        sim.ctx.finish()  # Force GPU completion for accurate timing
        batched_time = time.time() - start_batched
        batched_rate = remaining_steps / batched_time
        
        print(f"  Batched: {batched_rate:.1f} timesteps/sec")
        
        elapsed = time.time() - start_time
        rate = timesteps / elapsed
        total_ops = num_agents * timesteps
        ops_per_sec = total_ops / elapsed
        
        print(f"  Overall: {rate:.1f} timesteps/sec, {ops_per_sec:.0f} agent-steps/sec")
        print(f"  Best rate: {max(gpu_only_rate, batched_rate):.1f} timesteps/sec")
        
        # Store performance data
        performance_data[num_agents] = {
            'gpu_only_rate': gpu_only_rate,
            'batched_rate': batched_rate,
            'best_rate': max(gpu_only_rate, batched_rate)
        }
        
        # Check if we've been running for close to a minute
        total_elapsed = time.time() - total_start
    
    total_elapsed = time.time() - total_start
    print("\n" + "=" * 50)
    print(f"Speed test completed in {total_elapsed:.1f} seconds")
    
    # Calculate evolutionary simulation estimate using actual data
    print("\nEvolutionary simulation time estimates:")
    print("=" * 50)
    
    # Use actual 16K agent performance if available, otherwise extrapolate
    if 16384 in performance_data:
        rate_16k = performance_data[16384]['best_rate']
        print(f"Using measured 16K agent performance: {rate_16k:.1f} timesteps/sec")
    else:
        # Find the largest tested configuration and extrapolate
        max_agents = max(performance_data.keys())
        max_rate = performance_data[max_agents]['best_rate']
        # Rough O(N²) scaling estimate
        scaling_factor = (16384 / max_agents) ** 2
        rate_16k = max_rate / scaling_factor
        print(f"Extrapolated 16K performance from {max_agents} agents:")
        print(f"  {max_agents} agent rate: {max_rate:.1f} timesteps/sec")
        print(f"  Scaling factor (O(N²)): {scaling_factor:.1f}x slower")
        print(f"  Estimated 16K rate: {rate_16k:.1f} timesteps/sec")
    
    total_timesteps = 2500 * 30 * 2000  # generations * population * timesteps_per_individual
    estimated_seconds = total_timesteps / rate_16k
    estimated_hours = estimated_seconds / 3600
    estimated_days = estimated_hours / 24
    
    print(f"\nTarget simulation: 16,384 agents")
    print(f"Total timesteps: {total_timesteps:,} ({2500} generations × 30 runs × {2000} timesteps)")
    print(f"Estimated time: {estimated_seconds:,.0f} seconds ({estimated_hours:.1f} hours, {estimated_days:.1f} days)")

if __name__ == "__main__":
    run_speed_test()