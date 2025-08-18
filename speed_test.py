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
    
    # Test configurations: (num_agents, timesteps, description)
    test_configs = [
        (1024, 2000, "Small scale"),
        (4096, 1500, "Medium scale"), 
        (8192, 1000, "Large scale"),
        (16384, 800, "Extra large scale"),
    ]
    
    total_start = time.time()
    
    for num_agents, timesteps, description in test_configs:
        print(f"\nTesting {description}: {num_agents} agents, {timesteps} timesteps")
        
        # Create simulation with world size 100
        sim = SimplifiedSimulation(world_size=100.0, num_agents=num_agents)
        
        start_time = time.time()
        
        for step in range(timesteps):
            sim.timestep()
            
            # Progress indicator every 25 steps
            if (step + 1) % 200 == 0:
                elapsed = time.time() - start_time
                rate = (step + 1) / elapsed
                print(f"  Step {step + 1}/{timesteps} - {rate:.1f} timesteps/sec")
        
        elapsed = time.time() - start_time
        rate = timesteps / elapsed
        total_ops = num_agents * timesteps
        ops_per_sec = total_ops / elapsed
        
        print(f"  Completed: {rate:.1f} timesteps/sec, {ops_per_sec:.0f} agent-steps/sec")
        
        # Check if we've been running for close to a minute
        total_elapsed = time.time() - total_start
    
    total_elapsed = time.time() - total_start
    print("\n" + "=" * 50)
    print(f"Speed test completed in {total_elapsed:.1f} seconds")

if __name__ == "__main__":
    run_speed_test()