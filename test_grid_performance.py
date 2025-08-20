#!/usr/bin/env python3
"""
Test script to compare O(N²) vs O(N) grid-based performance
"""

import time
from run_simple import SimplifiedSimulation

def test_timestep_performance():
    """Compare performance of different timestep methods"""
    
    print("Grid-based Performance Test")
    print("=" * 50)
    
    # Test configurations
    test_configs = [
        (320, "Small scale test"),
        (640, "Medium scale test"), 
        (1280, "Large scale test"),
    ]
    
    for num_agents, description in test_configs:
        print(f"\n{description}: {num_agents} agents")
        print("-" * 40)
        
        # Create simulation
        sim = SimplifiedSimulation(
            world_size=2.0,
            num_agents=num_agents,
            r_a=0.05,
            r_s=0.3
        )
        
        num_timesteps = 50
        
        # Test O(N²) method
        print("Testing O(N²) timestep_gpu_only...")
        start_time = time.time()
        for _ in range(num_timesteps):
            sim.timestep_gpu_only(sync_gpu=False)
        sim.ctx.finish()  # Final sync
        on2_time = time.time() - start_time
        on2_rate = num_timesteps / on2_time
        
        print(f"  O(N²): {on2_time:.3f}s, {on2_rate:.1f} timesteps/sec")
        
        # Reset simulation state
        sim.current_time = 0.0
        
        # Test O(N) grid method
        print("Testing O(N) timestep_gpu_grid...")
        start_time = time.time()
        for _ in range(num_timesteps):
            sim.timestep_gpu_grid(sync_gpu=False)
        sim.ctx.finish()  # Final sync
        grid_time = time.time() - start_time
        grid_rate = num_timesteps / grid_time
        
        print(f"  O(N):  {grid_time:.3f}s, {grid_rate:.1f} timesteps/sec")
        
        # Calculate speedup
        speedup = on2_time / grid_time
        print(f"  Speedup: {speedup:.2f}x {'(Grid faster)' if speedup > 1 else '(O(N²) faster)'}")
        
        # Memory overhead analysis
        print(f"  Grid info: {sim.grid_size}x{sim.grid_size} cells, cell_size={sim.grid_cell_size:.4f}")


if __name__ == "__main__":
    try:
        test_timestep_performance()
    except Exception as e:
        print(f"Error during testing: {e}")
        print("This might be due to missing grid shader files or OpenGL context issues")
        import traceback
        traceback.print_exc()