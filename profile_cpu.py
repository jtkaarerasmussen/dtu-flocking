#!/usr/bin/env python3

import cProfile
import pstats
from evolution_checkpoint import CheckpointEvolutionSimulation

def run_profiled_evolution():
    """Run a short evolution with CPU profiling"""
    
    sim = CheckpointEvolutionSimulation(
        world_size=2.0,
        num_agents=320,  # Smaller for faster profiling
        density=0.0277,
        pg=1.0, ps=0.0,
        omega_gc=4.0, omega_sc=4.0,
        sigma_mu=0.01,
        tau_tr=50, tau_fit=25, nr=5,  # Smaller nr for faster profiling
        checkpoint_interval=1
    )
    
    # Run just 2 generations for profiling
    history = sim.run_evolution(2, start_fresh=True)
    return history

if __name__ == "__main__":
    print("Starting CPU profiling...")
    
    # Run with profiling
    profiler = cProfile.Profile()
    profiler.enable()
    
    history = run_profiled_evolution()
    
    profiler.disable()
    
    # Save profile results
    profiler.dump_stats('cpu_profile.prof')
    
    # Print top CPU consumers
    stats = pstats.Stats('cpu_profile.prof')
    stats.sort_stats('cumulative')
    print("=== TOP 20 CPU CONSUMERS (by cumulative time) ===")
    stats.print_stats(20)
    
    print("\n=== GPU/BUFFER/MODERNGL RELATED FUNCTIONS ===")
    stats.print_stats('.*gpu.*|.*buffer.*|.*moderngl.*|.*gl.*')
    
    print("\n=== TIMESTEP/SIMULATION RELATED FUNCTIONS ===")
    stats.print_stats('.*timestep.*|.*simulation.*|.*fitness.*')
    
    print("\nProfile saved to cpu_profile.prof")
    print("Run this script on the A100 system to see the actual bottleneck")