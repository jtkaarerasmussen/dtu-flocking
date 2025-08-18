#!/usr/bin/env python3

import os
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
import time
from datetime import datetime
from pathlib import Path
from evolution_static import run_evolution_from_params

def create_bootstrap_directory():
    """Create directory structure for bootstrap runs"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = f"bootstrap_static_evolution_{timestamp}"
    Path(base_dir).mkdir(exist_ok=True)
    
    # Create subdirectories
    Path(f"{base_dir}/fixed_g_runs").mkdir(exist_ok=True)
    Path(f"{base_dir}/fixed_s_runs").mkdir(exist_ok=True)
    Path(f"{base_dir}/plots").mkdir(exist_ok=True)
    
    return base_dir

def run_fixed_parameter_sweep(base_dir, parameter_name, fixed_values, num_generations=100, num_runs=5):
    """
    Run evolutionary simulations with one parameter fixed at different values
    
    Args:
        base_dir: Base directory for saving results
        parameter_name: 'g' or 's' - which parameter to fix
        fixed_values: Array of values to fix the parameter at
        num_generations: Number of generations per run
        num_runs: Number of independent runs per fixed value
    """
    results = []
    
    # Base parameters
    base_params = {
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
        "num_generations": num_generations
    }
    
    total_runs = len(fixed_values) * num_runs
    run_count = 0
    
    for fixed_value in fixed_values:
        print(f"\n=== Running with fixed {parameter_name} = {fixed_value:.3f} ===")
        
        # Set up parameters for this fixed value
        params = base_params.copy()
        
        if parameter_name == 'g':
            # Fix g parameter, allow s to evolve
            params['g_evol'] = False
            params['s_evol'] = True  
            params['g_mu'] = fixed_value
            params['g_sigma'] = 0.0  # No variation
            params['s_mu'] = 0.1     # Initial value for evolving parameter
            params['s_sigma'] = 0.1
        else:  # parameter_name == 's'
            # Fix s parameter, allow g to evolve
            params['g_evol'] = True
            params['s_evol'] = False
            params['g_mu'] = 0.1     # Initial value for evolving parameter 
            params['g_sigma'] = 0.1
            params['s_mu'] = fixed_value
            params['s_sigma'] = 0.0  # No variation
        
        fixed_value_results = []
        
        for run_idx in range(num_runs):
            run_count += 1
            print(f"  Run {run_idx + 1}/{num_runs} (overall {run_count}/{total_runs})")
            
            # Create unique checkpoint directory for this run
            checkpoint_dir = f"{base_dir}/fixed_{parameter_name}_runs/{parameter_name}_{fixed_value:.3f}_run_{run_idx}"
            params['checkpoint_dir'] = checkpoint_dir
            
            start_time = time.time()
            
            try:
                # Run evolution
                history = run_evolution_from_params(params, num_generations, start_fresh=True)
                
                # Extract final values
                if parameter_name == 'g':
                    # g is fixed, s evolved
                    final_fixed = fixed_value
                    final_evolved = np.mean(history['w_s_distribution'][-1])
                    final_evolved_std = np.std(history['w_s_distribution'][-1])
                else:
                    # s is fixed, g evolved  
                    final_fixed = fixed_value
                    final_evolved = np.mean(history['w_g_distribution'][-1])
                    final_evolved_std = np.std(history['w_g_distribution'][-1])
                
                run_result = {
                    'fixed_value': final_fixed,
                    'final_evolved_mean': final_evolved,
                    'final_evolved_std': final_evolved_std,
                    'final_fitness': np.mean(history['mean_fitness'][-10:]),  # Average of last 10 generations
                    'run_time': time.time() - start_time,
                    'checkpoint_dir': checkpoint_dir,
                    'converged': final_evolved_std < 0.01  # Simple convergence check
                }
                
                fixed_value_results.append(run_result)
                
                print(f"    Final evolved {('s' if parameter_name == 'g' else 'g')}: {final_evolved:.4f} ± {final_evolved_std:.4f}")
                
            except Exception as e:
                print(f"    ERROR in run {run_idx}: {e}")
                continue
        
        results.append({
            'fixed_parameter': parameter_name,
            'fixed_value': fixed_value,
            'runs': fixed_value_results
        })
    
    return results

def plot_bootstrap_results(results_g, results_s, base_dir):
    """Plot the relationship between fixed and evolved parameters"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Fixed g, evolved s
    if results_g:
        fixed_g_values = []
        evolved_s_means = []
        evolved_s_stds = []
        
        for result_group in results_g:
            fixed_val = result_group['fixed_value']
            runs = result_group['runs']
            
            if runs:  # If we have successful runs
                # Average across runs for this fixed value
                run_means = [run['final_evolved_mean'] for run in runs]
                run_stds = [run['final_evolved_std'] for run in runs]
                
                fixed_g_values.append(fixed_val)
                evolved_s_means.append(np.mean(run_means))
                evolved_s_stds.append(np.mean(run_stds))
        
        ax1.errorbar(fixed_g_values, evolved_s_means, yerr=evolved_s_stds, 
                    marker='o', capsize=5, capthick=2, linewidth=2)
        ax1.set_xlabel('Fixed w_g')
        ax1.set_ylabel('Final evolved w_s')
        ax1.set_title('Fixed Gradient Detection → Evolved Sociality')
        ax1.grid(True, alpha=0.3)
    
    # Plot 2: Fixed s, evolved g  
    if results_s:
        fixed_s_values = []
        evolved_g_means = []
        evolved_g_stds = []
        
        for result_group in results_s:
            fixed_val = result_group['fixed_value']
            runs = result_group['runs']
            
            if runs:  # If we have successful runs
                # Average across runs for this fixed value
                run_means = [run['final_evolved_mean'] for run in runs]
                run_stds = [run['final_evolved_std'] for run in runs]
                
                fixed_s_values.append(fixed_val)
                evolved_g_means.append(np.mean(run_means))
                evolved_g_stds.append(np.mean(run_stds))
        
        ax2.errorbar(fixed_s_values, evolved_g_means, yerr=evolved_g_stds,
                    marker='s', capsize=5, capthick=2, linewidth=2, color='orange')
        ax2.set_xlabel('Fixed w_s')
        ax2.set_ylabel('Final evolved w_g')
        ax2.set_title('Fixed Sociality → Evolved Gradient Detection')
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_file = f"{base_dir}/plots/bootstrap_static_evolution.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {plot_file}")
    
    # Also save as PDF
    plot_pdf = f"{base_dir}/plots/bootstrap_static_evolution.pdf"
    plt.savefig(plot_pdf, bbox_inches='tight')
    
    plt.show()

def main():
    """Main bootstrap analysis"""
    print("=== BOOTSTRAP STATIC EVOLUTION ANALYSIS ===")
    
    # Configuration
    fixed_values = np.linspace(0.0, 0.25, 30)  # 0.0, 0.05, 0.10, 0.15, 0.20, 0.25
    num_generations = 50  # Longer runs for convergence
    num_runs = 1  # Multiple runs per parameter value
    
    print(f"Fixed values: {fixed_values}")
    print(f"Generations per run: {num_generations}")
    print(f"Runs per fixed value: {num_runs}")
    print(f"Total runs: {len(fixed_values) * 2 * num_runs}")
    
    # Create directory structure
    base_dir = create_bootstrap_directory()
    print(f"Results will be saved to: {base_dir}")
    
    # Run parameter sweeps
    print("\n" + "="*60)
    print("PHASE 1: Fixed gradient detection (w_g), evolving sociality (w_s)")
    results_g = run_fixed_parameter_sweep(base_dir, 'g', fixed_values, num_generations, num_runs)
    
    print("\n" + "="*60)
    print("PHASE 2: Fixed sociality (w_s), evolving gradient detection (w_g)")  
    results_s = run_fixed_parameter_sweep(base_dir, 's', fixed_values, num_generations, num_runs)
    
    # Save all results
    results_file = f"{base_dir}/bootstrap_results.pkl"
    with open(results_file, 'wb') as f:
        pickle.dump({
            'results_g': results_g,
            'results_s': results_s,
            'config': {
                'fixed_values': fixed_values,
                'num_generations': num_generations,
                'num_runs': num_runs
            },
            'timestamp': datetime.now().isoformat()
        }, f)
    
    print(f"\nComplete results saved to: {results_file}")
    
    # Create plots
    print("\n" + "="*60)  
    print("CREATING PLOTS")
    plot_bootstrap_results(results_g, results_s, base_dir)
    
    print(f"\n=== BOOTSTRAP ANALYSIS COMPLETE ===")
    print(f"All results saved in: {base_dir}")

if __name__ == "__main__":
    main()