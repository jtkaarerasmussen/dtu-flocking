#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import time
import copy
from run_simple import SimplifiedSimulation, Agent

class FastEvolutionarySimulation:
    """
    Fast evolutionary simulation optimized for Figure S2 recreation
    Uses simplified parameters and fewer runs for speed
    """
    
    def __init__(self, world_size: float = 2.0, num_agents: int = 640, density: float = 2.77e-2,
                 pg: float = 1.0, ps: float = 0.0, omega_gc: float = 4.0, omega_sc: float = 4.0,
                 sigma_mu: float = 0.01, tau_tr: int = 50, tau_fit: int = 25, nr: int = 2):
        """
        Fast evolutionary simulation with reduced parameters for Figure S2
        
        Args:
            num_agents: Reduced population size (640 instead of 16384)
            tau_tr: Reduced transient time (50 instead of 2000)
            tau_fit: Reduced fitness evaluation time (25 instead of 500) 
            nr: Reduced fitness runs (2 instead of 30)
        """
        self.world_size = world_size
        self.num_agents = num_agents
        self.density = density
        
        # Calculate r_a from density
        self.r_a = world_size * np.sqrt(density / num_agents)
        self.r_s = 6.0 * self.r_a
        
        # Evolutionary parameters
        self.pg = pg
        self.ps = ps
        self.omega_gc = omega_gc
        self.omega_sc = omega_sc
        self.sigma_mu = sigma_mu
        self.tau_tr = tau_tr
        self.tau_fit = tau_fit
        self.nr = nr
        
        # Initialize population
        self.population = self._initialize_population()
        
        # Track evolution history
        self.generation = 0
        self.history = {
            'generation': [],
            'mean_w_g': [],
            'std_w_g': [],
            'mean_w_s': [],
            'std_w_s': [],
            'mean_fitness': [],
            'w_g_distribution': [],
            'w_s_distribution': []
        }
        
        print(f"FastEvolutionarySimulation: {num_agents} agents, {tau_tr}+{tau_fit} timesteps, {nr} runs")
        print(f"Parameters: r_a={self.r_a:.4f}, r_s={self.r_s:.4f}, pg={pg}, ps={ps}")
    
    def _initialize_population(self) -> list:
        """Initialize population with zero gradient detection and zero sociality"""
        population = []
        for i in range(self.num_agents):
            individual = {
                'w_g': np.random.rand()*2,  
                'w_s': np.random.rand()*2, 
                # 'w_g': 0.0,  
                # 'w_s': 0.0, 
                'fitness': 0.0,
                'id': i
            }
            population.append(individual)
        return population
    
    def evaluate_fitness_fast(self, population: list) -> np.ndarray:
        """
        Fast fitness evaluation with GPU memory pool
        """
        # Average fitness over nr runs
        fitness_sum = np.zeros(self.num_agents, dtype=np.float32)
        
        if not hasattr(self, '_sim'):
            self._sim = SimplifiedSimulation(
                world_size=self.world_size,
                num_agents=self.num_agents,
                r_a=self.r_a,
                r_s=self.r_s,
                s=self.r_a,
                sigma_g=np.sqrt(0.2),
                sigma_r=1.0,
                theta_max=2.0,
                dt=0.2,
                tp=100.0,
                compute_size_x=32
            )
        
        sim = self._sim
        
        for run in range(self.nr):
            self._reset_simulation_state(sim, population)
            
            # Run transient period using batched timesteps
            batch_size = 10
            full_batches = self.tau_tr // batch_size
            for _ in range(full_batches):
                sim.timestep_gpu_batched(batch_size=batch_size)
            remaining = self.tau_tr % batch_size
            if remaining > 0:
                sim.timestep_gpu_batched(batch_size=remaining)
            
            # Reset gradient travel for fitness evaluation
            self._reset_gradient_travel(sim)
            
            # Run fitness evaluation period using batched timesteps  
            full_batches = self.tau_fit // batch_size
            for _ in range(full_batches):
                sim.timestep_gpu_batched(batch_size=batch_size)
            remaining = self.tau_fit % batch_size
            if remaining > 0:
                sim.timestep_gpu_batched(batch_size=remaining)
            
            # Calculate fitness on GPU
            run_fitness = sim.calculate_fitness_gpu(
                pg=self.pg, ps=self.ps,
                omega_gc=self.omega_gc, omega_sc=self.omega_sc,
                fitness_eval_time=self.tau_fit
            )
            
            fitness_sum += run_fitness
        
        return fitness_sum / self.nr
    
    def _reset_simulation_state(self, sim, population):
        """Fast reset without GPU context recreation"""
        # Reset agent positions and phenotypes
        for i, individual in enumerate(population):
            # Random position
            sim.agents[i].c[0] = np.random.uniform(0, self.world_size)
            sim.agents[i].c[1] = np.random.uniform(0, self.world_size)
            
            # Random velocity
            theta = np.random.uniform(0, 2 * np.pi)
            speed = self.r_a
            sim.agents[i].v[0] = speed * np.cos(theta)
            sim.agents[i].v[1] = speed * np.sin(theta)
            
            # Set phenotypes
            sim.agents[i].w_g = individual['w_g']
            sim.agents[i].w_s = individual['w_s']
            sim.agents[i].grad_travel = 0.0
        
        # Update GPU buffers once
        sim._update_gpu_buffers_from_agents()
        sim.current_time = 0.0
    
    def _reset_gradient_travel(self, sim):
        """Fast gradient travel reset"""
        for agent in sim.agents:
            agent.grad_travel = 0.0
        sim._update_gpu_buffers_from_agents()
    
    def roulette_wheel_selection(self, population: list, fitness_values: np.ndarray) -> list:
        """Fitness-proportional selection using softmax"""
        min_fitness = np.min(fitness_values)
        if min_fitness < 0:
            adjusted_fitness = fitness_values - min_fitness
        else:
            adjusted_fitness = fitness_values

        total_fitness = np.sum(adjusted_fitness) 

        probabilities = adjusted_fitness / total_fitness
        
        # Create new population
        new_population = []
        for i in range(self.num_agents):
            parent_idx = np.random.choice(len(population), p=probabilities)
            parent = population[parent_idx]
            
            offspring = {
                'w_g': parent['w_g'],
                'w_s': parent['w_s'],
                'fitness': 0.0,
                'id': i,
                'parent_id': parent['id']
            }
            new_population.append(offspring)
        
        return new_population
    
    def mutate_population(self, population: list) -> list:
        """Apply Gaussian mutations"""
        for individual in population:
            # Gaussian mutation on gradient detection
            w_g_mutation = np.random.normal(0, self.sigma_mu)
            individual['w_g'] = max(0.0, individual['w_g'] + w_g_mutation)
            
            # Gaussian mutation on sociality
            w_s_mutation = np.random.normal(0, self.sigma_mu)
            individual['w_s'] = max(0.0, individual['w_s'] + w_s_mutation)
        
        return population
    
    def record_generation_stats(self, population: list, fitness_values: np.ndarray):
        """Record statistics for current generation"""
        w_g_values = np.array([ind['w_g'] for ind in population])
        w_s_values = np.array([ind['w_s'] for ind in population])
        
        self.history['generation'].append(self.generation)
        self.history['mean_w_g'].append(np.mean(w_g_values))
        self.history['std_w_g'].append(np.std(w_g_values))
        self.history['mean_w_s'].append(np.mean(w_s_values))
        self.history['std_w_s'].append(np.std(w_s_values))
        self.history['mean_fitness'].append(np.mean(fitness_values))
        
        self.history['w_g_distribution'].append(w_g_values.copy())
        self.history['w_s_distribution'].append(w_s_values.copy())
    
    def evolve_generations_fast(self, num_generations: int, verbose: bool = True) -> dict:
        """
        Fast evolution with progress reporting
        """
        if verbose:
            print(f"\n=== FAST EVOLUTIONARY SIMULATION ===")
            print(f"Running {num_generations} generations...")
        
        start_time = time.time()
        
        for gen in range(num_generations):
            self.generation = gen
            
            if verbose and gen % 5 == 0:
                elapsed = time.time() - start_time
                if gen > 0:
                    est_total = elapsed * num_generations / gen
                    remaining = est_total - elapsed
                    print(f"Generation {gen:3d}/{num_generations} - {elapsed:.1f}s elapsed, ~{remaining:.1f}s remaining")
                else:
                    print(f"Generation {gen:3d}/{num_generations} - starting...")
            
            # Evaluate fitness
            fitness_values = self.evaluate_fitness_fast(self.population)
            
            # Record statistics
            self.record_generation_stats(self.population, fitness_values)
            
            # Evolution 
            if gen < num_generations - 1:
                # Selection
                new_population = self.roulette_wheel_selection(self.population, fitness_values)
                
                # Mutation
                new_population = self.mutate_population(new_population)
                
                self.population = new_population
        
        total_time = time.time() - start_time
        
        if verbose:
            print(f"\n=== EVOLUTION COMPLETE ===")
            print(f"Total time: {total_time:.1f} seconds")
            print(f"Average time per generation: {total_time/num_generations:.2f} seconds")
            
            final_w_g = self.history['w_g_distribution'][-1]
            final_w_s = self.history['w_s_distribution'][-1]
            print(f"\nFinal population:")
            print(f"w_g: mean={np.mean(final_w_g):.4f}, std={np.std(final_w_g):.4f}")
            print(f"w_s: mean={np.mean(final_w_s):.4f}, std={np.std(final_w_s):.4f}")
            
        
        return self.history

def recreate_figure_s2_fast():
    """
    Fast recreation of Figure S2 top row with optimized parameters
    """
    print("=== FAST FIGURE S2 RECREATION ===")
    
    # Simulation parameters
    num_agents = 16384
    num_generations = 2000*30*5
    
    # Create evolutionary simulation  
    evolution_sim = FastEvolutionarySimulation(
        world_size=2.0,
        num_agents=num_agents,
        density=2.77e-2, 
        pg=1.0,
        ps=0.0,           
        omega_gc=4.0,
        omega_sc=4.0,
        sigma_mu=0.01,
        tau_tr=250,        
        tau_fit=250,       
        nr=1
    )
    
    # Run evolution
    start_time = time.time()
    history = evolution_sim.evolve_generations_fast(num_generations, verbose=True)
    total_time = time.time() - start_time
    
    # Save simulation data
    import pickle
    from datetime import datetime
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save complete history as pickle
    pickle_filename = f"evolution_history_{timestamp}.pkl"
    with open(pickle_filename, 'wb') as f:
        pickle.dump({
            'history': history,
            'parameters': {
                'num_agents': num_agents,
                'num_generations': num_generations,
                'density': 2.77e-2,
                'pg': evolution_sim.pg,
                'ps': evolution_sim.ps,
                'omega_gc': 4.0,
                'omega_sc': 4.0,
                'sigma_mu': evolution_sim.sigma_mu,
                'tau_tr': evolution_sim.tau_tr,
                'tau_fit': evolution_sim.tau_fit,
                'nr': evolution_sim.nr
            },
            'total_time': total_time
        }, f)
    
    # json_filename = f"evolution_summary_{timestamp}.json"
    
    # def convert_numpy(obj):
    #     if isinstance(obj, np.ndarray):
    #         return obj.tolist()
    #     elif isinstance(obj, (np.float32, np.float64)):
    #         return float(obj)
    #     elif isinstance(obj, (np.int32, np.int64)):
    #         return int(obj)
    #     elif isinstance(obj, list):
    #         return [convert_numpy(item) for item in obj]
    #     else:
    #         return obj
    
    # summary_data = {
    #     'parameters': {
    #         'num_agents': num_agents,
    #         'num_generations': num_generations,
    #         'density': 2.77e-2,
    #         'pg': 1.0,
    #         'ps': 0.0,
    #         'sigma_mu': 0.01,
    #         'tau_tr': 1000,
    #         'tau_fit': 500,
    #         'nr': 6
    #     },
    #     'results': {
    #         'total_time': float(total_time),
    #         'generations': convert_numpy(history['generation']),
    #         'mean_w_g': convert_numpy(history['mean_w_g']),
    #         'std_w_g': convert_numpy(history['std_w_g']),
    #         'mean_w_s': convert_numpy(history['mean_w_s']),
    #         'std_w_s': convert_numpy(history['std_w_s']),
    #         'mean_fitness': convert_numpy(history['mean_fitness'])
    #     }
    # }
    
    # with open(json_filename, 'w') as f:
    #     json.dump(summary_data, f, indent=2)
    
    print(f"\n=== DATA SAVED ===")
    print(f"Complete history: {pickle_filename}")
    
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    generations = history['generation']
    w_g_distributions = history['w_g_distribution']
    w_s_distributions = history['w_s_distribution']
    
    # Plot 1: Gradient detection evolution (2D histogram)
    max_w_g = max(np.max(dist) for dist in w_g_distributions if len(dist) > 0)
    max_w_g = max(max_w_g, 2.0)
    
    gen_edges = np.linspace(0, num_generations, num_generations + 1)
    w_g_edges = np.linspace(0, max_w_g, 50)
    
    hist_data_g = np.zeros((len(w_g_edges) - 1, len(gen_edges) - 1))
    
    for i, w_g_dist in enumerate(w_g_distributions):
        if len(w_g_dist) > 0:
            hist, _ = np.histogram(w_g_dist, bins=w_g_edges)
            hist_data_g[:, i] = hist
    
    im_g = axes[0].imshow(hist_data_g, aspect='auto', origin='lower',
                         extent=[0, num_generations, 0, max_w_g], 
                         cmap='hot', interpolation='nearest')
    axes[0].set_xlabel('Number of generations')
    axes[0].set_ylabel('Gradient detection ability: ωg')
    axes[0].set_title('Fast S2 Recreation\nGradient Detection')
    
    # Plot 2: Sociality evolution (2D histogram)
    max_w_s = max(np.max(dist) for dist in w_s_distributions if len(dist) > 0)
    max_w_s = max(max_w_s, 2.0)
    
    w_s_edges = np.linspace(0, max_w_s, 50)
    hist_data_s = np.zeros((len(w_s_edges) - 1, len(gen_edges) - 1))
    
    for i, w_s_dist in enumerate(w_s_distributions):
        if len(w_s_dist) > 0:
            hist, _ = np.histogram(w_s_dist, bins=w_s_edges)
            hist_data_s[:, i] = hist
    
    im_s = axes[1].imshow(hist_data_s, aspect='auto', origin='lower',
                         extent=[0, num_generations, 0, max_w_s],
                         cmap='hot', interpolation='nearest')
    axes[1].set_xlabel('Number of generations')
    axes[1].set_ylabel('Sociality trait: ωs')
    axes[1].set_title('Fast S2 Recreation\nSociality Trait')
    
    # Plot 3: Mean w_g over time
    mean_w_g = history['mean_w_g']
    std_w_g = history['std_w_g']
    axes[2].plot(generations, mean_w_g, 'r-', linewidth=2, label='Mean ωg')
    axes[2].fill_between(generations,
                        np.array(mean_w_g) - np.array(std_w_g),
                        np.array(mean_w_g) + np.array(std_w_g),
                        alpha=0.3, color='red', label='±1 std')
    axes[2].set_xlabel('Generation')
    axes[2].set_ylabel('Mean ωg')
    axes[2].set_title('Mean Gradient Detection')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    # Plot 4: Mean w_s over time
    mean_w_s = history['mean_w_s']
    std_w_s = history['std_w_s']
    axes[3].plot(generations, mean_w_s, 'b-', linewidth=2, label='Mean ωs')
    axes[3].fill_between(generations,
                        np.array(mean_w_s) - np.array(std_w_s),
                        np.array(mean_w_s) + np.array(std_w_s),
                        alpha=0.3, color='blue', label='±1 std')
    axes[3].set_xlabel('Generation')
    axes[3].set_ylabel('Mean ωs')
    axes[3].set_title('Mean Sociality')
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)
    
    plt.colorbar(im_g, ax=axes[0], label='Individuals')
    plt.colorbar(im_s, ax=axes[1], label='Individuals')
    
    plt.suptitle(f'Fast Figure S2 Recreation: Initial Condition Dependence\n'
                f'pg=1.0, ps=0.0, N={num_agents} agents, {num_generations} generations, '
                f'{total_time:.1f}s total time', fontsize=12)
    
    plt.tight_layout()
    
    # Save figure
    save_path = "figure_s2_fast_recreation.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nFigure saved to {save_path}")
    
    plt.show()
    
    return history

if __name__ == "__main__":
    history = recreate_figure_s2_fast()
    