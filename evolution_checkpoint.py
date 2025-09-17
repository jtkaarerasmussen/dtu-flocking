#!/usr/bin/env python3

import os
import json
import pickle
import numpy as np
import time
from datetime import datetime
from pathlib import Path
from run_simple import SimplifiedSimulation
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans 

class CheckpointEvolutionSimulation:
    """
    Evolutionary simulation with checkpointing support for long-running experiments
    """
    
    def __init__(self, world_size=2.0, num_agents=640, density=2.77e-2,
                 pg=1.0, ps=0.0, omega_gc=4.0, omega_sc=4.0,
                 sigma_mu=0.01, tau_tr=50, tau_fit=25, nr=2,
                 checkpoint_interval=10, checkpoint_dir=None, selection_method='tournament', leader_kill_prob = 0.0):
        """
        Initialize evolutionary simulation with checkpointing
        
        Args:
            checkpoint_interval: Save checkpoint every N generations
            checkpoint_dir: Directory to save checkpoints (auto-generated if None)
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
        self.selection_method = selection_method
        self.leader_kill_prob = leader_kill_prob
        
        # Checkpointing parameters
        self.checkpoint_interval = checkpoint_interval
        if checkpoint_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.checkpoint_dir = f"evolution_checkpoints_{timestamp}"
        else:
            self.checkpoint_dir = checkpoint_dir
        
        # Initialize or load simulation state
        self.generation = 0
        self.population = None
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
        
        # Create checkpoint directory
        Path(self.checkpoint_dir).mkdir(exist_ok=True)
        
        print(f"CheckpointEvolutionSimulation: {num_agents} agents")
        print(f"Checkpoint directory: {self.checkpoint_dir}")
        print(f"Parameters: r_a={self.r_a:.4f}, r_s={self.r_s:.4f}, pg={pg}, ps={ps}")
    
    def get_parameters_dict(self):
        """Return all simulation parameters as a dictionary"""
        return {
            'world_size': self.world_size,
            'num_agents': self.num_agents,
            'density': self.density,
            'r_a': self.r_a,
            'r_s': self.r_s,
            'pg': self.pg,
            'ps': self.ps,
            'omega_gc': self.omega_gc,
            'omega_sc': self.omega_sc,
            'sigma_mu': self.sigma_mu,
            'tau_tr': self.tau_tr,
            'tau_fit': self.tau_fit,
            'nr': self.nr,
            'checkpoint_interval': self.checkpoint_interval,
            'selection_method': self.selection_method,
            'leader_kill_prob': self.leader_kill_prob
        }
    
    def save_parameters(self):
        """Save simulation parameters to JSON file"""
        params_file = os.path.join(self.checkpoint_dir, 'parameters.json')
        with open(params_file, 'w') as f:
            json.dump(self.get_parameters_dict(), f, indent=2)
        print(f"Parameters saved to {params_file}")
    
    def _initialize_population(self):
        """Initialize population with random traits"""
        population = []
        for i in range(self.num_agents):
            # start close to eq
            if i % 4 ==0:
                individual = {
                    'w_g': max(0.0, np.random.normal(0.3,0.5)),
                    'w_s': 0.0,
                    'fitness': 0.0,
                    'id': i
                }
            else:
                individual = {
                    'w_g': 0.0,
                    'w_s': max(0.0, np.random.normal(0.3,0.5)),
                    'fitness': 0.0,
                    'id': i
                }
            # individual = {
            #     'w_g': np.random.rand() * 2,
            #     'w_s': np.random.rand() * 2,
            #     'fitness': 0.0,
            #     'id': i
            # }
            population.append(individual)
        return population
    
    def save_checkpoint(self, generation):
        """Save complete simulation state to checkpoint file"""
        checkpoint_data = {
            'generation': generation,
            'population': self.population,
            'history': self.history,
            'parameters': self.get_parameters_dict(),
            'timestamp': datetime.now().isoformat()
        }
        
        checkpoint_file = os.path.join(self.checkpoint_dir, f'checkpoint_gen_{generation:06d}.pkl')
        with open(checkpoint_file, 'wb') as f:
            pickle.dump(checkpoint_data, f)
        
        # Also save as latest checkpoint
        latest_file = os.path.join(self.checkpoint_dir, 'latest_checkpoint.pkl')
        with open(latest_file, 'wb') as f:
            pickle.dump(checkpoint_data, f)
        
        print(f"Checkpoint saved: generation {generation} -> {checkpoint_file}")
    
    def load_checkpoint(self, checkpoint_file=None):
        """Load simulation state from checkpoint file"""
        if checkpoint_file is None:
            checkpoint_file = os.path.join(self.checkpoint_dir, 'latest_checkpoint.pkl')
        
        if not os.path.exists(checkpoint_file):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_file}")
        
        with open(checkpoint_file, 'rb') as f:
            checkpoint_data = pickle.load(f)
        
        self.generation = checkpoint_data['generation']
        self.population = checkpoint_data['population']
        self.history = checkpoint_data['history']
        
        # Verify parameters match (optional warning)
        saved_params = checkpoint_data['parameters']
        current_params = self.get_parameters_dict()
        
        param_diffs = []
        for key in current_params:
            if key in saved_params and saved_params[key] != current_params[key]:
                param_diffs.append(f"{key}: {saved_params[key]} -> {current_params[key]}")
        
        if param_diffs:
            print("WARNING: Parameter differences detected:")
            for diff in param_diffs:
                print(f"  {diff}")
        
        print(f"Checkpoint loaded: generation {self.generation} from {checkpoint_file}")
        return self.generation
    
    def evaluate_fitness_fast(self, population):
        """Fast fitness evaluation with GPU memory pool"""
        fitness_sum = np.zeros(self.num_agents, dtype=np.float32)
        
        # Create simulation instance once, reuse with proper time reset
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
                compute_size_x=256
            )
        
        sim = self._sim
        
        for run in range(self.nr):
            # Reset time state for consistent random seeds
            sim.current_time = 0.0
            
            self._reset_simulation_state(sim, population)
            
            # Run transient period using large batched timesteps (minimize kernel launches)
            sim.timestep_batched(self.tau_tr)
            
            # Reset gradient travel for fitness evaluation
            self._reset_gradient_travel(sim)
            
            # Run fitness evaluation period using large batched timesteps
            sim.timestep_batched(self.tau_fit)
            
            # Calculate fitness on GPU
            run_fitness = sim.calculate_fitness_gpu(
                pg=self.pg, ps=self.ps,
                omega_gc=self.omega_gc, omega_sc=self.omega_sc,
                fitness_eval_time=self.tau_fit
            )
            
            fitness_sum += run_fitness
        
        return fitness_sum / self.nr
    
    def _reset_simulation_state(self, sim, population):
        # Reset agents entirely on GPU 
        sim.reset_agents_gpu(population, self.world_size, self.r_a)
        sim.current_time = 0.0
    
    def _reset_gradient_travel(self, sim):
        """GPU-only gradient travel reset """
        sim.reset_gradient_travel_gpu()

    def adjust_fitness_with_bimodal_detection(self, population, fitness_values):
        fittable = np.array([[p["w_g"]] for p in population])
        gm = GaussianMixture(n_components=2).fit(fittable)
        gm1 = GaussianMixture(n_components=1).fit(fittable)
        if gm.score(fittable) / gm1.score(fittable) < 2: # Test if population bifurcated on w_g axs
            return np.array([-np.inf if np.random.rand() < self.leader_kill_prob else f for f in fitness_values])

        leader_index = np.argmax(gm.means_)
        out_fit = np.zeros_like(fitness_values)
        for i,p in enumerate(population):
            p_loc = [[p["w_g"]]]
            if gm.predict(p_loc) == leader_index and np.random.rand() <= self.leader_kill_prob:
                out_fit[i] = -np.inf
            else:
                out_fit[i] = fitness_values[i]
        return out_fit

    def adjust_fitness_roulette(self, population, fitness_values):
        w_g_list = np.array([p["w_g"] for p in population])
        total_w_g = np.sum(w_g_list)
        probabilities = w_g_list / total_w_g
        
        new_fitness = np.copy(fitness_values)
        for i in range(self.num_agents):
            if np.random.rand() >= self.leader_kill_prob:
                continue
            kill_idx = np.random.choice(len(population), p=probabilities)
            new_fitness[kill_idx] = -np.inf
        
        return new_fitness

    
    def tournament_selection(self, population, fitness_values, tournament_size=5):
        """Tournament selection preserves fitness differences"""
        new_population = []
        for i in range(self.num_agents):
            # Select random individuals for tournament
            tournament_indices = np.random.choice(len(population), size=tournament_size, replace=False)
            tournament_fitness = fitness_values[tournament_indices]
            
            # Winner is individual with highest fitness
            winner_idx = tournament_indices[np.argmax(tournament_fitness)]
            parent = population[winner_idx]
            
            offspring = {
                'w_g': parent['w_g'],
                'w_s': parent['w_s'],
                'fitness': 0.0,
                'id': i,
                'parent_id': parent['id']
            }
            new_population.append(offspring)
        
        return new_population

    def roulette_wheel_selection(self, population, fitness_values):
        """Fitness-proportional selection with fitness shifting (original method)"""
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

    def rank_selection(self, population, fitness_values):
        """Rank-based selection preserves fitness ordering"""
        # Convert fitness to ranks (1 = worst, n = best)
        ranks = np.argsort(np.argsort(fitness_values)) + 1
        
        # Linear ranking: probability proportional to rank
        probabilities = ranks / np.sum(ranks)
        
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

    def select_population(self, population, fitness_values):
        """Dispatch to appropriate selection method and preform leader killing when it is turned on"""
        if self.leader_kill_prob != 0.0:
            fitness_values = self.adjust_fitness_roulette(population, fitness_values)
        if self.selection_method == 'tournament':
            return self.tournament_selection(population, fitness_values)
        elif self.selection_method == 'roulette':
            return self.roulette_wheel_selection(population, fitness_values)
        elif self.selection_method == 'rank':
            return self.rank_selection(population, fitness_values)
        else:
            raise ValueError(f"Unknown selection method: {self.selection_method}")

    def mutate_population(self, population):
        """Apply Gaussian mutations"""
        for individual in population:
            # Gaussian mutation on gradient detection
            w_g_mutation = np.random.normal(0, self.sigma_mu)
            individual['w_g'] = max(0.0, individual['w_g'] + w_g_mutation)
            
            # Gaussian mutation on sociality
            w_s_mutation = np.random.normal(0, self.sigma_mu)
            individual['w_s'] = max(0.0, individual['w_s'] + w_s_mutation)
        
        return population
    
    def record_generation_stats(self, population, fitness_values):
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
    
    def run_evolution(self, num_generations, start_fresh=False, verbose=True):
        """
        Run evolutionary simulation with checkpointing support
        
        Args:
            num_generations: Total number of generations to run
            start_fresh: If True, ignore existing checkpoints and start from generation 0
            verbose: Print progress information
        """
        if verbose:
            print(f"\n=== CHECKPOINT EVOLUTIONARY SIMULATION ===")
        
        # Load existing checkpoint or initialize fresh population
        if not start_fresh:
            try:
                start_gen = self.load_checkpoint()
                if verbose:
                    print(f"Resuming from generation {start_gen}")
            except FileNotFoundError:
                if verbose:
                    print("No checkpoint found, starting fresh")
                start_fresh = True
        
        if start_fresh or self.population is None:
            self.generation = 0
            self.population = self._initialize_population()
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
        
        # Save parameters
        self.save_parameters()
        
        start_time = time.time()
        start_generation = self.generation
        
        if verbose:
            print(f"Running evolution from generation {start_generation} to {start_generation + num_generations}")
        
        for gen_offset in range(num_generations):
            self.generation = start_generation + gen_offset
            
            if verbose and self.generation % 1 == 0:
                elapsed = time.time() - start_time
                if gen_offset > 0:
                    est_total = elapsed * num_generations / gen_offset
                    remaining = est_total - elapsed
                    print(f"Generation {self.generation:3d} - {elapsed:.1f}s elapsed, ~{remaining:.1f}s remaining")
                else:
                    print(f"Generation {self.generation:3d} - starting...")
            
            # Evaluate fitness
            fitness_values = self.evaluate_fitness_fast(self.population)
            
            # Record statistics
            self.record_generation_stats(self.population, fitness_values)
            
            # Save checkpoint
            if self.generation % self.checkpoint_interval == 0:
                self.save_checkpoint(self.generation)
            
            # Evolution (skip on final generation)
            if gen_offset < num_generations - 1:
                # Selection
                new_population = self.select_population(self.population, fitness_values)
                
                # Mutation
                new_population = self.mutate_population(new_population)
                
                self.population = new_population
        
        # Save final checkpoint
        final_generation = start_generation + num_generations - 1
        self.generation = final_generation
        self.save_checkpoint(final_generation)
        
        # Save complete history as pickle file for plotting
        history_file = os.path.join(self.checkpoint_dir, 'complete_history.pkl')
        with open(history_file, 'wb') as f:
            pickle.dump({
                'history': self.history,
                'parameters': self.get_parameters_dict(),
                'final_generation': final_generation,
                'total_time': time.time() - start_time
            }, f)
        
        total_time = time.time() - start_time
        
        if verbose:
            print(f"\n=== EVOLUTION COMPLETE ===")
            print(f"Final generation: {final_generation}")
            print(f"Total time: {total_time:.1f} seconds")
            print(f"Average time per generation: {total_time/num_generations:.2f} seconds")
            print(f"Complete history saved to: {history_file}")
            
            final_w_g = self.history['w_g_distribution'][-1]
            final_w_s = self.history['w_s_distribution'][-1]
            print(f"\nFinal population:")
            print(f"w_g: mean={np.mean(final_w_g):.4f}, std={np.std(final_w_g):.4f}")
            print(f"w_s: mean={np.mean(final_w_s):.4f}, std={np.std(final_w_s):.4f}")
        
        return self.history


def run_evolution_from_params(params_dict_or_file, num_generations=None, start_fresh=False, checkpoint_dir=None):
    """
    Main function to run evolutionary simulation from parameters
    
    Args:
        params_dict_or_file: Either a dictionary of parameters or path to JSON file
        num_generations: Number of generations to run (overrides params file if specified)
        start_fresh: Start from generation 0, ignoring checkpoints
        checkpoint_dir: Specific checkpoint directory (if resuming)
    
    Returns:
        history: Evolution history dictionary
    """
    # Load parameters
    if isinstance(params_dict_or_file, str):
        # Load from JSON file
        with open(params_dict_or_file, 'r') as f:
            params = json.load(f)
        print(f"Parameters loaded from {params_dict_or_file}")
    else:
        # Use provided dictionary
        params = params_dict_or_file
    
    # Extract num_generations from params if not explicitly provided
    if num_generations is None:
        num_generations = params.get('num_generations', 100)
        print(f"Using num_generations from params: {num_generations}")
    else:
        print(f"Using specified num_generations: {num_generations}")
    
    # Override checkpoint_dir if provided
    if checkpoint_dir is not None:
        params['checkpoint_dir'] = checkpoint_dir
    
    # Keep only necessary parameters for constructor
    necessary_params = {
        'world_size', 'num_agents', 'density', 'pg', 'ps', 'omega_gc', 'omega_sc',
        'sigma_mu', 'tau_tr', 'tau_fit', 'nr', 'checkpoint_interval', 'checkpoint_dir', 'selection_method', "leader_kill_prob"
    }
    simulation_params = {k: v for k, v in params.items() if k in necessary_params}
    
    # Create simulation instance
    sim = CheckpointEvolutionSimulation(**simulation_params)
    
    # Run evolution
    history = sim.run_evolution(num_generations, start_fresh=start_fresh, verbose=True)
    
    return history


def create_default_params_file(filename="evolution_params.json"):
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
        "num_generations": 300,
        "selection_method": "tournament",
        "leader_kill_prob": 0.0
    }
    
    with open(filename, 'w') as f:
        json.dump(default_params, f, indent=2)
    
    print(f"Default parameters saved to {filename}")
    return filename


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Run with parameters from JSON file
        params_file = sys.argv[1]
        num_gens = int(sys.argv[2]) if len(sys.argv) > 2 else None  # Use None to read from params file
        history = run_evolution_from_params(params_file, num_gens)
    else:
        # Create default params and run with them
        params_file = create_default_params_file()
        print(f"Running evolution with default parameters from {params_file}")
        history = run_evolution_from_params(params_file)  # Will read num_generations from file