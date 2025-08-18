#!/usr/bin/env python3

import moderngl
import numpy as np
import struct
import math
from typing import List, Tuple
import time

class Agent:
    def __init__(self, x: float, y: float, vx: float = 0.0, vy: float = 0.0, 
                 w_g: float = 1.0, w_s: float = 1.0, theta: float = 0.0, grad_travel: float = 0.0):
        self.c = np.array([x, y], dtype=np.float32)  # position
        self.v = np.array([vx, vy], dtype=np.float32)  # velocity
        self.w_g = np.float32(w_g)
        self.w_s = np.float32(w_s)
        self.theta = np.float32(theta)
        self.grad_travel = np.float32(grad_travel)
    
    def to_bytes(self) -> bytes:
        """Convert agent to binary format for GPU buffer"""
        return struct.pack('ffffffff', 
                          self.c[0], self.c[1],  # vec2 c
                          self.v[0], self.v[1],  # vec2 v
                          self.w_g, self.w_s, self.theta, self.grad_travel)

class SimplifiedSimulation:
    def __init__(self, world_size: float, num_agents: int, 
                 r_a: float = 0.05, r_s: float = 0.3, s: float = 0.1, 
                 sigma_g: float = 0.1, sigma_r: float = 0.1, theta_max: float = 2.0, 
                 dt: float = 0.01, tp: float = 100.0, compute_size_x: int = 32):
        """
        Simplified simulation without grid system
        
        Args:
            world_size: Size of the simulation world
            num_agents: Number of agents
            r_a: Zone of avoidance radius
            r_s: Zone of socialization radius  
            s: Constant speed
            sigma_g: Gradient detection noise
            sigma_r: Random motion noise
            theta_max: Maximum turning rate
            dt: Time step
            tp: Time period for boundary perturbation
            compute_size_x: Compute shader work group size
        """
        self.world_size = world_size
        self.num_agents = num_agents
        self.current_time = 0.0
        self.dt = dt
        self.compute_size_x = compute_size_x
        
        # Initialize OpenGL context
        try:
            self.ctx = moderngl.create_context(standalone=True, backend='egl')
        except Exception:
            self.ctx = moderngl.create_context(standalone=True)
        
        # Initialize simulation parameters
        self.sim_params = {
            'r_a': r_a,
            'r_s': r_s, 
            's': s,
            'l': world_size,
            't': self.current_time,
            'sigma_g': sigma_g,
            'sigma_r': sigma_r,
            'theta_max': theta_max,
            'dt': dt,
            'world_size': world_size,
            'tp': tp
        }
        
        # Load simulation shader
        self._load_sim_shader()
        
        # Load fitness shader
        self._load_fitness_shader()
        
        # Initialize agents and buffers
        self.agents = self._generate_random_agents(num_agents)
        self._setup_buffers()
        
        # Load batched shader
        self._load_batched_shader()
        
    
    def _load_sim_shader(self):
        """Load and compile the simplified simulation compute shader"""
        with open('sim_simple.comp', 'r') as f:
            shader_source = f.read()
        
        shader_source = shader_source.replace('COMPUTE_SIZE_X', str(self.compute_size_x))
        shader_source = shader_source.replace('COMPUTE_SIZE_Y', '1')
        
        self.sim_shader = self.ctx.compute_shader(shader_source)
    
    def _load_batched_shader(self):
        """Load and compile the batched simulation compute shader"""
        with open('sim_batched.comp', 'r') as f:
            shader_source = f.read()
        
        shader_source = shader_source.replace('COMPUTE_SIZE_X', str(self.compute_size_x))
        shader_source = shader_source.replace('COMPUTE_SIZE_Y', '1')
        
        self.batched_shader = self.ctx.compute_shader(shader_source)
    
    def _load_fitness_shader(self):
        """Load and compile the fitness compute shader"""
        with open('fitness.comp', 'r') as f:
            shader_source = f.read()
        
        shader_source = shader_source.replace('COMPUTE_SIZE_X', str(self.compute_size_x))
        shader_source = shader_source.replace('COMPUTE_SIZE_Y', '1')
        
        self.fitness_shader = self.ctx.compute_shader(shader_source)
    
    def _generate_random_agents(self, num_agents: int) -> List[Agent]:
        """Generate random agents within world bounds"""
        agents = []
        
        for i in range(num_agents):
            x = np.random.uniform(0, self.world_size)
            y = np.random.uniform(0, self.world_size)
            
            # Random initial velocity direction
            theta = np.random.uniform(np.pi, 2 * np.pi)
            vx = self.sim_params['s'] * np.cos(theta)
            vy = self.sim_params['s'] * np.sin(theta)
            
            # Agent parameters (mostly for testing)
            if i % 2 == 0:
                w_g = 0.45  
                w_s = 0.2  
            else:
                w_g = 0
                w_s = 1.0
            
            agents.append(Agent(x, y, vx, vy, w_g, w_s, theta))
        
        return agents
    
    def _setup_buffers(self):
        """Setup persistent GPU buffers for simulation"""
        # Agent data buffers (ping-pong buffer system)
        agent_data = b''.join(agent.to_bytes() for agent in self.agents)
        self.agents_input_buffer = self.ctx.buffer(agent_data)
        self.agents_output_buffer = self.ctx.buffer(agent_data) 
        
        # Parameters buffer (persistent)
        self._update_params_buffer()
        
        # Batch parameters buffer for batched shader
        batch_data = struct.pack('if', 1, 0.0)  # num_timesteps=1, start_time=0.0
        self.batch_params_buffer = self.ctx.buffer(batch_data)
    
    def _update_params_buffer(self):
        """Update parameters buffer with current simulation time"""
        self.sim_params['t'] = self.current_time
        param_data = struct.pack('fffffffffff', 
                                self.sim_params['r_a'],
                                self.sim_params['r_s'],
                                self.sim_params['s'],
                                self.sim_params['l'],
                                self.sim_params['t'],
                                self.sim_params['sigma_g'],
                                self.sim_params['sigma_r'],
                                self.sim_params['theta_max'],
                                self.sim_params['dt'],
                                self.sim_params['world_size'],
                                self.sim_params['tp'])
        
        if hasattr(self, 'params_buffer'):
            self.params_buffer.write(param_data)
        else:
            self.params_buffer = self.ctx.buffer(param_data)
    
    def timestep_gpu_only(self, sync_gpu: bool = False):
        """
        Pure GPU timestep - no CPU transfers, maximum performance
        Uses simplified O(N²) neighbor search instead of complex grid system
        
        Args:
            sync_gpu: If True, wait for GPU to complete (for accurate timing)
        """
        # Update time parameter
        self._update_params_buffer()
        
        # Bind buffers to shader
        self.agents_input_buffer.bind_to_storage_buffer(0)   # Input agents
        self.params_buffer.bind_to_storage_buffer(2)         # Parameters
        self.agents_output_buffer.bind_to_storage_buffer(5)  # Output agents
        
        # Execute simulation shader (single pass)
        num_work_groups = (self.num_agents + self.compute_size_x - 1) // self.compute_size_x
        self.sim_shader.run(num_work_groups, 1, 1)
        
        # Force GPU synchronization
        if sync_gpu:
            self.ctx.finish()
        
        self.ctx.memory_barrier()
        
        # Update time and swap buffers
        self.current_time += self.dt
        self.agents_input_buffer, self.agents_output_buffer = self.agents_output_buffer, self.agents_input_buffer
    
    def timestep_gpu_batched(self, batch_size: int = 10, sync_gpu: bool = False):
        """Run multiple timesteps in a single GPU dispatch for better performance"""
        # Update batch parameters
        batch_data = struct.pack('if', batch_size, self.current_time)
        self.batch_params_buffer.write(batch_data)
        
        # Bind buffers for batched shader
        self.agents_input_buffer.bind_to_storage_buffer(0)    # Input agents
        self._update_params_buffer()
        self.params_buffer.bind_to_storage_buffer(2)          # Simulation parameters
        self.batch_params_buffer.bind_to_storage_buffer(6)    # Batch parameters  
        self.agents_output_buffer.bind_to_storage_buffer(5)   # Output agents
        
        # Execute batched simulation shader
        num_work_groups = (self.num_agents + self.compute_size_x - 1) // self.compute_size_x
        self.batched_shader.run(num_work_groups, 1, 1)
        
        # Force GPU synchronization 
        if sync_gpu:
            self.ctx.finish()
        
        # Memory barrier for correct grad_travel accumulation
        self.ctx.memory_barrier()
        
        # Update time and swap buffers
        self.current_time += batch_size * self.dt
        self.agents_input_buffer, self.agents_output_buffer = self.agents_output_buffer, self.agents_input_buffer

    def timestep(self):
        """Legacy timestep method with CPU synchronization (for debugging)"""
        self.timestep_gpu_only()
        
        # Read back agent data for debugging (slow af)
        result = self.agents_input_buffer.read()
        self._update_agents_from_buffer(result)
    
    def _update_agents_from_buffer(self, buffer_data: bytes):
        """Update agent objects from GPU buffer data"""
        for i in range(self.num_agents):
            offset = i * 32  # 8 floats * 4 bytes each
            data = struct.unpack('ffffffff', buffer_data[offset:offset + 32])
            
            self.agents[i].c[0] = data[0]  # x position
            self.agents[i].c[1] = data[1]  # y position  
            self.agents[i].v[0] = data[2]  # x velocity
            self.agents[i].v[1] = data[3]  # y velocity
            self.agents[i].w_g = data[4]   # gradient weight
            self.agents[i].w_s = data[5]   # social weight
            self.agents[i].theta = data[6] # orientation
            self.agents[i].grad_travel = data[7] # gradient travel
    
    def get_agent_positions(self) -> np.ndarray:
        """Get current agent positions as numpy array (requires CPU sync)"""
        result = self.agents_input_buffer.read()
        self._update_agents_from_buffer(result)
        
        positions = np.zeros((self.num_agents, 2), dtype=np.float32)
        for i, agent in enumerate(self.agents):
            positions[i] = agent.c
        return positions
    
    def get_agent_velocities(self) -> np.ndarray:
        """Get current agent velocities as numpy array (requires CPU sync)"""
        result = self.agents_input_buffer.read()
        self._update_agents_from_buffer(result)
        
        velocities = np.zeros((self.num_agents, 2), dtype=np.float32)
        for i, agent in enumerate(self.agents):
            velocities[i] = agent.v
        return velocities
    
    def calculate_fitness_gpu(self, pg: float = 0.01, ps: float = 0.01, 
                             omega_gc: float = 1.0, omega_sc: float = 1.0,
                             fitness_eval_time: float = None) -> np.ndarray:
        """
        GPU-accelerated fitness calculation using compute shader
        
        Args:
            pg: gradient detection cost parameter
            ps: sociality cost parameter  
            omega_gc: gradient detection scaling parameter
            omega_sc: sociality scaling parameter
            fitness_eval_time: Time period over which fitness was evaluated
            
        Returns:
            Array of fitness values for each agent
        """
        # Force GPU synchronization
        self.ctx.finish()
        
        # Determine evaluation time
        if fitness_eval_time is not None:
            eval_time = fitness_eval_time
        else:
            eval_time = self.current_time
            
        # Create fitness parameters buffer
        fitness_params_data = struct.pack('ffffff', 
                                        pg, ps, omega_gc, omega_sc, 
                                        eval_time, self.sim_params['s'])
        fitness_params_buffer = self.ctx.buffer(fitness_params_data)
        
        # Create output fitness buffer
        fitness_output_data = np.zeros(self.num_agents, dtype=np.float32)
        fitness_output_buffer = self.ctx.buffer(fitness_output_data.tobytes())
        
        # Bind buffers to shader
        self.agents_input_buffer.bind_to_storage_buffer(0)    # Agent data
        fitness_params_buffer.bind_to_storage_buffer(1)       # Fitness parameters
        fitness_output_buffer.bind_to_storage_buffer(2)       # Output fitness values
        
        # Execute fitness shader
        num_work_groups = (self.num_agents + self.compute_size_x - 1) // self.compute_size_x
        self.fitness_shader.run(num_work_groups, 1, 1)
        
        # Read back fitness values
        fitness_result = fitness_output_buffer.read()
        fitness_values = np.frombuffer(fitness_result, dtype=np.float32)
        
        # Cleanup
        fitness_params_buffer.release()
        fitness_output_buffer.release()
        
        return fitness_values
    
    def _update_gpu_buffers_from_agents(self):
        """Update GPU buffers with current agent data"""
        agent_data = b''.join(agent.to_bytes() for agent in self.agents)
        self.agents_input_buffer.write(agent_data)
        self.agents_output_buffer.write(agent_data)
    
    def cleanup(self):
        """Release GPU resources"""
        self.agents_input_buffer.release()
        self.agents_output_buffer.release()
        self.params_buffer.release()
        self.ctx.release()

def generate_random_agents(num_agents: int, world_size: float, seed: int = 42) -> List[Agent]:
    """Generate random agents within world bounds"""
    agents = []
    np.random.seed(seed)
    
    for _ in range(num_agents):
        x = np.random.uniform(0, world_size)
        y = np.random.uniform(0, world_size)
        agents.append(Agent(x, y))
    
    return agents

if __name__ == "__main__":
    # Performance comparison test
    print("=== SIMPLIFIED SIMULATION PERFORMANCE TEST ===")
    
    WORLD_SIZE = 2.0
    NUM_AGENTS = 16384  
    NUM_TIMESTEPS = 1000
    DENSITY = 0.01
    
    # Calculate r_a from density: ra = l*sqrt(ρ/N)
    R_A = WORLD_SIZE * math.sqrt(DENSITY / NUM_AGENTS)
    
    print(f"Testing with {NUM_AGENTS} agents for {NUM_TIMESTEPS} timesteps")
    print(f"Zone parameters: r_a = {R_A:.4f}, r_s = {6.0 * R_A:.4f}")
    
    # Initialize simulation
    start_time = time.time()
    sim = SimplifiedSimulation(
        world_size=WORLD_SIZE,
        num_agents=NUM_AGENTS,
        r_a=R_A,
        r_s=6.0 * R_A,
        s=R_A,
        sigma_g=0.447,  # sqrt(0.2)
        sigma_r=1.0,
        theta_max=2.0,
        dt=0.2,
        tp=100.0
    )
    init_time = time.time() - start_time
    
    print(f"Initialization: {init_time:.3f} seconds")
    
    # Performance test 
    print(f"\nRunning {NUM_TIMESTEPS} GPU-only timesteps...")
    start_time = time.time()
    
    for i in range(NUM_TIMESTEPS):
        sim.timestep_gpu_only()
        
        if (i + 1) % 100 == 0:
            elapsed = time.time() - start_time
            steps_per_sec = (i + 1) / elapsed
            print(f"Step {i+1:4d}: {steps_per_sec:.1f} timesteps/sec")
    
    total_time = time.time() - start_time
    final_steps_per_sec = NUM_TIMESTEPS / total_time
    
    print(f"\n=== PERFORMANCE RESULTS ===")
    print(f"Total time: {total_time:.3f} seconds")
    print(f"Average: {final_steps_per_sec:.1f} timesteps/sec")
    print(f"Agent-timesteps/sec: {NUM_AGENTS * final_steps_per_sec:.0f}")
    
    
    sim.cleanup()