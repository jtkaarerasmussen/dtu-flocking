#!/usr/bin/env python3

import cupy as cp
import numpy as np
import time
from agent import Agent

class CuPySimulation:
    """Simple CuPy-based flocking simulation for A100 optimization"""
    
    def __init__(self, world_size: float, num_agents: int, 
                 r_a: float = 0.05, r_s: float = 0.3, s: float = 0.1,
                 sigma_r: float = 0.1, theta_max: float = 2.0, dt: float = 0.01,
                 block_size: int = 256):
        
        self.world_size = world_size
        self.num_agents = num_agents
        self.r_a = r_a
        self.r_s = r_s
        self.s = s
        self.sigma_r = sigma_r
        self.theta_max = theta_max
        self.dt = dt
        self.block_size = block_size
        
        print(f"CuPy Simulation: {num_agents} agents, block_size={block_size}")
        print(f"Parameters: r_a={r_a:.4f}, r_s={r_s:.4f}, s={s:.4f}")
        
        # Initialize agents on CPU then transfer to GPU
        cpu_agents = Agent.create_random_agents(num_agents, world_size, s)
        self.agents = cp.asarray(cpu_agents)
        self.agents_next = cp.zeros_like(self.agents)
        
        # Compile CUDA kernel
        self._compile_kernel()
    
    def _compile_kernel(self):
        """Compile the CUDA kernel for flocking simulation"""
        
        kernel_code = '''
        extern "C" __global__ void flocking_timestep(
            float* agents_in,     // Input: [x, y, vx, vy, w_g, w_s] * num_agents
            float* agents_out,    // Output: [x, y, vx, vy, w_g, w_s] * num_agents
            int num_agents,
            float world_size,
            float r_a, float r_s, float s,
            float sigma_r, float theta_max, float dt,
            unsigned int time_seed
        ) {
            int agent_id = blockIdx.x * blockDim.x + threadIdx.x;
            if (agent_id >= num_agents) return;
            
            // Current agent data (stride = 6)
            int idx = agent_id * 6;
            float x = agents_in[idx + 0];
            float y = agents_in[idx + 1];
            float vx = agents_in[idx + 2];
            float vy = agents_in[idx + 3];
            float w_g = agents_in[idx + 4];
            float w_s = agents_in[idx + 5];
            
            // Social force accumulators
            float avoid_x = 0.0f, avoid_y = 0.0f;
            float social_x = 0.0f, social_y = 0.0f;
            float align_x = 0.0f, align_y = 0.0f;
            bool has_avoidance = false;
            
            float r_a_sq = r_a * r_a;
            float r_s_sq = r_s * r_s;
            
            // Check all other agents for interactions
            for (int other_id = 0; other_id < num_agents; other_id++) {
                if (other_id == agent_id) continue;
                
                int other_idx = other_id * 6;
                float ox = agents_in[other_idx + 0];
                float oy = agents_in[other_idx + 1];
                float ovx = agents_in[other_idx + 2];
                float ovy = agents_in[other_idx + 3];
                
                // Calculate wrapped distance (periodic boundaries)
                float dx = ox - x;
                float dy = oy - y;
                
                // Periodic wrapping
                dx = dx - world_size * floorf(dx / world_size + 0.5f);
                dy = dy - world_size * floorf(dy / world_size + 0.5f);
                
                float dist_sq = dx * dx + dy * dy;
                if (dist_sq > r_s_sq || dist_sq < 1e-6f) continue;
                
                float dist = sqrtf(dist_sq);
                
                // Avoidance (highest priority)
                if (dist_sq <= r_a_sq) {
                    has_avoidance = true;
                    avoid_x += -dx / dist;
                    avoid_y += -dy / dist;
                }
                // Social interaction
                else if (dist_sq <= r_s_sq) {
                    // Attraction
                    social_x += dx / dist;
                    social_y += dy / dist;
                    
                    // Alignment
                    float other_speed = sqrtf(ovx * ovx + ovy * ovy);
                    if (other_speed > 1e-3f) {
                        align_x += ovx / other_speed;
                        align_y += ovy / other_speed;
                    }
                }
            }
            
            // Compute desired direction
            float desired_x, desired_y;
            
            if (has_avoidance) {
                // Pure avoidance
                float avoid_len = sqrtf(avoid_x * avoid_x + avoid_y * avoid_y);
                if (avoid_len > 0.0f) {
                    desired_x = avoid_x / avoid_len;
                    desired_y = avoid_y / avoid_len;
                } else {
                    desired_x = 1.0f;
                    desired_y = 0.0f;
                }
            } else {
                // Combine social forces and random noise
                float combined_x = w_s * (social_x + align_x);
                float combined_y = w_s * (social_y + align_y);
                
                // Add random direction (simplified - no proper RNG for now)
                unsigned int seed = (agent_id * 1000 + time_seed) % 32768;
                float rand_x = (float(seed % 1000) / 500.0f - 1.0f) * sigma_r;
                seed = (seed * 73 + 456) % 32768;
                float rand_y = (float(seed % 1000) / 500.0f - 1.0f) * sigma_r;
                
                combined_x += rand_x;
                combined_y += rand_y;
                
                // Normalize desired direction
                float combined_len = sqrtf(combined_x * combined_x + combined_y * combined_y);
                if (combined_len > 0.0f) {
                    desired_x = combined_x / combined_len;
                    desired_y = combined_y / combined_len;
                } else {
                    desired_x = 1.0f;
                    desired_y = 0.0f;
                }
            }
            
            // Apply turning rate limit
            float current_speed = sqrtf(vx * vx + vy * vy);
            float current_dir_x = (current_speed > 1e-3f) ? vx / current_speed : 1.0f;
            float current_dir_y = (current_speed > 1e-3f) ? vy / current_speed : 0.0f;
            
            float turn_rate = theta_max * dt;
            float new_dir_x = current_dir_x + turn_rate * (desired_x - current_dir_x);
            float new_dir_y = current_dir_y + turn_rate * (desired_y - current_dir_y);
            
            // Normalize and set constant speed
            float new_len = sqrtf(new_dir_x * new_dir_x + new_dir_y * new_dir_y);
            if (new_len > 0.0f) {
                new_dir_x /= new_len;
                new_dir_y /= new_len;
            }
            
            float new_vx = new_dir_x * s;
            float new_vy = new_dir_y * s;
            
            // Update position
            float new_x = x + new_vx * dt;
            float new_y = y + new_vy * dt;
            
            // Periodic boundaries
            if (new_x >= world_size) new_x -= world_size;
            else if (new_x < 0.0f) new_x += world_size;
            
            if (new_y >= world_size) new_y -= world_size;
            else if (new_y < 0.0f) new_y += world_size;
            
            // Write output
            agents_out[idx + 0] = new_x;
            agents_out[idx + 1] = new_y;
            agents_out[idx + 2] = new_vx;
            agents_out[idx + 3] = new_vy;
            agents_out[idx + 4] = w_g;  // Unchanged
            agents_out[idx + 5] = w_s;  // Unchanged
        }
        '''
        
        self.kernel = cp.RawKernel(kernel_code, 'flocking_timestep')
        print("CUDA kernel compiled successfully")
    
    def timestep(self):
        """Execute one simulation timestep"""
        # Calculate grid dimensions
        grid_size = (self.num_agents + self.block_size - 1) // self.block_size
        
        # Generate time seed for randomness
        time_seed = int(time.time() * 1000) % 100000
        
        # Execute kernel
        self.kernel(
            (grid_size,), (self.block_size,),
            (self.agents, self.agents_next, 
             self.num_agents, self.world_size,
             self.r_a, self.r_s, self.s,
             self.sigma_r, self.theta_max, self.dt,
             time_seed)
        )
        
        # Swap buffers
        self.agents, self.agents_next = self.agents_next, self.agents
    
    def get_positions(self):
        """Get current agent positions (CPU copy)"""
        cpu_agents = cp.asnumpy(self.agents)
        return cpu_agents[:, :2]  # Return only [x, y] columns
    
    def benchmark(self, num_timesteps: int = 1000):
        """Benchmark simulation performance"""
        print(f"\nBenchmarking {num_timesteps} timesteps...")
        
        # Warmup
        for _ in range(10):
            self.timestep()
        cp.cuda.Stream.null.synchronize()
        
        # Actual benchmark
        start_time = time.time()
        for _ in range(num_timesteps):
            self.timestep()
        cp.cuda.Stream.null.synchronize()  # Force GPU to finish
        end_time = time.time()
        
        total_time = end_time - start_time
        timesteps_per_sec = num_timesteps / total_time
        agent_timesteps_per_sec = self.num_agents * timesteps_per_sec
        
        print(f"Results:")
        print(f"  Total time: {total_time:.3f} seconds")
        print(f"  Timesteps/sec: {timesteps_per_sec:.1f}")
        print(f"  Agent-timesteps/sec: {agent_timesteps_per_sec:.0f}")
        
        return timesteps_per_sec