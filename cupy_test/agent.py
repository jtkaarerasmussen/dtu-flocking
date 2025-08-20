#!/usr/bin/env python3

import numpy as np

class Agent:
    """Simple agent class for CuPy simulation"""
    
    def __init__(self, x: float, y: float, vx: float = 0.0, vy: float = 0.0, 
                 w_g: float = 1.0, w_s: float = 1.0):
        self.pos = np.array([x, y], dtype=np.float32)
        self.vel = np.array([vx, vy], dtype=np.float32)
        self.w_g = np.float32(w_g)
        self.w_s = np.float32(w_s)
    
    @staticmethod
    def create_random_agents(num_agents: int, world_size: float, speed: float) -> np.ndarray:
        """Create array of random agents for GPU processing"""
        # Structure: [x, y, vx, vy, w_g, w_s] for each agent
        agents = np.zeros((num_agents, 6), dtype=np.float32)
        
        # Random positions
        agents[:, 0] = np.random.uniform(0, world_size, num_agents)  # x
        agents[:, 1] = np.random.uniform(0, world_size, num_agents)  # y
        
        # Random velocities with constant speed
        angles = np.random.uniform(0, 2 * np.pi, num_agents)
        agents[:, 2] = speed * np.cos(angles)  # vx
        agents[:, 3] = speed * np.sin(angles)  # vy
        
        # Agent parameters (mix of gradient followers and social agents)
        agents[:, 4] = np.where(np.arange(num_agents) % 2 == 0, 0.45, 0.0)  # w_g
        agents[:, 5] = np.where(np.arange(num_agents) % 2 == 0, 0.2, 1.0)   # w_s
        
        return agents