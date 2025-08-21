#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from run_simple import SimplifiedSimulation, Agent
import time

class LeaderFollowerSimulation(SimplifiedSimulation):
    def __init__(self, world_size: float, num_agents: int, num_leaders: int = 5,
                 r_a: float = 0.05, r_s: float = 0.3, s: float = 0.1, 
                 sigma_g: float = 0.1, sigma_r: float = 0.1, theta_max: float = 2.0, 
                 dt: float = 0.01, tp: float = 100.0, compute_size_x: int = 32):
        self.num_leaders = num_leaders
        super().__init__(world_size, num_agents, r_a, r_s, s, sigma_g, sigma_r, 
                         theta_max, dt, tp, compute_size_x)
    
    def _generate_random_agents(self, num_agents):
        """Generate agents with leaders having stronger gradient detection"""
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
            
            if i < self.num_leaders:
                w_g = 2.0 
                w_s = 0.0
            else:
                w_g = 0.0 
                w_s = np.random.uniform(0.3,1.0)

            agents.append(Agent(x, y, vx, vy, w_g, w_s, theta))
        return agents
    
    def get_leader_follower_positions(self):
        """Get positions separated by leader/follower status"""
        positions = self.get_agent_positions()
        leader_positions = positions[:self.num_leaders]
        follower_positions = positions[self.num_leaders:]
        return leader_positions, follower_positions

def create_leader_follower_animation(num_agents=100, num_leaders=8, timesteps=1500):
    """
    Create animation showing leader-follower dynamics
    """
    
    print(f"Creating leader-follower animation: {num_agents} agents ({num_leaders} leaders), {timesteps} timesteps")
    
    # Simulation parameters optimized for leader-follower dynamics
    world_size = 2.0
    r_a = 0.02  # Small avoidance zone
    r_s = 0.12   # Medium socialization zone
    
    print(f"Parameters: r_a={r_a}, r_s={r_s}, leaders={num_leaders}")
    
    # Create simulation
    sim = LeaderFollowerSimulation(
        world_size=world_size,
        num_agents=num_agents,
        num_leaders=num_leaders,
        r_a=r_a,
        r_s=r_s,
        s=r_a,  # Slower movement for better visualization
        sigma_g=np.sqrt(2),
        sigma_r=0.1,
        theta_max=1.5,
        dt=0.15,    
        tp=0.0
    )
    
    # Storage for animation data
    leader_positions_history = []
    follower_positions_history = []
    save_interval = 5  # Save every 8 steps for smooth animation
    
    print("Running simulation...")
    start_time = time.time()
    
    for step in range(timesteps):
        # Use two-pass grid-based O(N) timestep for optimal performance
        sim.timestep_gpu_two_pass_grid()
        
        # Save data for animation
        if step % save_interval == 0:
            leader_pos, follower_pos = sim.get_leader_follower_positions()
            leader_positions_history.append(leader_pos.copy())
            follower_positions_history.append(follower_pos.copy())
        
        # Progress
        if step % 150 == 0:
            print(f"Step {step}/{timesteps}")
    
    runtime = time.time() - start_time
    print(f"Simulation completed in {runtime:.2f}s ({timesteps/runtime:.0f} steps/sec)")
    
    sim.cleanup()
    
    # Create leader-follower animation
    print("Creating animation...")
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Animation function
    def animate(frame):
        ax.clear()
        
        leader_pos = leader_positions_history[frame]
        follower_pos = follower_positions_history[frame]
        
        # Plot followers as smaller blue dots
        ax.scatter(follower_pos[:, 0], follower_pos[:, 1], 
                  s=25, c='lightblue', alpha=0.7, label='Followers', edgecolors='blue', linewidths=0.5)
        
        # Plot leaders as larger red dots with distinct markers
        ax.scatter(leader_pos[:, 0], leader_pos[:, 1], 
                  s=80, c='red', alpha=0.9, label='Leaders', marker='*', edgecolors='darkred', linewidths=1)
        
        ax.set_xlim(0, world_size)
        ax.set_ylim(0, world_size)
        ax.set_aspect('equal')
        ax.set_title(f'Leader-Follower Dynamics (Two-Pass Grid O(N)) - Frame {frame}\n{num_leaders} Leaders (red stars) + {num_agents-num_leaders} Followers (blue dots)', 
                    fontsize=12, pad=15)
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Add frame counter
        ax.text(0.02, 0.98, f'Step: {frame * save_interval}', 
                transform=ax.transAxes, fontsize=10, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Create animation
    num_frames = len(leader_positions_history)
    print(f"Animation: {num_frames} frames")
    
    anim = animation.FuncAnimation(fig, animate, frames=num_frames, 
                                 interval=60, blit=False, repeat=True)
    
    # Save as MP4
    try:
        anim.save('leader_follower_dynamics.mp4', writer='ffmpeg', fps=30, bitrate=2400, dpi=120)
        print("✓ Saved as 'leader_follower_dynamics.mp4'")
    except Exception as e:
        print(f"Could not save MP4: {e}")
    
    plt.show()
    return anim

if __name__ == "__main__":
    create_leader_follower_animation()