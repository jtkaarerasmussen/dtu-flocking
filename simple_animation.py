#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from run_simple import SimplifiedSimulation
import time

def create_simple_flocking_animation(num_agents=16384, timesteps=2000):
    """
    Simple single-panel animation showing just the flocking behavior
    """
    
    print(f"Creating simple flocking animation: {num_agents} agents, {timesteps} timesteps")
    
    # Simple simulation parameters
    world_size = 2.0
    r_a = 0.005  # Zone of avoidance
    r_s = 0.03   # Zone of socialization
    
    print(f"Parameters: r_a={r_a}, r_s={r_s}")
    
    # Create simulation
    sim = SimplifiedSimulation(
        world_size=world_size,
        num_agents=num_agents,
        r_a=r_a,
        r_s=r_s,
        s=r_a*0.2, 
        sigma_g=np.sqrt(2),
        sigma_r=0.1,
        theta_max=1.0,
        dt=0.2,    
        tp=0.0
    )
    
    # Storage for animation data - save every few steps for smooth playback
    positions_history = []
    # save_interval = max(1, timesteps // 400)  
    save_interval = 10
    
    print("Running simulation...")
    start_time = time.time()
    
    for step in range(timesteps):
        sim.timestep_gpu_only()
        
        # Save data for animation
        if step % save_interval == 0:
            positions = sim.get_agent_positions()
            positions_history.append(positions.copy())
        
        # Progress
        if step % 100 == 0:
            print(f"Step {step}/{timesteps}")
    
    runtime = time.time() - start_time
    print(f"Simulation completed in {runtime:.2f}s ({timesteps/runtime:.0f} steps/sec)")
    
    sim.cleanup()
    
    # Create simple animation
    print("Creating animation...")
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Animation function
    def animate(frame):
        ax.clear()
        positions = positions_history[frame]
        
        # Plot agents as simple dots
        ax.scatter(positions[:, 0], positions[:, 1], s=20, c='blue', alpha=0.8)
        
        ax.set_xlim(0, world_size)
        ax.set_ylim(0, world_size)
        ax.set_aspect('equal')
        ax.set_title(f'Flocking Simulation - Frame {frame}')
        ax.grid(True, alpha=0.3)
    
    # Create animation
    num_frames = len(positions_history)
    print(f"Animation: {num_frames} frames")
    
    anim = animation.FuncAnimation(fig, animate, frames=num_frames, 
                                 interval=50, blit=False, repeat=True)
    
    # Save as GIF
    # print("Saving GIF...")
    # try:
    #     anim.save('simple_flocking.gif', writer='pillow', fps=25, dpi=100)
    #     print("✓ Saved as 'simple_flocking.gif'")
    # except Exception as e:
    #     print(f"Could not save GIF: {e}")
    
    # Save as MP4 if possible
    try:
        anim.save('simple_flocking.mp4', writer='ffmpeg', fps=20, bitrate=1800)
        print("✓ Saved as 'simple_flocking.mp4'")
    except Exception as e:
        print(f"Could not save MP4: {e}")
    
    plt.show()
    return anim

if __name__ == "__main__":
    create_simple_flocking_animation()