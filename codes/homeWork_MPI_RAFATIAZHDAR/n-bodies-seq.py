import sys
import time
import numpy as np
from n_bodies import init_world, interaction, update, signature

def n_bodies_seq(N, NB_STEPS, DISPLAY=True):
    # Initialize the world with N bodies
    data = init_world(N)
    
    if DISPLAY:
        from n_bodies import displayPlot
        displayPlot(data)
    
    start_time = time.time()
    
    # Simulation loop
    for _ in range(NB_STEPS):
        # Compute forces
        force = np.zeros((N, 2))  # Array to store resultant forces for each body
        for i in range(N):
            for j in range(N):
                if i != j:
                    force[i] += interaction(data[i], data[j])
        
        # Update positions and velocities
        for i in range(N):
            data[i] = update(data[i], force[i])
    
    end_time = time.time()
    
    # Output results
    print("Duration:", end_time - start_time)
    print("Signature: %.4e" % signature(data))
    print("Unbalance: 0")  # Unbalance is always 0 for the sequential version

if __name__ == "__main__":
    # Read inputs
    N = int(sys.argv[1])          # Number of bodies
    NB_STEPS = int(sys.argv[2])   # Number of steps
    DISPLAY = len(sys.argv) != 4  # No display if a third argument is provided
    
    n_bodies_seq(N, NB_STEPS, DISPLAY)
