from mpi4py import MPI
import sys
import math
import random
import time
import numpy as np
from n_bodies import init_world, interaction, update, signature

# Helper functions to split and merge data
def split(x, size):
    n = math.ceil(len(x) / size)
    return [x[n * i:n * (i + 1)] for i in range(size - 1)] + [x[n * (size - 1):]]

def unsplit(x):
    return [item for sublist in x for item in sublist]

# Read input arguments
nbbodies = int(sys.argv[1])
NBSTEPS = int(sys.argv[2])
DISPLAY = len(sys.argv) != 4

# MPI initialization
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Initialize data on rank 0
if rank == 0:
    random.seed(0)
    data = init_world(nbbodies)
    data_split = split(data, size)
    start_time = time.time()
else:
    data = None
    data_split = None

# Scatter data to all processes
data_split = comm.scatter(data_split, root=0)

# Broadcast full dataset to all processes for force calculations
data = comm.bcast(data, root=0)

# Track interaction counts for unbalance calculation
local_interaction_count = 0

for t in range(NBSTEPS):
    force = [[0, 0] for _ in range(len(data_split))]
    
    # Calculate forces for local bodies
    for i in range(len(data_split)):
        for j in range(nbbodies):
            [fx, fy] = force[i]
            [in_x, in_y] = interaction(data_split[i], data[j])
            force[i] = [fx + in_x, fy + in_y]
            local_interaction_count += 1

    # Update positions and velocities for local bodies
    for i in range(len(data_split)):
        data_split[i] = update(data_split[i], force[i])

    # Gather updated positions across all processes
    data = unsplit(comm.allgather(data_split))

# Gather interaction counts for unbalance calculation
interaction_counts = comm.gather(local_interaction_count, root=0)

# Finalize on rank 0
if rank == 0:
    duration = time.time() - start_time
    total_interactions = sum(interaction_counts)
    max_interactions = max(interaction_counts)
    min_interactions = min(interaction_counts)
    unbalance = 100 * (max_interactions - min_interactions) / total_interactions if total_interactions > 0 else 0

    print(f"Duration: {duration}")
    print(f"Signature: {signature(data):.4e}")
    print(f"Unbalance: {unbalance:}")
