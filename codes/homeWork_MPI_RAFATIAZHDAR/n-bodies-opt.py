from mpi4py import MPI
import sys
import math
import time
import numpy as np
from n_bodies import init_world, interaction, update, signature

# Helper functions to split and merge data
def split_pairs(n, size, rank):
    """
    Split body pairs (i, j) across processes.
    Each process gets a range of indices for (i, j) pairs.
    """
    total_pairs = n * (n - 1) // 2  # Total number of unique pairs (i, j) where i < j
    pairs_per_process = total_pairs // size
    remainder = total_pairs % size

    # Compute start and end indices for the pairs
    start = rank * pairs_per_process + min(rank, remainder)
    end = start + pairs_per_process + (1 if rank < remainder else 0)

    # Generate the assigned pairs (i, j) for this process
    pairs = []
    index = 0
    for i in range(n):
        for j in range(i + 1, n):
            if start <= index < end:
                pairs.append((i, j))
            index += 1

    return pairs

# Main simulation function
def n_bodies_opt(nbbodies, NBSTEPS):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Initialize data on rank 0
    if rank == 0:
        data = init_world(nbbodies)
        start_time = time.time()
    else:
        data = None

    # Broadcast the initial data to all processes
    data = comm.bcast(data, root=0)

    # Get the pairs assigned to this process
    local_pairs = split_pairs(nbbodies, size, rank)

    # Track the number of interactions for unbalance calculation
    local_interaction_count = len(local_pairs)

    # Force array for each body
    local_force = np.zeros((nbbodies, 2))

    # Simulation loop
    for t in range(NBSTEPS):
        # Reset local forces
        local_force.fill(0)

        # Compute forces for the assigned pairs
        for i, j in local_pairs:
            f = interaction(data[i], data[j])
            local_force[i] += f
            local_force[j] -= f

        # Combine forces across all processes
        global_force = np.zeros_like(local_force)
        comm.Allreduce(local_force, global_force, op=MPI.SUM)

        # Update positions and velocities locally
        for i in range(nbbodies):
            data[i] = update(data[i], global_force[i])

    # Gather interaction counts on rank 0
    interaction_counts = comm.gather(local_interaction_count, root=0)

    # Finalize and output results on rank 0
    if rank == 0:
        duration = time.time() - start_time
        total_interactions = sum(interaction_counts)
        max_interactions = max(interaction_counts)
        min_interactions = min(interaction_counts)
        unbalance = 100 * (max_interactions - min_interactions) / total_interactions if total_interactions > 0 else 0

        print(f"Duration: {duration}")
        print(f"Signature: {signature(data):.4e}")
        print(f"Unbalance: {unbalance:.2f}")

# Entry point
if __name__ == "__main__":
    nbbodies = int(sys.argv[1])  # Number of bodies
    NBSTEPS = int(sys.argv[2])  # Number of steps
    n_bodies_opt(nbbodies, NBSTEPS)
