from mpi4py import MPI
import numpy as np
import sys

def nb_primes(n):
    """Compute the number of divisors of n."""
    result = 0
    for i in range(1, n + 1):
        if n % i == 0:
            result += 1
    return result

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Get the upper bound from the command-line arguments
if len(sys.argv) < 2:
    if rank == 0:
        print("Usage: python3 prime_parallel_blocks.py <upper_bound>")
    sys.exit(1)

upper_bound = int(sys.argv[1])

# Start timing
comm.Barrier()  # Synchronize before starting the timer
start_time = MPI.Wtime()

# Determine the block size
block_size = upper_bound // size
remainder = upper_bound % size

# Prepare the blocks (on rank 0 only)
if rank == 0:
    numbers = np.arange(1, upper_bound + 1, dtype='i')
    blocks = np.array_split(numbers, size)
else:
    blocks = None

# Scatter the blocks to all processes
local_block = comm.scatter(blocks, root=0)

# Compute the local maximum
local_max = 0
for val in local_block:
    tmp = nb_primes(val)
    local_max = max(local_max, tmp)

# Reduce to find the global maximum
global_max = comm.reduce(local_max, op=MPI.MAX, root=0)

# End timing
comm.Barrier()
end_time = MPI.Wtime()

# Output results on rank 0
if rank == 0:
    print(f"Global maximum number of divisors: {global_max}")
    print(f"Total execution time: {end_time - start_time:.4f} seconds")
