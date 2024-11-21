from mpi4py import MPI
import numpy as np

# Initialize MPI environment
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Initialize the random generator (comment out this line in the second run)
#np.random.seed(0)

# Generate 10 random integers between 0 and 99
random_array = np.random.randint(0, 100, 10)

# Find the local min and max
local_min = np.min(random_array)
local_max = np.max(random_array)

# Gather all local min and max values at rank 0
global_min = comm.allreduce(local_min, op=MPI.MIN)
global_max = comm.allreduce(local_max, op=MPI.MAX)

# Check if all arrays are identical
arrays_identical = (local_min == global_min) and (local_max == global_max)

# Print result
if rank == 0:
    print(f"Random arrays are {'identical : True' if arrays_identical else 'different : False'}.")

# Print local results for debugging (optional)
print(f"Process {rank}: Array = {random_array}, Min = {local_min}, Max = {local_max}")
