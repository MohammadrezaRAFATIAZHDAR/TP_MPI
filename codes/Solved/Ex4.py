from mpi4py import MPI
import sys

# Initialize MPI environment
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Function to compute sum in a range
def cumul(a, b):
    return sum(range(a, b))

# Get the number N from command-line arguments
nb = int(sys.argv[1])

# Divide the work among processes
local_start = rank * (nb // size)
local_end = (rank + 1) * (nb // size)

# Each process computes its partial sum
local_sum = cumul(local_start, local_end)

# Use MPI.Reduce to gather the sums and compute the total sum at root
total_sum = comm.reduce(local_sum, op=MPI.SUM, root=0)

# Print the result on the root process
if rank == 0:
    print(f"The sum of the first {nb} integers is: {total_sum}")
