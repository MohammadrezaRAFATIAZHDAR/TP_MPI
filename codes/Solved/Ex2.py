from mpi4py import MPI
import numpy as np

# Initialize MPI environment
comm = MPI.COMM_WORLD
rank = comm.Get_rank()  # Get the rank of the process
size = comm.Get_size()  # Get the total number of processes

# Define the number
if rank == 0:
    passnumber = np.array([42], dtype='i')  # Use a NumPy array for broadcasting
else:
    passnumber = np.array([0], dtype='i')  # Initialize to 0 on other processes

# Print the number before broadcasting
print(f"At start in process {rank}, the passnumber is {passnumber[0]}")

# Broadcast the number
comm.Bcast(passnumber, root=0)

# Print the number after broadcasting
print(f"After collective in process {rank}, the passnumber is {passnumber[0]}")
