from mpi4py import MPI
import numpy as np

# Initialize MPI environment
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Only process 0 generates the team assignments
if rank == 0:
    teams = np.random.randint(2, size=size, dtype='i')  # Generate random 0/1 team assignments
    print(f"The file contains {teams}")
else:
    teams = np.empty(size, dtype='i')  # Placeholder for receiving the array

# Broadcast the teams array to all processes
comm.Bcast(teams, root=0)

# Determine the team for the current process
my_team = teams[rank]

# Map team numbers to colors
colors = {0: 'blue', 1: 'green'}
print(f"I am {rank} and my team is {colors[my_team]}")
