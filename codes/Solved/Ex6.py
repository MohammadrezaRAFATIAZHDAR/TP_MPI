from mpi4py import MPI
import numpy as np

# Function to find the local maximum and its position in an array
def get_max(tab):
    pos = np.argmax(tab)
    return tab[pos], pos

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Array size and initialization
SIZE = 12  # Size of the array
chunk_size = SIZE // size  # Divide array evenly among processes

# Process 0 initializes the array
if rank == 0:
    np.random.seed(42)
    tab = np.random.randint(100, size=SIZE, dtype='i')
    print(f"Original array: {tab}")
else:
    tab = None

# Each process receives its chunk of the array
local_chunk = np.empty(chunk_size, dtype='i')
comm.Scatter(tab, local_chunk, root=0)

# Find local maximum and its position
local_max, local_pos = get_max(local_chunk)

# Gather all local maxima and their positions at root
global_max_data = comm.gather((local_max, local_pos + rank * chunk_size), root=0)

# Process 0 determines the global maximum and its position
if rank == 0:
    global_max, global_pos = max(global_max_data, key=lambda x: x[0])
    print(f"Global maximum value: {global_max}, at position: {global_pos}")
