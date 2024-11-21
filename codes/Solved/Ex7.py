from mpi4py import MPI
import random
import time

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Total number of points to generate (constant across runs)
nb = 10000001  # Total random draws (constant)
local_nb = nb // size  # Points per process

# Ensure each process has a unique random seed
random.seed(rank + int(time.time()))

# Start the timer
comm.Barrier()  # Synchronize processes before starting the timer
start_time = MPI.Wtime()

# Count points inside the circle
inside = 0
for _ in range(local_nb):
    x = random.random()
    y = random.random()
    if x * x + y * y <= 1:
        inside += 1

# Print the value of "inside" for each process
print(f"Process {rank}: inside = {inside}")

# Reduce the counts to the root process
total_inside = comm.reduce(inside, op=MPI.SUM, root=0)

# End the timer
comm.Barrier()  # Synchronize processes before stopping the timer
end_time = MPI.Wtime()

# Compute and print π and execution time on process 0
if rank == 0:
    pi = 4 * total_inside / nb
    print(f"Pi = {pi}")
    print(f"Execution time with {size} processes: {end_time - start_time:.4f} seconds")
