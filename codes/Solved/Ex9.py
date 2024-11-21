from mpi4py import MPI
import numpy as np

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Constants
n = 30  # Power of the matrix
DIM = 12  # Dimension of the square matrix

# Initialize matrix A on rank 0
if rank == 0:
    np.random.seed(10)
    A = np.random.rand(DIM, DIM)
    print("Matrix A:\n", A)
    # Reference result for verification
    ref_res = np.linalg.matrix_power(A, n)
    print("Reference trace:", ref_res.trace())
else:
    A = None

# Broadcast the matrix A to all processes
A = comm.bcast(A, root=0)

# Determine the number of rows each process will handle
rows_per_process = DIM // size
extra_rows = DIM % size  # Handle cases where DIM is not divisible by size

# Determine the range of rows for each process
start_row = rank * rows_per_process + min(rank, extra_rows)
end_row = start_row + rows_per_process + (1 if rank < extra_rows else 0)
local_rows = end_row - start_row

# Extract local portion of the matrix
local_A = A[start_row:end_row, :]

# Initialize result matrix
local_res = local_A.copy()

# Perform matrix power computation
for _ in range(n - 1):
    tmp = np.zeros_like(local_res)
    # Perform local computation
    for i in range(local_rows):
        for j in range(DIM):
            tmp[i, j] = np.dot(local_A[i, :], A[:, j])
    local_res = tmp

# Gather the results back to the root process
res = None
if rank == 0:
    res = np.zeros((DIM, DIM), dtype=np.float64)

comm.Gather(local_res, res, root=0)

# Print the trace of the resulting matrix on rank 0
if rank == 0:
    print("Computed trace:", res.trace())
