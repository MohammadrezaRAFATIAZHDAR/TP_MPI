#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>

#define ROOT 0

// Function to find the local maximum and its position
void find_local_max(int *local_array, int local_size, int *local_max, int *local_pos) {
    *local_max = INT_MIN;
    for (int i = 0; i < local_size; i++) {
        if (local_array[i] > *local_max) {
            *local_max = local_array[i];
            *local_pos = i;
        }
    }
}

int main(int argc, char **argv) {
    int rank, size, global_max, global_pos;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int *array = NULL;      // Full array (only on ROOT)
    int array_size = 12;    // Size of the full array
    int local_size = array_size / size; // Number of elements per process
    int remainder = array_size % size;

    // ROOT process initializes the array
    if (rank == ROOT) {
        array = (int *)malloc(array_size * sizeof(int));
        printf("Original array:\n");
        for (int i = 0; i < array_size; i++) {
            array[i] = rand() % 100; // Random numbers in range [0, 99]
            printf("%d ", array[i]);
        }
        printf("\n");
    }

    // Handle remainder by dynamically adjusting local sizes
    if (rank == size - 1) {
        local_size += remainder; // Last process handles extra elements
    }

    // Allocate memory for the local portion of the array
    int *local_array = (int *)malloc(local_size * sizeof(int));

    // Scatter data: dynamically handle irregular blocks
    int *sendcounts = NULL;
    int *displs = NULL;
    if (rank == ROOT) {
        sendcounts = (int *)malloc(size * sizeof(int));
        displs = (int *)malloc(size * sizeof(int));

        int offset = 0;
        for (int i = 0; i < size; i++) {
            sendcounts[i] = (i == size - 1) ? local_size : array_size / size;
            displs[i] = offset;
            offset += sendcounts[i];
        }
    }

    MPI_Scatterv(array, sendcounts, displs, MPI_INT, local_array, local_size, MPI_INT, ROOT, MPI_COMM_WORLD);

    // Find local maximum and its position
    int local_max, local_pos;
    find_local_max(local_array, local_size, &local_max, &local_pos);

    // Adjust position relative to global array
    int global_offset;
    MPI_Scan(&local_size, &global_offset, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    global_offset -= local_size; // Start of this process's data in the global array
    local_pos += global_offset;

    // Reduce to find global maximum and its position
    int global_max_pos;
    struct {
        int value;
        int rank;
    } local_result = {local_max, local_pos}, global_result;

    MPI_Reduce(&local_result, &global_result, 1, MPI_2INT, MPI_MAXLOC, ROOT, MPI_COMM_WORLD);

    // ROOT process outputs the result
    if (rank == ROOT) {
        printf("Global maximum value: %d at position: %d\n", global_result.value, global_result.rank);
    }

    // Clean up
    free(local_array);
    if (rank == ROOT) {
        free(array);
        free(sendcounts);
        free(displs);
    }

    MPI_Finalize();
    return 0;
}
