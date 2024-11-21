#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define ROOT 0

int main(int argc, char** argv) {
    int rank, size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int* data = NULL;

    // Root process initializes the data
    if (rank == ROOT) {
        data = (int*)malloc(size * sizeof(int)); // Allocate array for the number of processes
        for (int i = 0; i < size; i++) {
            data[i] = 2 * i + 1; // Example initialization: odd numbers
        }
    }

    int recv_data;

    // Scatter the data from the root process
    MPI_Scatter(data, 1, MPI_INT, &recv_data, 1, MPI_INT, ROOT, MPI_COMM_WORLD);

    // Print the data received by each process
    printf("Process %d received data %d\n", rank, recv_data);

    // Free allocated memory on the root process
    if (rank == ROOT) {
        free(data);
    }

    MPI_Finalize();
    return 0;
}
