#include <iostream>
#include <mpi.h>
#include <omp.h>
#include <vector>
#include <stdio.h>
#include <stdlib.h>

// Jingyao Code

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int N = 100; // change this for the matrix size
    int rows_per_rank = N / size;

    double A[N][N], B[N][N];

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i][j] = 1;
            B[i][j] = 1;
        }
    }

    double start_time = omp_get_wtime();
  
    // Distribute rows of A through MPI ranks
    std::vector<double> local_A(rows_per_rank * N);
    MPI_Scatter(A, rows_per_rank * N, MPI_DOUBLE, local_A.data(), rows_per_rank * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // matrix multiplication using OpenMP
    std::vector<double> local_C(rows_per_rank * N, 0);
    #pragma omp parallel for
    for (int i = 0; i < rows_per_rank; ++i) {
        for (int j = 0; j < N; ++j) {
            for (int k = 0; k < N; ++k) {
                local_C[i * N + j] += local_A[i * N + k] * B[k][j];
            }
        }
    }

    // put results on rank 0
    double C[N][N];
    MPI_Gather(local_C.data(), rows_per_rank * N, MPI_DOUBLE, C, rows_per_rank * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    double end_time = omp_get_wtime();

    // output on rank 0
    if (rank == 0) {
        std::cout << "Result matrix C:" << std::endl;
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                std::cout << C[i][j] << " ";
            }
            std::cout << std::endl;
        }
    }

    printf("Time taken: %f seconds\n", end_time - start_time);


    MPI_Finalize();
    return 0;
}
