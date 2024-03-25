#include <iostream>
#include <vector>
#include <iomanip>
#include <cstdlib>
#include <chrono>
#include <mpi.h>
#include <stdio.h>
#include <omp.h>

using namespace std;
using namespace std::chrono;

// Needs to be compiled with:
// mpicxx freem_MMM_hybrid.cpp -fopenmp -o MMM_hybrid.exe

// Needs to be run with:
// export OMP_NUM_THREADS=k
// mpiexec ./MMM_hybrid N

// Populating Matrices
void populateMatrix(vector<vector<double>>& matrix) {
    for(auto &row : matrix) {
        for(auto &elem : row) {
            elem = static_cast <double> (rand()) / RAND_MAX;
        }
    }
}

// Multiplying matrices
vector<vector<double>> multiplyMatrices(const vector<vector<double>>& matrix1, const vector<vector<double>>& matrix2, int startRow, int endRow) {
    int N = matrix1[0].size();
    vector<vector<double>> result(endRow - startRow, vector<double>(N, 0));

    // Multiplying matrices, but only from the first to last row each processor "owns" and can see
    #pragma omp parallel
    for(int i = startRow; i < endRow; ++i) {
        for(int j = 0; j < N; ++j) {
            for(int k = 0; k < N; ++k) {
                result[i - startRow][j] += matrix1[i][k] * matrix2[k][j];
            }
        }
    }
    return result;
}

int main(int argc, char* argv[]) {
    // Initializing MPI
    MPI_Init(&argc, &argv);
    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Initializing output column headers
    if(rank == 0) {
        cout << "#  N=  , time (s)" << endl;
    }

    // Running this 100 times so we can take an average later
    for(int ii=0; ii < 100; ii++) {
        // Grabbing N from user's command call, and initializing three matrices used for calculation
        int N = atoi(argv[1]);
        vector<vector<double>> matrix1(N, vector<double>(N));
        vector<vector<double>> matrix2(N, vector<double>(N));
        vector<vector<double>> result(N, vector<double>(N));

        // Populating the matrices only on the first rank
        if(rank == 0) {
            populateMatrix(matrix1);
            populateMatrix(matrix2);
        }

        // Broadcasting the matrix to all the other processors
        MPI_Bcast(&matrix2[0][0], N*N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        // Each processor will calculate this differently
        int rows_per_process = N / size;
        int startRow = rank * rows_per_process;
        int endRow = (rank == size - 1) ? N : startRow + rows_per_process;

        // Starting clock for this specific multiplication (doing on all processors to prevent weird instances)
        auto start = high_resolution_clock::now();
        

        // Actual matrix multiplication
        vector<vector<double>> local_result = multiplyMatrices(matrix1, matrix2, startRow, endRow);

        // Gathering results on 0th processor
        for(int i = 0; i < local_result.size(); i++) {
            MPI_Gather(&local_result[i][0], N, MPI_DOUBLE, &result[startRow + i][0], N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        }

        // Outputting timing data
        if(rank == 0) {
            auto stop = high_resolution_clock::now();
            auto duration = duration_cast<microseconds>(stop - start);
            cout << setw(5) << N << ", " << duration.count() / 1.0e6 << endl;
        }
    }

    MPI_Finalize();
    return 0;
}
