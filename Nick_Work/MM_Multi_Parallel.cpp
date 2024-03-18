#include <iostream>
#include <cstdlib>
#include <ctime>
#include <sys/time.h>
#include <chrono>
#include <fstream>
#include <vector>
#include <omp.h>
// Part 1 - Multiplying Two Matrices
// Ian's Processor information:
/*
11th Gen Intel i7 (2.80 GHz)

Sockets:    1
Cores:      4
L1 cache:   320  kB
L2 cache:   5.0  MB
L3 cache:   12.0 MB
*/

using namespace std;

// function that returns a vector of vectors (matrix)
vector<vector<double>> generate_matrix(int m) {
    // Initialize M, to be a vector of vectors with entries of 0.0
    /*
    vector<vector<double>> M is creating the empty vector of vectors called M
    M(m, vector<double>(m, 0.0)) initializes the parent vector with entries of vectors
    vector<double>(m,0.0) initializes those child vectors with 0.0
    */
    vector<vector<double>> M(m, vector<double>(m, 0.0));
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < m; ++j) {
            M[i][j] = static_cast <double> (rand()) / RAND_MAX;
        }
    }
    return M;
}


int main(int argc, char *argv[]) {
    srand(0);
    int size = atoi(argv[1]);
    // make the dimensions of each matrix: A, B, C
    auto matrixA = generate_matrix(size);
    auto matrixB = generate_matrix(size);    
    vector<vector<double>> C(size, vector<double>(size, 0.0));;
    // Starting high_resolution_clock before multiplication
    // Implementation from https://www.geeksforgeeks.org/measure-execution-time-function-cpp/
    
    // multiply A and B to get matrix C
    auto start = chrono::high_resolution_clock::now();
    #pragma omp parallel for collapse(3)
    for (int j = 0; j < size; ++j) {
        //omp_set_num_threads(4);
        for (int i = 0; i < size; ++i) {
            for (int k = 0; k < size; ++k) 
            {
                C[i][j] += matrixA[i][k] * matrixB[k][j];
            }
        }
    }
    
    // Stopping timer
    auto stop = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::microseconds>(stop - start);

    cout << "Time taken by matrix of size " << size << " = " 
                << duration.count() << " us. Thread:" << omp_get_thread_num() << endl;

    return 0;
}
