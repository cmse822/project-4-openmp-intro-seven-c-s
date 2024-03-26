#include <iostream>
#include <vector>
#include <iomanip>
#include <cstdlib>
#include <chrono>
#include <stdio.h>
#include <omp.h>

using namespace std;
using namespace std::chrono;

// This is a code that Ian worked on - view data in Freem_Work folder

// Function to populate a matrix with random values
void populateMatrix(vector<vector<double>>& matrix) {
    for(auto &row : matrix) {
        for(auto &elem : row) {
            elem = static_cast <double> (rand()) / RAND_MAX;
        }
    }
}

// Function for serial matrix multiplication
vector<vector<double>> multiplyMatricesSerial(const vector<vector<double>>& matrix1, const vector<vector<double>>& matrix2) {
    int N = matrix1.size();
    vector<vector<double>> result(N, vector<double>(N, 0));

    for(int i = 0; i < N; ++i) {
        for(int j = 0; j < N; ++j) {
            for(int k = 0; k < N; ++k) {
                result[i][j] += matrix1[i][k] * matrix2[k][j];
            }
        }
    }

    return result;
}

// Function for Parallel matrix multiplication
vector<vector<double>> multiplyMatrices(const vector<vector<double>>& matrix1, const vector<vector<double>>& matrix2) {
    int N = matrix1.size();
    vector<vector<double>> result(N, vector<double>(N, 0));

    #pragma omp parallel for
    for(int i = 0; i < N; ++i) {
        for(int j = 0; j < N; ++j) {
            for(int k = 0; k < N; ++k) {
                result[i][j] += matrix1[i][k] * matrix2[k][j];
            }
        }
    }

    return result;
}

int main(int argc, char* argv[]) {
    // Error checking for debug because I'm mistake-prone
    if(argc < 2) {
        cout << "Usage: " << argv[0] << " <N>" << endl;
        return 1;
    }

    // Storing N as an int
    int N = atoi(argv[1]);
    if (N <= 0) {
        cout << "Matrix size must be a positive integer." << endl;
        return 1;
    }

    vector<vector<double>> matrix1(N, vector<double>(N));
    vector<vector<double>> matrix2(N, vector<double>(N));

    // Priming output columns
    cout << "#  N=  , time (s)" << endl;

    for(int i=1; i<=100; i++){
        // Populate matrices with random values
        populateMatrix(matrix1);
        populateMatrix(matrix2);

        // Start the timer
        auto start = high_resolution_clock::now();

        // Perform matrix multiplication
        vector<vector<double>> result = multiplyMatrices(matrix1, matrix2);
        
        // Stop the timer
        auto stop = high_resolution_clock::now();

        // Calculate duration
        auto duration = duration_cast<microseconds>(stop - start);

        // Verify results by generating a test matrix calculated in serial
        // vector<vector<double>> test = multiplyMatricesSerial(matrix1, matrix2);
        auto test = result;      // Use this line to bypass the test conditions

        // Testing 3 corner entries of the multiplied matrix to ensure equality, and the operation was completed properly
        double precision = 1.0e-7;

        if(test[1][1] - result[1][1] < precision && test[N-1][1] - result[N-1][1] < precision && test[N-1][N-1] - result[N-1][N-1] < precision) {
            cout << setw(5) << N << ", " << duration.count() / 1.0e6 << endl;
        } else { 
            cout << "Multiplication was NOT successful, breaking" << endl;
            return 2;
        }
    }
    return 0;
}