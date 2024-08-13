#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <cmath> // For fabs and std::numeric_limits
#include <algorithm> // For std::min

// Function to generate a random matrix of size rows x cols with values between p and q
std::vector<std::vector<double>> generateRandomMatrix(int rows, int cols, double p, double q) {
    std::vector<std::vector<double>> matrix(rows, std::vector<double>(cols));
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            matrix[i][j] = p + static_cast<double>(rand()) / RAND_MAX * (q - p);
        }
    }
    return matrix;
}

// Standard matrix multiplication: A (m x k) * B (k x n) = C (m x n)
void multiplyMatrices(
        const std::vector<std::vector<double>> &A,
        const std::vector<std::vector<double>> &B,
        std::vector<std::vector<double>> &C,
        int m, int k, int n) {

    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            for (int x = 0; x < k; ++x) {
                C[i][j] += A[i][x] * B[x][j];
            }
        }
    }
}

// Optimized block matrix multiplication: A (m x k) * B (k x n) = C (m x n) with block size
void multiplyMatricesBlock(
        const std::vector<std::vector<double>> &A,
        const std::vector<std::vector<double>> &B,
        std::vector<std::vector<double>> &C,
        int m, int k, int n, int blockSize) {

    for (int i = 0; i < m; i += blockSize) {
        for (int x = 0; x < k; x += blockSize) {
            for (int j = 0; j < n; j += blockSize) {
                // Multiply the inner blocks
                // C[i][j] += A[i][x] * B[x][j]
                for (int ii = i; ii < std::min(i + blockSize, m); ++ii) {
                    for (int xx = x; xx < std::min(x + blockSize, k); ++xx) {
                        double a = A[ii][xx];
                        for (int jj = j; jj < std::min(j + blockSize, n); ++jj) {
                            C[ii][jj] += a * B[xx][jj];
                        }
                    }
                }
            }
        }
    }
}

// Function to verify that the matrix multiplication result C is correct within machine epsilon
bool verifyMatrixMultiplication(
        const std::vector<std::vector<double>> &A,
        const std::vector<std::vector<double>> &B,
        const std::vector<std::vector<double>> &C,
        int m, int k, int n) {

    double epsilon = std::numeric_limits<double>::epsilon();
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            double sum = 0.0;
            for (int x = 0; x < k; ++x) {
                sum += A[i][x] * B[x][j];
            }
            if (std::fabs(sum - C[i][j]) > epsilon) {
                std::cerr << "Matrix multiplication verification failed at (" << i << ", " << j << ")!\n";
                return false;
            }
        }
    }
    return true;
}

int main() {
    srand(static_cast<unsigned int>(time(0)));

    int start = 256;
    int end = 2048;
    int step = 256;
    int blockSize = 64;
    double p = -1.0, q = 1.0;

    for (int size = start; size <= end; size += step) {
        int m = size, k = size, n = size;

        // Generate random matrices A (m x k) and B (k x n)
        auto A = generateRandomMatrix(m, k, p, q);
        auto B = generateRandomMatrix(k, n, p, q);

        // Measure the time taken by standard matrix multiplication
        std::vector<std::vector<double>> C1(m, std::vector<double>(n, 0.0));
        auto start1 = std::chrono::high_resolution_clock::now();

        multiplyMatrices(A, B, C1, m, k, n);

        auto end1 = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double, std::milli> duration1 = end1 - start1;

        // Measure the time taken by block matrix multiplication
        std::vector<std::vector<double>> C2(m, std::vector<double>(n, 0.0));
        auto start2 = std::chrono::high_resolution_clock::now();

        multiplyMatricesBlock(A, B, C2, m, k, n, blockSize);

        auto end2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration2 = end2 - start2;

        // Verify the result
        if (!verifyMatrixMultiplication(A, B, C1, m, k, n)) {
            std::cerr << "Verification failed for standard multiplication at size " << size << "\n";
        }
        if (!verifyMatrixMultiplication(A, B, C2, m, k, n)) {
            std::cerr << "Verification failed for block multiplication at size " << size << "\n";
        }

        // Print out the results
        std::cout << size << ", " << duration1.count() << ", " << duration2.count() << "\n";
    }

    return 0;
}
