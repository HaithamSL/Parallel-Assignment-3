void matrixMultiplicationBasic(int* A, int* B, int* C, int M, int N, int K)
{
    #pragma acc parallel loop collapse(2) present(A, B, C)
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            int sum = 0;
            #pragma acc loop reduction(+:sum)
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

void matrixMultiplicationTiled(int* A, int* B, int* C, int M, int N, int K, int TILE_SIZE)
{
    #pragma acc parallel present(A, B, C)
    {
        #pragma acc loop collapse(2)
        for (int i = 0; i < M; i += TILE_SIZE) {
            for (int j = 0; j < N; j += TILE_SIZE) {
                for (int k = 0; k < K; k += TILE_SIZE) {
                    #pragma acc loop collapse(2)
                    for (int ii = i; ii < min(i + TILE_SIZE, M); ii++) {
                        for (int jj = j; jj < min(j + TILE_SIZE, N); jj++) {
                            int sum = 0;
                            #pragma acc loop reduction(+:sum)
                            for (int kk = k; kk < min(k + TILE_SIZE, K); kk++) {
                                sum += A[ii * K + kk] * B[kk * N + jj];
                            }
                            C[ii * N + jj] += sum;
                        }
                    }
                }
            }
        }
    }
}
