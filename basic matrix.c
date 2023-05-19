#include <stdio.h>

#define M 4
#define N 4
#define K 4

__global__ void matrixMultiplication(int* A, int* B, int* C)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N)
    {
        int sum = 0;
        for (int i = 0; i < K; i++)
        {
            sum += A[row * K + i] * B[i * N + col];
        }
        C[row * N + col] = sum;
    }
}

int main()
{
    int A[M][K] = {{1, 2, 3, 4},
                   {5, 6, 7, 8},
                   {9, 10, 11, 12},
                   {13, 14, 15, 16}};

    int B[K][N] = {{1, 2, 3, 4},
                   {5, 6, 7, 8},
                   {9, 10, 11, 12},
                   {13, 14, 15, 16}};

    int C[M][N] = {0};

    int* d_A;
    int* d_B;
    int* d_C;

    cudaMalloc((void**)&d_A, sizeof(int) * M * K);
    cudaMalloc((void**)&d_B, sizeof(int) * K * N);
    cudaMalloc((void**)&d_C, sizeof(int) * M * N);

    cudaMemcpy(d_A, A, sizeof(int) * M * K, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, sizeof(int) * K * N, cudaMemcpyHostToDevice);

    dim3 grid(1, 1);
    dim3 block(N, M);

    matrixMultiplication<<<grid, block>>>(d_A, d_B, d_C);

    cudaMemcpy(C, d_C, sizeof(int) * M * N, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    printf("Result:\n");
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            printf("%d ", C[i][j]);
        }
        printf("\n");
    }

    return 0;
}