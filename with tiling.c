#include <stdio.h>

#define M 4
#define N 4
#define K 4

#define TILE_SIZE 2

__global__ void matrixMultiplication(int* A, int* B, int* C)
{
    __shared__ int ds_A[TILE_SIZE][TILE_SIZE];
    __shared__ int ds_B[TILE_SIZE][TILE_SIZE];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * blockDim.y + ty;
    int col = bx * blockDim.x + tx;

    int sum = 0;

    for (int i = 0; i < (K-1) / TILE_SIZE + 1; i++)
    {
        if (row < M && i * TILE_SIZE + tx < K)
            ds_A[ty][tx] = A[row * K + i * TILE_SIZE + tx];
        else
            ds_A[ty][tx] = 0;

        if (col < N && i * TILE_SIZE + ty < K)
            ds_B[ty][tx] = B[(i * TILE_SIZE + ty) * N + col];
        else
            ds_B[ty][tx] = 0;

        __syncthreads();

        for (int j = 0; j < TILE_SIZE; j++)
        {
            sum += ds_A[ty][j] * ds_B[j][tx];
        }

        __syncthreads();
    }

    if (row < M && col < N)
    {
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

    dim3 grid((N-1) / TILE_SIZE + 1, (M-1) / TILE_SIZE + 1);
    dim3 block(TILE_SIZE, TILE_SIZE);

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