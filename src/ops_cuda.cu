#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>

// GPU側で並列実行されるカーネル関数 (要素ごとの足し算)
__global__ void matrix_add_kernel(
    const float *A, 
    const float *B,
    float *C,
    int size
) {
    // スレッドのインデックスを計算
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 配列のサイズ内であれば、一斉に足し算を実行
    if (idx < size) {
        C[idx] = A[idx] + B[idx];
    }
}

// CPU側から呼び出すためのブリッジ関数
extern "C" void matrix_add_gpu(
    const float* host_A,
    const float* host_B,
    float* host_C,
    int size
) {
    float *device_A = NULL;
    float *device_B= NULL;
    float *device_C = NULL;

    size_t bytes = size * sizeof(float);

    // GPU の VRAM 上にメモリを確保
    cudaMalloc((void **)&device_A, bytes);
    cudaMalloc((void **)&device_B, bytes);
    cudaMalloc((void **)&device_C, bytes);

    // CPU のデータを GPU のVRAMへ転送
    cudaMemcpy(device_A, host_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(device_B, host_B, bytes, cudaMemcpyHostToDevice);

    // GPUカーネルの起動 (1ブロック256スレッドとして計算員を配置)
    int threads_per_block = 256;
    int blocks_per_grid = (size + threads_per_block - 1) / threads_per_block;

    // <<< グリッド数、 ブロック数 >>> CUDA特有の構文でGPUに命令を送る
    matrix_add_kernel<<< blocks_per_grid, threads_per_block >>> (device_A, device_B, device_C, size);

    // 同期: GPUの計算が終わるのをCPU側で一瞬待つ
    cudaDeviceSynchronize();

    // 計算結果をGPUのVRAMからCPUのメモリへ返す
    cudaMemcpy(host_C, device_C, bytes, cudaMemcpyDeviceToHost);

    // VRAMを解放
    cudaFree(device_A);
    cudaFree(device_B);
    cudaFree(device_C);
}

__global__ void matrix_mul_kernal(
    const float *A, 
    const float *B,
    float *C,
    int A_rows,
    int A_cols,
    int B_cols
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < A_rows && col <  B_cols) {
        float sum = 0.0f;
        for (int k = 0; k < A_cols; k++) {
            sum += A[row * A_cols + k] * B[k * B_cols + col];
        }
        C[row * B_cols + col] = sum;
    }
}

// CPU側から呼び出すためのブリッジ関数
extern "C" void matrix_mul_gpu(
    const float* host_A,
    const float* host_B,
    float* host_C,
    int A_rows,
    int A_cols,
    int B_cols
) {
    float *device_A = NULL;
    float *device_B= NULL;
    float *device_C = NULL;

    size_t size_A = A_rows * A_cols * sizeof(float);
    size_t size_B = A_cols * B_cols * sizeof(float);
    size_t size_C = A_rows * B_cols * sizeof(float);

    // GPU の VRAM 上にメモリを確保
    cudaMalloc((void **)&device_A, size_A);
    cudaMalloc((void **)&device_B, size_B);
    cudaMalloc((void **)&device_C, size_C);

    // CPU のデータを GPU のVRAMへ転送
    cudaMemcpy(device_A, host_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(device_B, host_B, size_B, cudaMemcpyHostToDevice);

    // GPUカーネルの起動 (1ブロック 16x16 の二次元スレッドとして計算員を配置)
    dim3 threads_per_block(16, 16);
    dim3 blocks_per_grid((B_cols + threads_per_block.x - 1) / threads_per_block.x,
                    (A_rows + threads_per_block.y - 1) / threads_per_block.y);

    // <<< グリッド数、 ブロック数 >>> CUDA特有の構文でGPUに命令を送る
    matrix_mul_kernal<<< blocks_per_grid, threads_per_block >>> (device_A, device_B, device_C, A_rows, A_cols, B_cols);

    // 同期: GPUの計算が終わるのをCPU側で一瞬待つ
    cudaDeviceSynchronize();

    // 計算結果をGPUのVRAMからCPUのメモリへ返す
    cudaMemcpy(host_C, device_C, size_C, cudaMemcpyDeviceToHost);

    // VRAMを解放
    cudaFree(device_A);
    cudaFree(device_B);
    cudaFree(device_C);
}

__global__ void matrix_mul_a_trans_kernel(
    const float *A,
    const float *B,
    float *C,
    int A_rows,
    int A_cols,
    int B_cols
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < A_cols && col < B_cols) {
        float sum = 0.0f;
        for (int k = 0; k < A_rows; k++) {
            sum += A[k * A_cols + row] * B[k * B_cols + col];
        }
        C[row * B_cols + col] = sum;
    }
}

// CPU側から呼び出すためのブリッジ関数
extern "C" void matrix_mul_a_trans_gpu(
    const float* host_A,
    const float* host_B,
    float* host_C,
    int A_rows,
    int A_cols,
    int B_cols
) {
    float *device_A = NULL;
    float *device_B= NULL;
    float *device_C = NULL;

    size_t size_A = A_rows * A_cols * sizeof(float);
    size_t size_B = A_rows * B_cols * sizeof(float);
    size_t size_C = A_cols * B_cols * sizeof(float);

    // GPU の VRAM 上にメモリを確保
    cudaMalloc((void **)&device_A, size_A);
    cudaMalloc((void **)&device_B, size_B);
    cudaMalloc((void **)&device_C, size_C);

    // CPU のデータを GPU のVRAMへ転送
    cudaMemcpy(device_A, host_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(device_B, host_B, size_B, cudaMemcpyHostToDevice);

    // GPUカーネルの起動 (1ブロック 16x16 の二次元スレッドとして計算員を配置)
    dim3 threads_per_block(16, 16);
    dim3 blocks_per_grid((B_cols + threads_per_block.x - 1) / threads_per_block.x,
                    (A_rows + threads_per_block.y - 1) / threads_per_block.y);

    // <<< グリッド数、 ブロック数 >>> CUDA特有の構文でGPUに命令を送る
    matrix_mul_a_trans_kernel<<< blocks_per_grid, threads_per_block >>> (device_A, device_B, device_C, A_rows, A_cols, B_cols);

    // 同期: GPUの計算が終わるのをCPU側で一瞬待つ
    cudaDeviceSynchronize();

    // 計算結果をGPUのVRAMからCPUのメモリへ返す
    cudaMemcpy(host_C, device_C, size_C, cudaMemcpyDeviceToHost);

    // VRAMを解放
    cudaFree(device_A);
    cudaFree(device_B);
    cudaFree(device_C);
}

__global__ void matrix_mul_b_trans_kernel(
    const float *A,
    const float *B,
    float *C,
    int A_rows,
    int A_cols,
    int B_rows
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < A_cols && col < B_rows) {
        float sum = 0.0f;
        for (int k = 0; k < A_rows; k++) {
            sum += A[row * A_cols + k] * B[col * A_cols + k]; }
        C[row * B_rows + col] = sum;
    }
}

// CPU側から呼び出すためのブリッジ関数
extern "C" void matrix_mul_b_trans_gpu(
    const float* host_A,
    const float* host_B,
    float* host_C,
    int A_rows,
    int A_cols,
    int B_rows
) {
    float *device_A = NULL;
    float *device_B= NULL;
    float *device_C = NULL;

    size_t size_A = A_rows * A_cols * sizeof(float);
    size_t size_B = B_rows * A_cols * sizeof(float);
    size_t size_C = A_rows * B_rows * sizeof(float);

    // GPU の VRAM 上にメモリを確保
    cudaMalloc((void **)&device_A, size_A);
    cudaMalloc((void **)&device_B, size_B);
    cudaMalloc((void **)&device_C, size_C);

    // CPU のデータを GPU のVRAMへ転送
    cudaMemcpy(device_A, host_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(device_B, host_B, size_B, cudaMemcpyHostToDevice);

    // GPUカーネルの起動 (1ブロック 16x16 の二次元スレッドとして計算員を配置)
    dim3 threads_per_block(16, 16);
    dim3 blocks_per_grid((B_rows + threads_per_block.x - 1) / threads_per_block.x,
                    (A_rows + threads_per_block.y - 1) / threads_per_block.y);

    // <<< グリッド数、 ブロック数 >>> CUDA特有の構文でGPUに命令を送る
    matrix_mul_b_trans_kernel<<< blocks_per_grid, threads_per_block >>> (device_A, device_B, device_C, A_rows, A_cols, B_rows);

    // 同期: GPUの計算が終わるのをCPU側で一瞬待つ
    cudaDeviceSynchronize();

    // 計算結果をGPUのVRAMからCPUのメモリへ返す
    cudaMemcpy(host_C, device_C, size_C, cudaMemcpyDeviceToHost);

    // VRAMを解放
    cudaFree(device_A);
    cudaFree(device_B);
    cudaFree(device_C);
}