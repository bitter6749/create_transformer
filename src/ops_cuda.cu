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