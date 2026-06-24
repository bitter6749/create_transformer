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
    const float* device_A,
    const float* device_B,
    float* device_C,
    int size
) {
    // GPUカーネルの起動 (1ブロック256スレッドとして計算員を配置)
    int threads_per_block = 256;
    int blocks_per_grid = (size + threads_per_block - 1) / threads_per_block;

    // <<< グリッド数、 ブロック数 >>> CUDA特有の構文でGPUに命令を送る
    matrix_add_kernel<<< blocks_per_grid, threads_per_block >>> (device_A, device_B, device_C, size);
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
    const float* device_A,
    const float* device_B,
    float* device_C,
    int A_rows,
    int A_cols,
    int B_cols
) {
    // GPUカーネルの起動 (1ブロック 16x16 の二次元スレッドとして計算員を配置)
    dim3 threads_per_block(16, 16);
    dim3 blocks_per_grid((B_cols + threads_per_block.x - 1) / threads_per_block.x,
                    (A_rows + threads_per_block.y - 1) / threads_per_block.y);

    // <<< グリッド数、 ブロック数 >>> CUDA特有の構文でGPUに命令を送る
    matrix_mul_kernal<<< blocks_per_grid, threads_per_block >>> (device_A, device_B, device_C, A_rows, A_cols, B_cols);
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
    const float* device_A,
    const float* device_B,
    float* device_C,
    int A_rows,
    int A_cols,
    int B_cols
) {
    // GPUカーネルの起動 (1ブロック 16x16 の二次元スレッドとして計算員を配置)
    dim3 threads_per_block(16, 16);
    dim3 blocks_per_grid((B_cols + threads_per_block.x - 1) / threads_per_block.x,
                    (A_rows + threads_per_block.y - 1) / threads_per_block.y);

    // <<< グリッド数、 ブロック数 >>> CUDA特有の構文でGPUに命令を送る
    matrix_mul_a_trans_kernel<<< blocks_per_grid, threads_per_block >>> (device_A, device_B, device_C, A_rows, A_cols, B_cols);
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

    if (row < A_rows && col < B_rows) {
        float sum = 0.0f;
        for (int k = 0; k < A_cols; k++) {
            sum += A[row * A_cols + k] * B[col * A_cols + k]; }
        C[row * B_rows + col] = sum;
    }
}

// CPU側から呼び出すためのブリッジ関数
extern "C" void matrix_mul_b_trans_gpu(
    const float* device_A,
    const float* device_B,
    float* device_C,
    int A_rows,
    int A_cols,
    int B_rows
) {    
    // GPUカーネルの起動 (1ブロック 16x16 の二次元スレッドとして計算員を配置)
    dim3 threads_per_block(16, 16);
    dim3 blocks_per_grid((B_rows + threads_per_block.x - 1) / threads_per_block.x,
                    (A_rows + threads_per_block.y - 1) / threads_per_block.y);

    // <<< グリッド数、 ブロック数 >>> CUDA特有の構文でGPUに命令を送る
    matrix_mul_b_trans_kernel<<< blocks_per_grid, threads_per_block >>> (device_A, device_B, device_C, A_rows, A_cols, B_rows);
}

// 勾配の平均化 & クリッピングをGPU側で行うカーネル
__global__ void matrix_avg_clip_kernel(
    float *acc_dW, 
    float inv_b,
    float clip_val,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // 平均化
        float val = acc_dW[idx] * inv_b;

        // 勾配クリッピング (-clip_val から +clip_val の間に丸める)
        if (val > clip_val) val = clip_val;
        if (val < -clip_val) val = -clip_val;

        acc_dW[idx] = val;
    }
}

extern "C" void matrix_avg_clip_gpu(
    float *d_acc_dW, 
    float inv_b,
    float clip_val,
    int size
) {
    int threads_per_block = 256;
    int blocks_per_grid = (size + threads_per_block - 1) / threads_per_block;
    matrix_avg_clip_kernel<<< blocks_per_grid, threads_per_block >>> (d_acc_dW, inv_b, clip_val, size);
}

// GPU側で一斉に重みを更新するカーネル
__global__ void sgd_update_kernel(
    float *W,
    const float *dW, 
    float lr,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        W[idx] -= lr * dW[idx];
    }
}

extern "C" void sgd_update_gpu(
    float *device_W,
    const float *device_dW,
    float lr,
    int size
) {
    int threads_per_block = 256;
    int blocks_per_grid = (size + threads_per_block - 1) / threads_per_block;
    sgd_update_kernel<<< blocks_per_grid, threads_per_block>>>(device_W, device_dW, lr, size);
}

// 行列を 0.0f でリセットするカーネル
__global__ void matrix_clear_kernel(float *data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = 0.0f;
    }
}

extern "C" void matrix_clear_gpu(float *d_data, int size) {
    int threads_per_block = 256;
    int blocks_per_grid = (size + threads_per_block - 1) / threads_per_block;
    matrix_clear_kernel<<< blocks_per_grid, threads_per_block >>>(d_data, size);
}