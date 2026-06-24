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

__global__ void bias_add_kernel(float *m, const float *bias, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols) {
        // 各行の col 番目の要素に、バイアスの col 番目の値を足す
        m[row * cols + col] += bias[col];
    }
}

extern "C" void bias_add_gpu(float *d_m, const float *d_bias, int rows, int cols) {
    dim3 threads_per_block(16, 16);
    dim3 blocks_per_grid((cols + threads_per_block.x - 1) / threads_per_block.x,
                         (rows + threads_per_block.y - 1) / threads_per_block.y);

    bias_add_kernel<<<blocks_per_grid, threads_per_block>>>(d_m, d_bias, rows, cols);
}

__global__ void copy_matrix_kernel(const float *src, float *dest, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        dest[idx] = src[idx];
    }
}

extern "C" void copy_matrix_gpu(const float *d_src, float *d_dest, int size) {
    int threads_per_block = 256;
    int blcoks_per_grid = (size + threads_per_block - 1) / threads_per_block;

    copy_matrix_kernel<<< blcoks_per_grid, threads_per_block >>>(d_src, d_dest, size);
}

__global__ void scale_matrix_kernel(float *m, float scaler, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        m[idx] *= scaler;
    }
}

extern "C" void scale_matrix_gpu(float *d_m, float scaler, int size) {
    int threads_per_block = 256;
    int blocks_per_grid = (size + threads_per_block - 1) / threads_per_block;

    scale_matrix_kernel<<< blocks_per_grid, threads_per_block >>>(d_m, scaler, size);
}

__global__ void extract_row_kernel(const float *src, int row, float *dest, int cols) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < cols) {
        dest[col] = src[row * cols + col];
    }
}

extern "C" void extract_row_gpu(const float *d_src, int row, float *d_dest, int cols) {
    int threads_per_block = 256;
    int blocks_per_grid = (cols + threads_per_block - 1) / threads_per_block;

    extract_row_kernel<<< blocks_per_grid, threads_per_block >>>(d_src, row, d_dest, cols);
}

// =================================================================
// 5. バイアス勾配計算カーネル (各列の縦方向の総和を計算)
// =================================================================
__global__ void compute_bias_gradient_kernel(const float *dOut, float *dBias, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols) {
        // 各スレッドが、自分の担当する要素の値を、対応する列のバイアス箱に安全に足し算する
        // (原子操作 atomicAdd を使うことでスレッド間の衝突を防ぐ)
        atomicAdd(&dBias[col], dOut[row * cols + col]);
    }
}

extern "C" void compute_bias_gradient_gpu(const float *d_dOut, float *d_dBias, int rows, int cols) {
    // 最初にバイアス勾配の箱をゼロクリアする（前回のゴミを残さないため）
    cudaMemset(d_dBias, 0, cols * sizeof(float));

    dim3 threads_per_block(16, 16);
    dim3 blocks_per_grid((cols + threads_per_block.x - 1) / threads_per_block.x,
                         (rows + threads_per_block.y - 1) / threads_per_block.y);

    compute_bias_gradient_kernel<<<blocks_per_grid, threads_per_block>>>(d_dOut, d_dBias, rows, cols);
}

// =================================================================
// 6. ReLU 順伝播 ＆ 逆伝播カーネル
// =================================================================
__global__ void relu_kernel(float *m, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size && m[idx] < 0.0f) {
        m[idx] = 0.0f;
    }
}

extern "C" void relu_gpu(float *d_m, int size) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    relu_kernel<<<blocks, threads>>>(d_m, size);
}

__global__ void backward_relu_kernel(const float *dOut, const float *Out, float *dIn, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        dIn[idx] = (Out[idx] > 0.0f) ? dOut[idx] : 0.0f;
    }
}

extern "C" void backward_relu_gpu(const float *d_dOut, const float *d_Out, float *d_dIn, int size) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    backward_relu_kernel<<<blocks, threads>>>(d_dOut, d_Out, d_dIn, size);
}

// =================================================================
// 7. Softmax 順伝播 ＆ 逆伝播カーネル (行ごとに独立して処理)
// =================================================================
__global__ void softmax_row_kernel(float *m, int row, int cols) {
    float *row_data = &m[row * cols];

    // 1. 最大値の探索 (オーバーフロー対策)
    float max_val = row_data[0];
    for (int i = 1; i < cols; i++) {
        if (row_data[i] > max_val) max_val = row_data[i];
    }

    // 2. 指数関数の計算と総和
    float sum = 0.0f;
    for (int i = 0; i < cols; i++) {
        row_data[i] = expf(row_data[i] - max_val);
        sum += row_data[i];
    }

    // 3. 正規化
    for (int i = 0; i < cols; i++) {
        row_data[i] /= sum;
    }
}

extern "C" void softmax_row_gpu(float *d_m, int row, int cols) {
    // 1行の処理なので、1ブロック・1スレッドのみ起動させて内部ループを回す
    softmax_row_kernel<<<1, 1>>>(d_m, row, cols);
}

__global__ void backward_softmax_kernel(const float *dA, const float *A, float *dS, int rows, int cols) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; // 各スレッドが「1つの行」を担当
    if (i < rows) {
        // 1. 期待値の計算 sum_da_a = Σ (dA_i * y_i)
        float sum_da_a = 0.0f;
        for (int k = 0; k < cols; k++) {
            int idx = i * cols + k;
            sum_da_a += dA[idx] * A[idx];
        }

        // 2. 下流への誤差を確定: dS_j = y_j * (dA_j - sum_da_a)
        for (int j = 0; j < cols; j++) {
            int idx = i * cols + j;
            dS[idx] = A[idx] * (dA[idx] - sum_da_a);
        }
    }
}

extern "C" void backward_softmax_gpu(const float *d_dA, const float *d_A, float *d_dS, int rows, int cols) {
    int threads = 64; // 行の数（SEQ_LENなど、通常は非常に小さい）に合わせた並列化
    int blocks = (rows + threads - 1) / threads;
    backward_softmax_kernel<<<blocks, threads>>>(d_dA, d_A, d_dS, rows, cols);
}

// =================================================================
// 8. LayerNorm 順伝播カーネル
// =================================================================
__global__ void forward_layernorm_kernel(const float *X, const float *gamma, const float *beta, float *Out, int rows, int cols) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; // 各スレッドが1つの行を担当
    if (i < rows) {
        float eps = 1e-5f;
        
        // 1. 平均の計算
        float sum = 0.0f;
        for (int j = 0; j < cols; j++) sum += X[i * cols + j];
        float mean = sum / (float)cols;

        // 2. 分散の計算
        float var_sum = 0.0f;
        for (int j = 0; j < cols; j++) {
            float diff = X[i * cols + j] - mean;
            var_sum += diff * diff;
        }
        float var = var_sum / (float)cols;

        // 3. 正規化とアフィン変換
        for (int j = 0; j < cols; j++) {
            int idx = i * cols + j;
            float x_hat = (X[idx] - mean) / sqrtf(var + eps);
            Out[idx] = gamma[j] * x_hat + beta[j];
        }
    }
}

extern "C" void forward_layernorm_gpu(const float *d_X, const float *d_gamma, const float *d_beta, float *d_Out, int rows, int cols) {
    int threads = 64;
    int blocks = (rows + threads - 1) / threads;
    forward_layernorm_kernel<<<blocks, threads>>>(d_X, d_gamma, d_beta, d_Out, rows, cols);
}

// =================================================================
// 9. LayerNorm 逆伝播カーネル
// =================================================================
__global__ void backward_layernorm_kernel(const float *dOut, const float *X, const float *gamma, float *dX, float *dgamma, float *dbeta, int rows, int cols) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; // 行ごとの並列処理
    if (i < rows) {
        float eps = 1e-5f;

        float sum = 0.0f;
        for (int j = 0; j < cols; j++) sum += X[i * cols + j];
        float mean = sum / (float)cols;

        float var_sum = 0.0f;
        for (int j = 0; j < cols; j++) {
            float diff = X[i * cols + j] - mean;
            var_sum += diff * diff;
        }
        float var = var_sum / (float)cols;
        float inv_std = 1.0f / sqrtf(var + eps);

        float sum_dout_xhat = 0.0f;
        float sum_dout = 0.0f;

        for (int j = 0; j < cols; j++) {
            int idx = i * cols + j;
            float x_hat = (X[idx] - mean) * inv_std;

            // パラメータの勾配蓄積 (スレッド間衝突を防ぐため atomicAdd)
            atomicAdd(&dgamma[j], dOut[idx] * x_hat);
            atomicAdd(&dbeta[j], dOut[idx]);

            sum_dout_xhat += dOut[idx] * gamma[j] * x_hat;
            sum_dout      += dOut[idx] * gamma[j];
        }

        for (int j = 0; j < cols; j++) {
            int idx = i * cols + j;
            float x_hat = (X[idx] - mean) * inv_std;
            dX[idx] = (inv_std / (float)cols) * (
                (float)cols * dOut[idx] * gamma[j] - sum_dout - x_hat * sum_dout_xhat
            );
        }
    }
}

extern "C" void backward_layernorm_gpu(const float *d_dOut, const float *d_X, const float *d_gamma, float *d_dX, float *d_dgamma, float *d_dbeta, int rows, int cols) {
    int threads = 64;
    int blocks = (rows + threads - 1) / threads;
    backward_layernorm_kernel<<<blocks, threads>>>(d_dOut, d_X, d_gamma, d_dX, d_dgamma, d_dbeta, rows, cols);
}

// =================================================================
// 10. Embedding 順伝播 ＆ 逆伝播 ＆ 位置エンコーディング
// =================================================================
__global__ void forward_embedding_kernel(const int *input_ids, const float *table, float *X, int seq_len, int embed_dim) {
    int t = blockIdx.y * blockDim.y + threadIdx.y;
    int d = blockIdx.x * blockDim.x + threadIdx.x;

    if (t < seq_len && d < embed_dim) {
        int id = input_ids[t];
        X[t * embed_dim + d] = table[id * embed_dim + d];
    }
}

extern "C" void forward_embedding_gpu(const int *d_input_ids, const float *d_table, float *d_X, int seq_len, int embed_dim) {
    dim3 threads(16, 16);
    dim3 blocks((embed_dim + threads.x - 1) / threads.x, (seq_len + threads.y - 1) / threads.y);
    forward_embedding_kernel<<<blocks, threads>>>(d_input_ids, d_table, d_X, seq_len, embed_dim);
}

__global__ void apply_positional_encoding_kernel(float *X, int seq_len, int embed_dim) {
    int pos = blockIdx.y * blockDim.y + threadIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x; // embed_dim / 2 までのインデックス

    if (pos < seq_len && i < embed_dim / 2) {
        float budget = (float)(2 * i) / (float)embed_dim;
        float denom = powf(10000.0f, budget);

        float sin_val = sinf((float)pos / denom);
        float cos_val = cosf((float)pos / denom);

        X[pos * embed_dim + (2 * i)]     += sin_val;
        X[pos * embed_dim + (2 * i + 1)] += cos_val;
    }
}

extern "C" void apply_positional_encoding_gpu(float *d_X, int seq_len, int embed_dim) {
    dim3 threads(16, 16);
    dim3 blocks(((embed_dim / 2) + threads.x - 1) / threads.x, (seq_len + threads.y - 1) / threads.y);
    apply_positional_encoding_kernel<<<blocks, threads>>>(d_X, seq_len, embed_dim);
}

__global__ void backward_embedding_kernel(const float *dX, const int *input_ids, float *table, int seq_len, int embed_dim) {
    int t = blockIdx.y * blockDim.y + threadIdx.y;
    int d = blockIdx.x * blockDim.x + threadIdx.x;

    if (t < seq_len && d < embed_dim) {
        int id = input_ids[t];
        // 異なるタイムステップで同じIDが出現した際、衝突を防ぐため atomicAdd
        atomicAdd(&table[id * embed_dim + d], dX[t * embed_dim + d]);
    }
}

extern "C" void backward_embedding_gpu(const float *d_dX, const int *d_input_ids, float *d_table, int seq_len, int embed_dim) {
    dim3 threads(16, 16);
    dim3 blocks((embed_dim + threads.x - 1) / threads.x, (seq_len + threads.y - 1) / threads.y);
    backward_embedding_kernel<<<blocks, threads>>>(d_dX, d_input_ids, d_table, seq_len, embed_dim);
}

// =================================================================
// 11. 最上層の Cross Entropy 誤差計算 (予測確率から正解ラベル 1.0f を引く)
// =================================================================
__global__ void compute_cross_entropy_grad_kernel(const float *prob, float *d_out, int target_id, int vocab_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < vocab_size) {
        d_out[idx] = prob[idx];
        if (idx == target_id) {
            d_out[idx] -= 1.0f;
        }
    }
}

extern "C" void compute_cross_entropy_grad_gpu(const float *d_prob, float *d_d_out, int target_id, int vocab_size) {
    int threads = 256;
    int blocks = (vocab_size + threads - 1) / threads;
    compute_cross_entropy_grad_kernel<<<blocks, threads>>>(d_prob, d_d_out, target_id, vocab_size);
}

// =================================================================
// 12. 特定の行への書き戻しカーネル (src_1行 を dest行列 の指定行に上書き)
// =================================================================
__global__ void write_row_kernel(const float *src, float *dest, int target_row, int cols) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < cols) {
        dest[target_row * cols + col] = src[col];
    }
}

extern "C" void write_row_gpu(const float *d_src, float *d_dest, int target_row, int cols) {
    int threads = 256;
    int blocks = (cols + threads - 1) / threads;
    write_row_kernel<<<blocks, threads>>>(d_src, d_dest, target_row, cols);
}