#ifndef MATRIX_H
#define MATRIX_H

typedef struct {
  int rows;     
  int cols;     
  float *data;        // CPU側のメモリポインタ
  float *device_data; // GPU(VRAM)側のメモリポインタ
} Matrix;

Matrix create_matrix(int rows, int cols);
void free_matrix(Matrix *M);

void upload_matrix(const Matrix *M); // CPUからGPUへデータを送る
void download_matrix(const Matrix *M); // GPUからCPUへデータを回収する

void mat_mul(const Matrix *A, const Matrix *B, Matrix *C);
void mat_add(const Matrix *A, const Matrix *B, Matrix *C);
void mat_mul_a_trans(const Matrix *A, const Matrix *B, Matrix *C);
void mat_mul_b_trans(const Matrix *A, const Matrix *B, Matrix *C);
void mat_avg_clip(Matrix *acc_dW, float inv_b, float clip_val);
void mat_sgd_update(Matrix *W, const Matrix *dW, float lr);
void mat_clear(Matrix *M);

// GPU版演算関数のプロトタイプ宣言
void matrix_add_gpu(const float* device_A, const float* device_B, float* device_C, int size);
void matrix_mul_gpu(const float* device_A, const float* device_B, float* device_C, int A_rows, int A_cols, int B_cols);
void matrix_mul_a_trans_gpu(const float* device_A, const float* device_B, float* device_C, int A_rows, int A_cols, int B_cols);
void matrix_mul_b_trans_gpu(const float* device_A, const float* device_B, float* device_C, int A_rows, int A_cols, int B_rows);

void matrix_avg_clip_gpu(float *d_acc_dW, float inv_b, float clip_val, int size);
void sgd_update_gpu(float *d_W, const float *d_dW, float lr, int size);
void matrix_clear_gpu(float *d_data, int size);

void bias_add_gpu(float *d_m, const float *d_bias, int rows, int cols);
void copy_matrix_gpu(const float *d_src, float *d_dest, int size);
void scale_matrix_gpu(float *d_m, float scalar, int size);
void extract_row_gpu(const float *d_src, int row, float *d_dest, int cols);

void compute_bias_gradient_gpu(const float *d_dOut, float *d_dBias, int rows, int cols);
void relu_gpu(float *d_m, int size);
void backward_relu_gpu(const float *d_dOut, const float *d_Out, float *d_dIn, int size);
void softmax_row_gpu(float *d_m, int row, int cols);
void backward_softmax_gpu(const float *d_dA, const float *d_A, float *d_dS, int rows, int cols);

void forward_layernorm_gpu(const float *d_X, const float *d_gamma, const float *d_beta, float *d_Out, int rows, int cols);
void backward_layernorm_gpu(const float *d_dOut, const float *d_X, const float *d_gamma, float *d_dX, float *d_dgamma, float *d_dbeta, int rows, int cols);
void forward_embedding_gpu(const int *d_input_ids, const float *d_table, float *d_X, int seq_len, int embed_dim);
void apply_positional_encoding_gpu(float *d_X, int seq_len, int embed_dim);
void backward_embedding_gpu(const float *d_dX, const int *d_input_ids, float *d_table, int seq_len, int embed_dim);

void compute_cross_entropy_grad_gpu(const float *d_prob, float *d_d_out, int target_id, int vocab_size);
void write_row_gpu(const float *d_src, float *d_dest, int target_row, int cols);

#endif // MATRIX_H