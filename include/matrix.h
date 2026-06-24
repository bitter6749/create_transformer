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
#endif // MATRIX_H