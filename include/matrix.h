#ifndef MATRIX_H
#define MATRIX_H

typedef struct {
  float *data;  
  int rows;     
  int cols;     
} Matrix;

Matrix create_matrix(int rows, int cols);
void free_matrix(Matrix *M);
void mat_mul(const Matrix *A, const Matrix *B, Matrix *C);
void mat_add(const Matrix *A, const Matrix *B, Matrix *C);
void mat_mul_a_trans(const Matrix *A, const Matrix *B, Matrix *C);
void mat_mul_b_trans(const Matrix *A, const Matrix *B, Matrix *C);

// GPU版演算関数のプロトタイプ宣言
void matrix_add_gpu(const float* host_A, const float* host_B, float* host_C, int size);
void matrix_mul_gpu(const float* host_A, const float* host_B, float* host_C, int A_rows, int A_cols, int B_cols);
void matrix_mul_a_trans_gpu(const float* host_A, const float* host_B, float* host_C, int A_rows, int A_cols, int B_cols);
void matrix_mul_b_trans_gpu(const float* host_A, const float* host_B, float* host_C, int A_rows, int A_cols, int B_rows);
#endif // MATRIX_H