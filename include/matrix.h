#ifndef MATRIX_H
#define MATRIX_H

// 行列の情報を表す構造体
typedef struct {
  float *data;  // ヒープ領域へのポインタ
  int rows;     // 行数 (縦のサイズ)
  int cols;     // 列数 (横のサイズ)
} Matrix;

Matrix create_matrix(int rows, int cols);
void free_matrix(Matrix *M);
void mat_mul(const Matrix *A, const Matrix *B, const Matrix *C);
void mat_mul_a_trans(const Matrix *A, const Matrix *B, const Matrix *C);
void mat_mul_b_trans(const Matrix *A, const Matrix *B, const Matrix *C);

#endif // MATRIX_H