#include <stdlib.h>
#include <stdio.h>
#include "matrix.h"

// ============================
// === 行列計算のヘルパー関数 ===
// ============================

// 行列のメモリを確保し、0で初期化する関数
Matrix create_matrix(int rows, int cols) {
  Matrix M;
  M.rows = rows;
  M.cols = cols;

  // calloc: メモリの確保と初期化を同時に行う
  M.data = (float *)calloc(rows * cols, sizeof(float));

  if (M.data == NULL) {
    fprintf(stderr, "Error: 行列のメモリの確保に失敗しました。");
    exit(EXIT_FAILURE);
  }

  return M;
}

// 行列のメモリを安全に開放する関数
void free_matrix(Matrix *M) {
  if (M->data != NULL) {
    free(M->data);
    M->data = NULL; // 二重開放 (Double Free) を防ぐ
  }

  M->rows = 0;
  M->cols = 0;
}
// 通常の行列掛け算 C = A ・ B
void mat_mul(const Matrix *A, const Matrix *B, Matrix *C) {
  // 行列のサイズから、計算可能かチェック
  if (A->cols != B->rows || C->rows != A->rows || C->cols != B->cols) {
    fprintf(stderr, "Error: 行列のサイズが不一致です。\n");
    exit(EXIT_FAILURE);
  }

  for (int A_row = 0; A_row < A->rows; A_row++) {
    for (int B_col = 0; B_col < B->cols; B_col++) {
      // 各行と列の内積
      float inner_product = 0.0f;

      for (int A_col = 0; A_col < A->cols; A_col++) {
        float a = A->data[A_row * A->cols + A_col];
        float b = B->data[A_col * B->cols + B_col];

        inner_product += a * b;
      }
      C->data[A_row * C->cols + B_col] = inner_product;
    }
  }
}

// 左側転置の行列掛け算 C = A^T ・ B
void mat_mul_a_trans(const Matrix *A, const Matrix *B, Matrix *C) {
  // Aは転置されるので、Aの行数とBの行数が一致すべき
  if (A->rows != B->rows || C->rows != A->cols || C->cols != B->cols) {
    fprintf(stderr, "Error: 行列のサイズが不一致です。\n");
    exit(EXIT_FAILURE);
  }

  for (int A_col; A_col < A->cols; A_col++) {
    for (int B_col; B_col < B->cols; B_col++) {
      // 各行の列と内積
      float inner_product = 0.0f;

      for (int A_row; A_row < A->rows; A_row++) {
        float a = A->data[A_row * A->cols + A_col];
        float b = B->data[A_row * B->cols + B_col];

        inner_product += a * b;
      }
      C->data[A_col * C->cols + B_col] = inner_product;
    }
  }
}

// 右側転置の行列掛け算 C = A ・ B^T
void mat_mul_b_trans(const Matrix *A, const Matrix *B, Matrix *C) {
  // Bは転置されるので、Aの列数とBの列数が一致すべき
  if (A->cols != B->cols || C->rows != A-> rows || C->cols != B->rows) {
    fprintf(stderr, "Error: 行列のサイズが不一致です。\n");
    exit(EXIT_FAILURE);
  }

  for (int A_row = 0; A_row < A->rows; A_row++) {
    for (int B_row = 0; B_row < B->rows; B_row++) {
      // 各行と列の内積
      float inner_product = 0.0f;

      for (int A_col = 0; A_col < A->cols; A_col++) {
        float a = A->data[A_row * A->cols + A_col];
        float b = B->data[B_row * B->cols + A_col];
        
        inner_product += a * b;
      }
      C->data[A_row * C->cols + B_row] = inner_product;
    }
  }
}