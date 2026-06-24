#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include "matrix.h"

// ============================
// === 行列計算のヘルパー関数 ===
// ============================

// 行列のメモリを確保し、0で初期化する関数
Matrix create_matrix(int rows, int cols) {
  Matrix M;
  M.rows = rows;
  M.cols = cols;

  // CPU: メモリの確保と初期化を同時に行う
  M.data = (float *)calloc(rows * cols, sizeof(float));
  if (M.data == NULL) {fprintf(stderr, "Error: CPU memory allocation failed for %d x %d\n", rows, cols);
    exit(EXIT_FAILURE);
  }
  
  // GPU: メモリの確保と初期化を同時に行う
  cudaError_t err = cudaMalloc((void **)&M.device_data, rows * cols * sizeof(float));
  if (err != cudaSuccess) {
    fprintf(stderr, "Error: GPU memory allocation failed for size %d x %d (%llu bytes): %s\n", 
            rows, cols, (unsigned long long)(rows * cols * sizeof(float)), cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  return M;
}

// 行列のメモリを安全に開放する関数
void free_matrix(Matrix *M) {
  // CPU側のメモリ解放
  if (M->data != NULL) {
    free(M->data);
    // 二重開放 (Double Free) を防ぐ
    M->data = NULL; 
  }

  // GPU側のメモリ解放
  if (M->device_data != NULL) {
    cudaFree(M->device_data);
    M->device_data = NULL;
  }

  M->rows = 0;
  M->cols = 0;
}

// CPUからGPUへデータを送る
void upload_matrix(const Matrix *M) {
  if (M->data != NULL && M->device_data != NULL) {
    cudaMemcpy(M->device_data, M->data, M->rows * M->cols * sizeof(float), cudaMemcpyHostToDevice);
  }
}

// GPUからCPUへデータを回収する
void download_matrix(const Matrix *M) {
  if (M->data != NULL && M->device_data != NULL) {
    cudaMemcpy(M->data, M->device_data, M->rows * M->cols * sizeof(float), cudaMemcpyDeviceToHost);
  }
}

// 通常の行列掛け算 C = A ・ B
void mat_mul(const Matrix *A, const Matrix *B, Matrix *C) {
  // 行列のサイズから、計算可能かチェック
  if (A->cols != B->rows || C->rows != A->rows || C->cols != B->cols) {
    fprintf(stderr, "Error: 行列のサイズが不一致です。\n");
    exit(EXIT_FAILURE);
  }

  matrix_mul_gpu(
    A->device_data,
    B->device_data,
    C->device_data,
    A->rows,
    A->cols,
    B->cols
  );

  // for (int A_row = 0; A_row < A->rows; A_row++) {
  //   for (int B_col = 0; B_col < B->cols; B_col++) {
  //     // 各行と列の内積
  //     float inner_product = 0.0f;

  //     for (int A_col = 0; A_col < A->cols; A_col++) {
  //       float a = A->data[A_row * A->cols + A_col];
  //       float b = B->data[A_col * B->cols + B_col];

  //       inner_product += a * b;
  //     }
  //     C->data[A_row * C->cols + B_col] = inner_product;
  //   }
  // }
}

// 行列の足し算 C = A + B
void mat_add(const Matrix *A, const Matrix *B, Matrix *C) {
  // サイズチェック
  if (A->rows != B->rows || A->cols != B->cols ||
      A->rows != C->rows || A->cols != C->cols) {

    fprintf(stderr, "Error: 行列のサイズが不一致です。\n");
    exit(EXIT_FAILURE);
  }

  int total_elements = A->rows * A->cols;

  // for (int i = 0; i < total_elements; i++) {
  //   C->data[i] = A->data[i] + B->data[i];
  // }

  matrix_add_gpu(A->device_data, B->device_data, C->device_data, total_elements);
}

// 左側転置の行列掛け算 C = A^T ・ B
void mat_mul_a_trans(const Matrix *A, const Matrix *B, Matrix *C) {
  // Aは転置されるので、Aの行数とBの行数が一致すべき
  if (A->rows != B->rows || C->rows != A->cols || C->cols != B->cols) {
    fprintf(stderr, "Error: 行列のサイズが不一致です。\n");
    exit(EXIT_FAILURE);
  }

  matrix_mul_a_trans_gpu(
    A->device_data,
    B->device_data,
    C->device_data,
    A->rows,
    A->cols,
    B->cols
  );

  // for (int A_col = 0; A_col < A->cols; A_col++) {
  //   for (int B_col = 0; B_col < B->cols; B_col++) {
  //     // 各行の列と内積
  //     float inner_product = 0.0f;

  //     for (int A_row = 0; A_row < A->rows; A_row++) {
  //       float a = A->data[A_row * A->cols + A_col];
  //       float b = B->data[A_row * B->cols + B_col];

  //       inner_product += a * b;
  //     }
  //     C->data[A_col * C->cols + B_col] = inner_product;
  //   }
  // }
}

// 右側転置の行列掛け算 C = A ・ B^T
void mat_mul_b_trans(const Matrix *A, const Matrix *B, Matrix *C) {
  // Bは転置されるので、Aの列数とBの列数が一致すべき
  if (A->cols != B->cols || C->rows != A-> rows || C->cols != B->rows) {
    fprintf(stderr, "Error: 行列のサイズが不一致です。\n");
    exit(EXIT_FAILURE);
  }

  matrix_mul_b_trans_gpu(
    A->device_data,
    B->device_data,
    C->device_data,
    A->rows,
    A->cols,
    B->rows
  );

  // for (int A_row = 0; A_row < A->rows; A_row++) {
  //   for (int B_row = 0; B_row < B->rows; B_row++) {
  //     // 各行と列の内積
  //     float inner_product = 0.0f;

  //     for (int A_col = 0; A_col < A->cols; A_col++) {
  //       float a = A->data[A_row * A->cols + A_col];
  //       float b = B->data[B_row * B->cols + A_col];
        
  //       inner_product += a * b;
  //     }
  //     C->data[A_row * C->cols + B_row] = inner_product;
  //   }
  // }
}

// GPU上で平均化とクリッピングを一気に実行
void mat_avg_clip(Matrix *acc_dW, float inv_b, float clip_val) {
  int size = acc_dW->rows * acc_dW->cols;
  matrix_avg_clip_gpu(acc_dW->device_data, inv_b, clip_val, size);
}

// ? GPU上で重みの引き算更新（W = W - lr * dW）を実行
void mat_sgd_update(Matrix *W, const Matrix *dW, float lr) {
  int size = W->rows * W->cols;
  sgd_update_gpu(W->device_data, dW->device_data, lr, size);
}

// ? GPU上のデータを一瞬で 0.0f にクリア
void mat_clear(Matrix *M) {
  int size = M->rows * M->cols;
  matrix_clear_gpu(M->device_data, size);
}