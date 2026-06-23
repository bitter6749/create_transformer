#ifndef OPS_H
#define OPS_H

#include "matrix.h"

// ===============
// === 行列関連 ===
// ===============

// 行列 src の指定された 行 row のデータを 1行の行列 dest にコピーする
void extract_row(const Matrix *src, int row, Matrix *dest);

// スケール
void scale_matrix(Matrix *m, float scalar);

// ===================
// === 順伝播の関数 ===
// ===================

// 埋め込み(Embedding)層
void forward_embedding(
    const int *input_ids,
    int seq_len,
    int embed_dim,
    const Matrix *embedding_table,
    Matrix *X
);

// 位置エンコーディング
void apply_positional_encoding(Matrix *X);

// layerNormの順伝播
void forward_layernom(
  const Matrix *X,
  const Matrix *gamma,
  const Matrix *beta,
  Matrix *Out
);

void softmax_row(Matrix *m, int row);
void relu(Matrix *m);

// ===================
// === 逆伝播の関数 ===
// ===================

// 埋め込み(Embedding)層の逆伝播
void backward_embedding (
  const Matrix *dX,
  const int *input_ids,
  int seq_len, 
  int embed_dim,
  Matrix *embedding_table
);

// layerNormの逆伝播
void backward_layernorm(
  const Matrix *dOut,
  const Matrix *X,
  const Matrix *gamma,
  Matrix *dX,
  Matrix *dgamma,
  Matrix *dbeta
);

void backward_softmax(const Matrix *dA, const Matrix *A, Matrix *dS);
void backward_relu(const Matrix *dOut, const Matrix *Out, Matrix *dIn);

// ============================
// === 重みパラメーターの更新 ===
// ============================

void gradient_descent_update(Matrix *W, const Matrix *dW, float lr);

// =====================================
// === 重みパラメーターの保存・読み込み ===
// =====================================

// 行列のデータをファイルに保存する
void save_matrix(const Matrix *m, const char *filename);
// ファイルから行列のデータを読み込む (メモリ確保は事前に済んでいる前提)
void load_matrix(Matrix *m, const char *filename);

// GPUでの処理
void matrix_add_gpu(
  const float* host_A,
  const float* host_B,
  float* host_C,
  int size
);

#endif // OPS_H