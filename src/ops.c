#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "ops.h"

// ===============
// === 行列関連 ===
// ===============

// 行列 src の指定された 行 row のデータを 1行の行列 dest にコピーする
void extract_row(const Matrix *src, int row, Matrix *dest) {
  int cols = src->cols;

  for (int j = 0; j < cols; j++) {
    dest->data[j] = src->data[row * cols + j];
  }
}

// 行列のすべての要素にスカラー(実数)を掛け算する
void scale_matrix(Matrix *m, float scalar) {
  int total_elements = m->rows * m->cols;
  for (int i = 0; i < total_elements; i++) {
    m->data[i] *= scalar;
  }
}

// ===================
// === 順伝播の関数 ===
// ===================

// 埋め込み(Embedding)層
// input_ids: [SEQ_LEN] のID配列
// embedding_table: モデルが持つ単語埋め込み行列 [VOCAB_SIZE x EMBED_DIM]
// X: 出力先の行列 [SEQ_LEN x EMBED_DIM]
void forward_embedding(
  const int *input_ids,
  int seq_len,
  int embed_dim,
  const Matrix *embedding_table,
  Matrix *X
) {
  for (int t = 0; t < seq_len; t++) {
    int id = input_ids[t];
    for (int d = 0; d < embed_dim; d++) {
      X->data[t * embed_dim + d] = embedding_table->data[id * embed_dim + d];
    }
  }
}

// 位置エンコーディング
// サイン・コサイン方式 (絶対位置エンコーディング)
// 行列 x [SEQ_LEN x EMBED_DIM] に対し、位置エンコーディングを直接足し算する
void apply_positional_encoding(Matrix *X) {
  int seq_len = X->rows;
  int embed_dim = X->cols;

  for (int pos = 0; pos < seq_len; pos++) {
    for (int i = 0; i < embed_dim / 2; i++) {
      // 分母の計算: 10000^(2i / d_model)
      float budget = (float)(2 * i) / (float)embed_dim;
      float denom = powf(10000.0f, budget);

      // 偶数次元には Sin, 奇数次元には Cos
      float sin_val = sinf((float)pos / denom);
      float cos_val = cosf((float)pos / denom);

      // 元の行列 X に足し算する
      X->data[pos * embed_dim + (2 * i)] += sin_val;
      X->data[pos * embed_dim + (2 * i + 1)] += cos_val;
    }
  }
}

// layerNormの順伝播
void forward_layernom(
  const Matrix *X,
  const Matrix *gamma,
  const Matrix *beta,
  Matrix *Out
) {
  int rows = X->rows;
  int cols = X->cols;
  float eps = 1e-5f; // 0除算を防ぐ微小値

  for (int i = 0; i < rows; i++) {
    // 1. 平均 μ の計算
    float sum = 0.0f;
    for (int j = 0; j < cols; j++) {
      sum += X->data[i * cols + j];
    }
    float mean = sum / (float)cols;

    // 2. 分散 σ^2 の計算
    float var_sum = 0.0f;
    for (int j = 0; j < cols; j++) {
      float diff = X->data[i * cols + j] - mean;
      var_sum += diff * diff;
    }
    float var = var_sum / (float)cols;

    // 3. 正規化とアフィン変換 (y = gamma * x_hat + beta)
    for (int j = 0; j < cols; j++) {
      int idx = i * cols + j;
      float x_hat = (X->data[idx] - mean) / sqrtf(var + eps);

      Out->data[idx] = gamma->data[j] * x_hat + beta->data[j];
    }
  }
}

// ソフトマックス関数
void softmax_row(Matrix *m, int row) {
  int cols = m->cols;
  float *row_data = &m->data[row * cols];

   // 最大値の探索 (オーバーフロー対策)
  float max_val = row_data[0];
  for (int i = 1; i < cols; i++) {
    if (row_data[i] > max_val) max_val = row_data[i];
  }

  // 指数関数の計算と総和
  float sum = 0.0f;
  for (int i = 0; i < cols; i++) {
    row_data[i] = expf(row_data[i] - max_val);
    sum += row_data[i];
  }

  // 正規化
  for (int i = 0; i < cols; i++) {
    row_data[i] /= sum;
  }
}

void relu(Matrix *m) {
  int total_elemnts = m->rows * m->cols;
  for (int i = 0; i < total_elemnts; i++) {
    if (m->data[i] < 0.0f) {
      m->data[i] = 0.0f;
    }
  }
}

void bias_add(Matrix *M, const Matrix *bias) {
  int rows = M->rows;
  int cols = M->cols;
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      M->data[i * cols + j] += bias->data[j];
    }
  }
}

void compute_bias_gradient(const Matrix *dOut, Matrix *dBias) {
  int rows = dOut->rows;
  int cols = dOut->cols;

  for (int i = 0; i < cols; i++) {
    float sum = 0.0f;
    for (int j = 0; j < rows; j++) {
      sum += dOut->data[j * cols + i];
    }
    dBias->data[i] = sum;
  }
}
void copy_matrix(const Matrix *src, Matrix *dest) {
  int total_elements = src->rows * src->cols;
  for (int i = 0; i < total_elements; i++) {
    dest->data[i] = src->data[i];
  }
}


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
) {
  for (int t = 0; t < seq_len; t++) {
    int id = input_ids[t]; // 今回の文字ID を特定
    
    for (int d = 0; d < embed_dim; d++) {
      // One-Hot 行列との積
      // 大量の 0 との掛け算を省くためにインデックスを指定して加算
      embedding_table->data[id * embed_dim + d] += dX->data[t * embed_dim + d];
    }
  }
}

// layerNormの逆伝播
void backward_layernorm(
  const Matrix *dOut,
  const Matrix *X,
  const Matrix *gamma,
  Matrix *dX,
  Matrix *dgamma,
  Matrix *dbeta
) {
  int rows = X->rows;
  int cols = X->cols;
  float eps = 1e-5f;

  for (int i = 0; i < rows; i++) {
    // 順伝播の平均と分散を再計算
    float sum = 0.0f;
    for (int j = 0; j < cols; j++) sum += X->data[i * cols + j];
    float mean = sum / (float)cols;

    float var_sum = 0.0f;
    for (int j = 0; j < cols; j++) {
      float diff = X->data[i * cols + j] - mean;
      var_sum += diff * diff;
    }
    float var = var_sum / (float)cols;
    float inv_std = 1.0f / sqrtf(var + eps);

    // 中間計算用
    float sum_dout_xhat = 0.0f;
    float sum_dout = 0.0f;
    for (int j = 0; j < cols; j++) {
      int idx = i * cols + j;
      float x_hat = (X->data[idx] - mean) * inv_std;

      // パラメーターの勾配蓄積
      dgamma->data[j] += dOut->data[idx] * x_hat;
      dbeta->data[j] += dOut->data[idx];

      sum_dout_xhat += dOut->data[idx] * gamma->data[j] * x_hat;
      sum_dout      += dOut->data[idx] * gamma->data[j];
    }

    // 下流への入力誤差 dX の計算 (公式の展開)
    for (int j = 0; j < cols; j++) {
      int idx = i * cols + j;
      float x_hat = (X->data[idx] - mean) * inv_std;
      dX->data[idx] = (inv_std / (float)cols) * (
        (float)cols * dOut->data[idx] * gamma->data[j] - sum_dout - x_hat * sum_dout_xhat
      );
    }
  }
}

void backward_softmax(const Matrix *dA, const Matrix *A, Matrix *dS) {
  int rows = A->rows;
  int cols = A->cols;

  // 行ごとにSoftmaxの逆伝播（微分の合流）を計算
  for (int i = 0; i < rows; i++) {

    /* * 【数学的背景の解説】
     * Softmax公式: y_i = exp(x_i) / Σ exp(x_k)
     * * 1つの入力 x_j を動かすと、分母の「総和」を通じてすべての出力 y_1 ~ y_n が変化するため、
     * チェインルールにより、すべての出力から戻ってきた誤差 dA_i の責任を合計する必要がある。
     * * 商の微分公式より、x_j に関する微分は以下の2ケースに分かれる:
     * 1) 自分が分子にいるとき (i == j): ∂y_i / ∂x_i = y_i * (1 - y_i)
     * 2) 自分が分母にいるとき (i != j): ∂y_i / ∂x_j = -y_i * y_j
     * * これらをチェインルールで合流させて整理すると、最終的な誤差 dS_j の公式は以下になる:
     * 上流から戻ってきた誤差を dA_i、求めたいSoftmax前の生スコアへの誤差を dS_j とします。
     * dS_j = y_j * ( dA_j - Σ (dA_i * y_i) )
     * * つまり、「自分の上流誤差(dA_j)」と「全体の誤差の期待値(Σ dA_i * y_i)」の
     * 差分（どれだけ平均より突出してズレているか）に対して、自分の存在感（y_j）を掛け算する。
     * 
     * y_i => A_prob.data[idx] (順伝播のSoftmax出力確率)
     * dA_i => dA.data[idx]    (上流からの誤差)
     * dS_j => dS.data[idx]    (求める生のスコアの誤差)
     */

    // 1. 公式の右側にある「誤差の期待値（総和部分）」を先に計算: sum_da_a = Σ (dA_i * y_i)
    float sum_da_a = 0.0f;
    for (int k = 0; k < cols; k++) {
      int idx = i * cols + k;
      sum_da_a += dA->data[idx] * A->data[idx];
    }
    
    // 2. 公式の外側を掛け算して下流への誤差を確定: dS_j = y_j * (dA_j - sum_da_a)
    for (int j = 0; j < cols; j++) {
      int idx = i * cols + j;
      dS->data[idx] = A->data[idx] * (dA->data[idx] - sum_da_a);
    }
  }
}

void backward_relu(const Matrix *dOut, const Matrix *Out, Matrix *dIn) {
  int total_elements = Out->rows * Out->cols;

  for (int i = 0; i < total_elements; i++) {
    // 順伝播の出力が 0 より大きければ誤差をそのまま流す
    if (Out->data[i] > 0.0f) {
      dIn->data[i] = dOut->data[i];
    } else {
      // 0 以下なら下流への誤差は 0 になる
      dIn->data[i] = 0.0f;
    }
  } 
}

// ============================
// === 重みパラメーターの更新 ===
// ============================

// パラメーター更新の式: W = W - lr *dW
void gradient_descent_update(Matrix *W, const Matrix *dW, float lr) { 
  int size = W->rows * W->cols;

  sgd_update_gpu(W->device_data, dW->device_data, lr, size);

  // for (int i = 0; i < size; i++) {
  //   W->data[i] -= lr * dW->data[i];
  // }
}

// =====================================
// === 重みパラメーターの保存・読み込み ===
// =====================================

// 行列のデータをファイルに保存する
void save_matrix(const Matrix *m, const char *filename) {
  FILE *f = fopen(filename, "wb");

  if (f == NULL) {
    fprintf(stderr, "Error: ファイル %s を開けませんでした。\n", filename);
    return;
  }

  // 行数、列数、データの中身を順番にバイナリとして書き出す
  fwrite(&m->rows, sizeof(int), 1, f);
  fwrite(&m->cols, sizeof(int), 1, f);
  fwrite(m->data, sizeof(float), m->rows * m->cols, f);

  fclose(f);
}

// ファイルから行列のデータを読み込む 
void load_matrix(Matrix *m, const char *filename) {
  FILE *f = fopen(filename, "rb");
  if (f == NULL) {
    fprintf(stderr, "Error: ファイル %s を開けませんでした。\n", filename);
    return;
  }

  int rows, cols;
  fread(&rows, sizeof(int), 1, f);
  fread(&cols, sizeof(int), 1, f);

  // 安全チェック: 読み込もうとしているファイルの行列サイズが、プログラム側の想定と一致しているか
  if (rows != m->rows || cols != m->cols) {
    fprintf(stderr, "Error: %s の行列サイズ [%dx%d] が、プログラムの想定 [%dx%d] 与えられた行列と不一致です。\n",
            filename, rows, cols, m->rows, m->cols);
    fclose(f);
    return;
  }

  // データをメモリに読み込む
  fread(m->data, sizeof(float), rows * cols, f);

  fclose(f);
}