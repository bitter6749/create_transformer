#include <math.h>
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
  // 【入力されたID】      【token_embedding（対応表）】          【出力される行列 X】
  // id = 2  ('c')  ───>  [行0]  ( 'a' の16次元ベクトル )         [1行目] ('c' のベクトル)
  // id = 0  ('a')  ───>  [行1]  ( 'b' の16次元ベクトル ) ───>   [2行目] ('a' のベクトル)
  // id = 19 ('t')  ───>  [行2]  ( 'c' の16次元ベクトル )         [3行目] ('t' のベクトル)
  // id = 26 (' ')  ───>   ...                                    [4行目] (' ' のベクトル)
  //                      [行26] ( ' ' の16次元ベクトル )
  for (int t = 0; t < seq_len; t++) {
    int id = input_ids[t];
    for (int d = 0; d < embed_dim; d++) {
      X->data[t * embed_dim + d] = embedding_table->data[id * embed_dim + d];
    }
  }
}

// 位置エンコーディング
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
  int total_elements = W->rows * W->cols;
  for (int i = 0; i < total_elements; i++) {
    W->data[i] -= lr * dW->data[i];
  }
}