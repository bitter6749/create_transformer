#include <math.h>
#include "ops.h"

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

// 行列のすべての要素にスカラー(実数)を掛け算する
void scale_matrix(Matrix *m, float scalar) {
  int total_elements = m->rows * m->cols;
  for (int i = 0; i < total_elements; i++) {
    m->data[i] *= scalar;
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