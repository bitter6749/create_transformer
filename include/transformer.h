#ifndef TRANSFORMER_H
#define TRANSFORMER_H

#include "model.h"
#include "matrix.h"

// 設定
#define VOCAB_SIZE 27 // a~z + space
#define SEQ_LEN    4  // 入力する文字数(例: "cat " の四文字)
#define EMBED_DIM  16  // 文字を表現するベクトルの次元数 (RNN の HIDDENに相当)

// 出力層の順伝播
// 入力特徴量 X_features の最後のトークンから、次に来る文字の確立分布を計算する
void forward_lm_head(
  const Matrix *X_features, // [SEQ_LEN x EMBED_DIM] (Attention または MLP の出力)
  const Matrix *W_out,      // [VOCAB_SIZE x EMBED_DIM]
  Matrix *output_probabilities  // [1 x VOCAB_SIZE] (最終出力)
);

// モデル全体の順伝播
void forward_transformer(
  SimpleTransformer *model,
  const int *input_ids,
  Matrix *output_probabilities
);

// モデル全体の逆伝播 (各層の backward を順番に呼ぶ)
void backward_transformer(
  SimpleTransformer *model,
  const int *input_ids,
  int target_id,

  // 各層の重み・バイアスの勾配
  Matrix *dW_out,
  Matrix *dW1, Matrix *db1, Matrix *dW2, Matrix *db2, 
  Matrix *dW_q, Matrix *dW_k, Matrix *dW_v
);

#endif // TRANSFORMER_H