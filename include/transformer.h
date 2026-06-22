#ifndef TRANSFORMER_H
#define TRANSFORMER_H

#include "model.h"
#include "matrix.h"

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

// 出力層の逆伝播
// 最終出力の確率分布と正解ID (target_id) から誤差を計算し、dY_mlp と dW_out を求める
void backward_lm_head(
  const Matrix *output_probabilities, // [1 x VOCAB_SIZE]
  int target_id,
  const Matrix *Y_mlp,                // [SEQ_LEN x EMBED_DIM]
  const Matrix *W_out,                // [EMBED_DIM x VOCAB_SIZE]
  Matrix *dY_mlp,                     // [SEQ_LEN x EMBED_DIM] (下流へ流す誤差)
  Matrix *dW_out                      // [EMBED_DIM x VOCAB_SIZE] (重みの勾配)
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