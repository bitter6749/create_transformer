#ifndef TRANSFORMER_H
#define TRANSFORMER_H

#include "model.h"
#include "matrix.h"

// 設定
#define VOCAB_SIZE 27 // a~z + space
#define SEQ_LEN    4  // 入力する文字数(例: "cat " の四文字)
#define EMBED_DIM  16  // 文字を表現するベクトルの次元数 (RNN の HIDDENに相当)

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
  // 各層の重みの勾配
  Matrix *dW_out,
  Matrix *dW_q,
  Matrix *dW_k,
  Matrix *dW_v
);

#endif // TRANSFORMER_H