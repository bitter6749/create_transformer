#ifndef TRANSFORMER_H
#define TRANSFORMER_H

#include "model.h"
#include "matrix.h"

// 設定
#define VOCAB_SIZE 27 // a~z + space
#define SEQ_LEN    4  // 入力する文字数(例: "cat " の四文字)
#define EMBED_DIM  16  // 文字を表現するベクトルの次元数 (RNN の HIDDENに相当)

// ===================
// === Transformer ===
// ===================

// 順伝播関数
void forward_transformer(
  SimpleTransformer *model, 
  const int *input_ids, 
  Matrix *output_probabilties
);

// 逆伝播: 出力層
void backward_ouput(
  SimpleTransformer *model, 
  const Matrix *output_probabilities, 
  int target_id, 
  const Matrix *Z, 
  Matrix *dW_out, 
  Matrix*dZ
);

// 逆伝播: Attention層
void backward_attention(
  const Matrix *dZ,
  const Matrix *A_scores,
  const Matrix *V,
  Matrix *dA,
  Matrix *dV
);

// 逆伝播: Attention層のQ, K, V
void backward_attention_qkv(
    const Matrix *dA,
    const Matrix *A_prob,
    const Matrix *Q,
    const Matrix *K,
    Matrix *dQ,
    Matrix *dK
);

// =================
// === 活性化関数 ===
// =================

// ソフトマックス関数
void softmax(float *x, int size);

#endif // TRANSFORMER_H