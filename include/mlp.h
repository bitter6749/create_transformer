#ifndef MLP_H
#define MLP_H

#include "transformer.h"
#include "matrix.h"

void forward_mlp(
  const Matrix *X, 
  const Matrix *W1,
  const Matrix *b1,
  const Matrix *W2,
  const Matrix *b2,
  Matrix *H,
  Matrix *Y
);

void backward_mlp(
  const Matrix *dY, // 上流からの誤差 [SEQ_LEN x EMBED_DIM] 
  const Matrix *Z,  // 順伝播時の入力 [SEQ_LEN x EMBED_DIM]
  const Matrix *W1, // [EMBED_DIM x MLP_HIDDEN_DIM]
  const Matrix *H,  // 順伝播時のReLU適用後の出力 [SEQ_LEN x MLP_HIDDEN_DIM]
  const Matrix *W2, // [MLP_HIDDEN_DIM x EMBED_DIM]
  Matrix *dZ,       // 下流 (Attention) への誤差 [SEQ_LEN x EMBED_DIM]
  Matrix *dW1, Matrix *db1,
  Matrix *dW2, Matrix *db2
);

#endif // MLP_H