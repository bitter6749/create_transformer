#ifndef MLP_H
#define MLP_H

#include "transformer.h"
#include "matrix.h"

// 隠れ層の次元数を設定 (EMBED_DIM の 4倍が一般的)
#define MLP_HIDDEN_DIM    (EMBED_DIM * 4)

void forward_mlp(
  const Matrix *X, 
  const Matrix *W1,
  const Matrix *b1,
  const Matrix *W2,
  const Matrix *b2,
  Matrix *H,
  Matrix *Y
);

// 逆伝播はまだ未定義

#endif // MLP_H