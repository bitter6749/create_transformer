#ifndef ATTENTION_H
#define ATTENTION_H

#include "matrix.h"

void forward_attention(
  const Matrix *X,
  const Matrix *W_q,
  const Matrix *W_k,
  const Matrix *W_v,
  Matrix *Q, 
  Matrix *K, 
  Matrix *V,
  Matrix *A_prob, 
  Matrix *Z
);

void backward_attention(
  const Matrix *dZ,
  const Matrix *A_prob, 

  const Matrix *Q,
  const Matrix *K,
  const Matrix *V,
  const Matrix *X,

  const Matrix *W_q,
  const Matrix *W_k,
  const Matrix *W_v,

  Matrix *dX,
  Matrix *dW_q,
  Matrix *dW_k,
  Matrix *dW_v
);

#endif  // ATTENTION_H