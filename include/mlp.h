#ifndef MLP_H
#define MLP_H

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

// ‹t“`”d‚Í‚Ü‚¾–¢’è‹`

#endif // MLP_H