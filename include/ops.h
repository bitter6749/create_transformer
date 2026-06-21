#ifndef OPS_H
#define OPS_H

#include "matrix.h"

// èáì`îd
void softmax_row(Matrix *m, int row);
void relu(Matrix *m);
void layer_norm(const Matrix *X, Matrix  *Y, float *gamma, float *beta);
void residual_add(const Matrix *X_in, const Matrix *X_layer, Matrix *Y);

// ãtì`îd
void backward_softmax(const Matrix *dA, const Matrix *A, Matrix *dS);
void backward_relu(const Matrix *dOut, const Matrix *Out, Matrix *dIn);

#endif // OPS_H