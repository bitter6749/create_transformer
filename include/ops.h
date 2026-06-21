#ifndef OPS_H
#define OPS_H

#include "transformer.h"
#include "matrix.h"

// 埋め込み(Embedding)層
void forward_embedding(
    const int *input_ids,
    int seq_len,
    int embed_dim,
    const Matrix *embedding_table,
    Matrix *X
);

// 行列 src の指定された 行 row のデータを 1行の行列 dest にコピーする
void extract_row(const Matrix *src, int row, Matrix *dest);

// スケール
void scale_matrix(Matrix *m, float scalar);

// 順伝播
void softmax_row(Matrix *m, int row);
void relu(Matrix *m);
void layer_norm(const Matrix *X, Matrix  *Y, float *gamma, float *beta);
void residual_add(const Matrix *X_in, const Matrix *X_layer, Matrix *Y);

// 逆伝播
void backward_softmax(const Matrix *dA, const Matrix *A, Matrix *dS);
void backward_relu(const Matrix *dOut, const Matrix *Out, Matrix *dIn);

#endif // OPS_H