#include <math.h>
#include "attention.h"
#include "ops.h"

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
) {
  // 1. Q, K, V の計算
  mat_mul(X, W_q, Q);
  mat_mul(X, W_k, K);
  mat_mul(X, W_v, V);

  // 2. 注意スコアの計算 (Q ・ K^T = A)
  mat_mul_b_trans(Q, K, A_prob);

  // 3. スケール調整とSoftmax
  int seq_len = X->rows;
  int embed_dim = Q->cols;
  float scale = sqrtf(embed_dim);

  for (int i = 0; i < seq_len; i++) {
    for (int j = 0; j < seq_len; j++) {
      A_prob->data[i * seq_len + j] /= scale;
    }
    softmax_row(A_prob, i);
  }

  // 4. V (A ・ V = Z)
  mat_mul(A_prob, V, Z);
}

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
) {
  int seq_len = dZ->rows;
  int embed_dim = Q->cols;

  // 各自の中間勾配バッファを確保
  Matrix dA = create_matrix(seq_len, seq_len);
  Matrix dS = create_matrix(seq_len, seq_len); // Softmax 前の生スコア勾配

  Matrix dV = create_matrix(seq_len, embed_dim);
  Matrix dQ = create_matrix(seq_len, embed_dim);
  Matrix dK = create_matrix(seq_len, embed_dim);

  // ==============================
  // === Step 1: Z = A ・ V の逆伝播
  // ==============================
  mat_mul_b_trans(dZ, V, &dA);  // dA = dZ ・ V^T
  mat_mul_a_trans(A_prob, dZ, &dV);  // dV = A^T ・ dZ

  // ============================================
  // === Step 2: Softmax & スケール調整の逆伝播 ===
  // ============================================
  for (int i = 0; i < seq_len; i++) {
    float sum_da_a = 0.0f;
    for (int k = 0; k < seq_len; k++) {
      int idx = i *seq_len + k;
      sum_da_a += dA.data[idx] * A_prob->data[idx];
    }
    for (int j = 0; j < seq_len; j++) {
      int idx = i * seq_len + j;
      float ds_val = A_prob->data[idx] * (dA.data[idx] - sum_da_a);
      dS.data[idx] = ds_val / sqrtf(embed_dim);
    }
  }

  // ====================================
  // === Step 3: A = Q ・ K^T の逆伝播 ===
  // ====================================
  mat_mul(&dS, K, &dQ); // dQ = dS ・ K
  mat_mul_a_trans(&dS, Q, &dK); // dK = dS^T ・ Q

  // =================================================================
  // === Step 4: Q, K, V = X ・ W の逆伝播 (重みの勾配と入力への勾配) ===
  // =================================================================
  // Q = X ・ W_q => dW_q = X^T ・ dQ, dX_Q = dQ ・ W_q^T
  mat_mul_a_trans(X, &dQ, dW_q);
  mat_mul_b_trans(&dQ, W_q, dX); // dX にまずQからの誤差を格納

  // K = X ・ W_k => dW_q = X^T ・ dQ, dX_q = dQ ・ W_q^T
  mat_mul_a_trans(X, &dK, dW_k);
  // dX += dK ・ W_k^T
  Matrix dX_k = create_matrix(seq_len, embed_dim);
  mat_mul_b_trans(&dK, W_k, &dX_k);
  for (int i = 0; i < seq_len * embed_dim; i++) {
    dX->data[i] += dX_k.data[i];
  }

  // V = X ・ W_v => dW_v = X^T ・ dV, dX_v = dV ・ W_v^T
  mat_mul_a_trans(X, &dV, dW_v);
  Matrix dX_v = create_matrix(seq_len, embed_dim);
  mat_mul_b_trans(&dV, W_v, &dX_v);
  for (int i = 0; i < seq_len * embed_dim; i++) {
    dX->data[i] += dX_v.data[i];
  }
  
  // 一時バッファの解放
  free_matrix(&dA);
  free_matrix(&dQ);
  free_matrix(&dK);
  free_matrix(&dV);
  free_matrix(&dS);
  free_matrix(&dX_k);
  free_matrix(&dX_v);
}