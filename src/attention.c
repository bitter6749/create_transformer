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
  float scale = 1.0f / sqrtf(embed_dim);

  scale_matrix(A_prob, scale); // 全要素を一括で 1/sqrt(d_k) 倍する

  for (int i = 0; i < seq_len; i++) {
    softmax_row(A_prob, i);
  }

  // 4. V (A ・ V = Z)
  mat_mul(A_prob, V, Z);
}

void backward_attention(
  const Matrix *dZ,       // 上流 (MLP) からの誤差 [4 x 16]
  const Matrix *A_prob,   // 順伝播時のSoftmax出力 [4 x 4]

  const Matrix *Q,        // 順伝播時のQ [4 x 16]
  const Matrix *K,        // 順伝播時のK [4 x 16]
  const Matrix *V,        // 順伝播時のV [4 x 16]
  const Matrix *X,        // 順伝播時の入力特徴量 [4 x 16]

  const Matrix *W_q,      // 重みQ [16 x 16]
  const Matrix *W_k,      // 重みK [16 x 16]
  const Matrix *W_v,      // 重みV [16 x 16]

  Matrix *dX,             // 下流 (Embedding) への誤差 [4 x 16]
  Matrix *dW_q,           // 重みQの勾配 [16 x 16]
  Matrix *dW_k,           // 重みKの勾配 [16 x 16]
  Matrix *dW_v            // 重みVの勾配 [16 x 16]
) {
  int seq_len = dZ->rows;
  int embed_dim = Q->cols;

  // 各自の中間勾配バッファを確保
  Matrix dA = create_matrix(seq_len, seq_len); // [4 x 4]
  Matrix dS = create_matrix(seq_len, seq_len); // [4 x 4] Softmax 前の生スコア勾配

  Matrix dQ = create_matrix(seq_len, embed_dim); // [4 x 16]
  Matrix dK = create_matrix(seq_len, embed_dim); // [4 x 16]
  Matrix dV = create_matrix(seq_len, embed_dim); // [4 x 16]

  Matrix dX_buf = create_matrix(seq_len, embed_dim);  // [4 x 16] 一時的なバッファ

  // ==================================
  // === Step 1: Z = A ・ V の逆伝播 ===
  // ==================================
  // dA = dZ ・ V^T
  // [4 x 16] * [16 x 4] = [4 x 4]
  mat_mul_b_trans(dZ, V, &dA);   
  // dV = A^T ・ dZ
  // [4 x 4] * [4 x 16] = [4 x 16]
  mat_mul_a_trans(A_prob, dZ, &dV); 

  // ============================================
  // === Step 2: Softmax & スケール調整の逆伝播 ===
  // ============================================
  backward_softmax(&dA, A_prob, &dS);


  // Softmax微分の結果をスケール調整
  float scale = 1.0f / sqrtf(embed_dim);
  scale_matrix(&dS, scale);

  // ====================================
  // === Step 3: A = Q ・ K^T の逆伝播 ===
  // ====================================
  mat_mul(&dS, K, &dQ); // dQ = dS ・ K
  mat_mul_a_trans(&dS, Q, &dK); // dK = dS^T ・ Q

  // =================================================================
  // === Step 4: Q, K, V = X ・ W の逆伝播 (重みの勾配と入力への勾配) ===
  // =================================================================

  // -----------------------------------
  // --- 1. 重みパラメーターの勾配計算 ---
  // -----------------------------------

  // Q = X ・ W_q => dW_q = X^T ・ dQ
  mat_mul_a_trans(X, &dQ, dW_q);
  // K = X ・ W_k => dW_q = X^T ・ dQ
  mat_mul_a_trans(X, &dK, dW_k);
  // V = X ・ W_v => dW_v = X^T ・ dV
  mat_mul_a_trans(X, &dV, dW_v);

  // ----------------------------------------
  // --- 2. 各ルートから戻る X への誤差計算 ---
  // ----------------------------------------
  
  mat_mul_b_trans(&dQ, W_q, dX); // dX_q = dQ ・ W_q^T => dXに格納
  mat_mul_b_trans(&dK, W_k, &dX_buf); // dX_k = dK ・ W_k^T => dX_buf に格納
  // dX (dX_q) と　dX_buf (dX_k) の誤差を加算
  mat_add(dX, &dX_buf, dX); 
  mat_mul_b_trans(&dV, W_v, &dX_buf); // dX_v += dV ・ W_v^T => dX_buf に格納
  // dX (dX_q + dX_k) と dX_buf (dX_v) の誤差を加算
  mat_add(dX, &dX_buf, dX);
    
  // 一時バッファの解放
  free_matrix(&dA);
  free_matrix(&dQ);
  free_matrix(&dK);
  free_matrix(&dV);
  free_matrix(&dS);
  free_matrix(&dX_buf);
}