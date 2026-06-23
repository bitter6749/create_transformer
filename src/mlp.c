#include "mlp.h"
#include "ops.h"

// 順伝播: Y = Activation(X ・ W1 + b1) ・ W2 + b2
void forward_mlp(
  const Matrix *X, 
  const Matrix *W1,
  const Matrix *b1,
  const Matrix *W2,
  const Matrix *b2,
  Matrix *H, // 1層目の出力 (活性化前/後, 逆伝播用に保持)
  Matrix *Y  // MLP最終出力
) { 
  // =====================================
  // === 1. 1層目の線形変換: H = X ・ W1 ===
  // =====================================
  mat_mul(X, W1, H);

  // バイアス b1 の加算
  for (int i = 0; i < H->rows; i++) {
    for (int j = 0; j < H->cols; j++) {
      H->data[i * H->cols + j] += b1->data[j];
    }
  }
  
  // =========================
  // === 2. 活性化関数の適用 ===
  // =========================
  relu(H);

  // ======================================
  // === 3. 2層目の線形変換: Y = H ・ W2 ===
  // ======================================
  mat_mul(H, W2, Y);

  // バイアス b2 の加算
  for (int i = 0; i < Y->rows; i++) {
    for (int j = 0; j < Y->cols; j++) {
      Y->data[i * Y->cols + j] += b2->data[j];
    }
  }

}

void backward_mlp(
  const Matrix *dY, // 上流からの誤差 [SEQ_LEN x EMBED_DIM] 
  const Matrix *Z,  // 順伝播時の入力 [SEQ_LEN x EMBED_DIM]
  const Matrix *W1, // [EMBED_DIM x MLP_HIDDEN_DIM]
  const Matrix *H,  // 順伝播時のReLU適用後の出力 [SEQ_LEN x MLP_HIDDEN_DIM]
  const Matrix *W2, // [MLP_HIDDEN_DIM x EMBED_DIM]
  Matrix *dZ,       // 下流 (Attention) への誤差 [SEQ_LEN x EMBED_DIM]
  Matrix *dW1, Matrix *db1,
  Matrix *dW2, Matrix *db2
) {
  int seq_len = dY->rows;
  int embed_dim = dY->cols;

  // =====================================
  // === STEP 1: 2層目の線形変換の逆伝播 ===
  // =====================================
  // 1. バイアス b2 の勾配 (dY の各列の総和)
  for (int j = 0; j < embed_dim; j++) {
    float sum = 0.0f;
    for (int i = 0; i < seq_len; i++) {
      sum += dY->data[i * embed_dim + j];
    }
    db2->data[j] = sum;
  }

  // 2. 重み W2 の勾配: dW2 = H^T ・ dY
  // [MLP_HIDDEN_DIM x SEQ_LEN] * [SEQ_LEN x EMBED_DIM] = [MLP_HIDDEN_DIM x EMBED_DIM]
  mat_mul_a_trans(H, dY, dW2);

  // 3. 活性化関数 (ReLU) 直後への誤差 dH_out = dY ・ W2^T
  // [SEQ_LEN x EMBED_DIM] * [EMBED_DIM x MLP_HIDDEN_DIM] = [SEQ_LEN x MLP_HIDDEN_DIM]
  Matrix dH_out = create_matrix(seq_len, MLP_HIDDEN_DIM);
  mat_mul_b_trans(dY, W2, &dH_out);

  // ======================================
  // === STEP 2: 活性化関数(ReLU)の逆伝播 ===
  // ======================================
  Matrix dH_in = create_matrix(seq_len, MLP_HIDDEN_DIM);
  backward_relu(&dH_out, H, &dH_in);

  // =====================================
  // === STEP 3: 1層目の線形変換の逆伝播 ===
  // =====================================
  // 1. バイアス b1 の勾配 (dH_in の各列の総和)
  for (int j = 0; j < MLP_HIDDEN_DIM; j++) {
    float sum = 0.0f;
    for (int i = 0; i < seq_len; i++) {
      sum += dH_in.data[i * MLP_HIDDEN_DIM + j];
    }
    db1->data[j] = sum;
  }

  // 2. 重み W1 の勾配: dW1 = Z^T ・ dH_in
  // [EMBED_DIM x SEQ_LEN] * [SEQ_LEN x MLP_HIDDEN_DIM] = [EMBED_DIM x MLP_HIDDEN_DIM]
  mat_mul_a_trans(Z, &dH_in, dW1);

  // 3. 入力 Z への誤差(下流への誤差): dZ = dH_in ・ W1^T
  // [SEQ_LEN x MLP_HIDDEN_DIM] * [MLP_HIDDEN_DIM x EMBED_DIM] = [SEQ_LEN x EMBED_DIM]
  mat_mul_b_trans(&dH_in, W1, dZ);

  // ワークスペースの解放
  free_matrix(&dH_out);
  free_matrix(&dH_in);
}