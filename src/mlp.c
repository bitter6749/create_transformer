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