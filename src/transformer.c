#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "transformer.h"

void forward_transformer(
    SimpleTransformer *model, 
    const int *input_ids, 
    Matrix *output_probabilties
) {
  // 一時的なワークスペース (ヒープに確保)
  // X: 入力文字をベクトル化したもの [SEQ_LEN x EMBED_DIM]
  //
  //      { x0,0 x0,1 ... x0,15 }  <- 一文字目 'c' のベクトル
  //      { x1,0 x1,1 ... x1,15 }  <- 二文字目 'a' のベクトル
  // X =  |                     |
  //      { x2,0 x2,1 ... x2,15 }  <- 三文字目 't' のベクトル
  //      { x3,0 x3,1 ... x3,15 }  <- 四文字目 ' ' のベクトル
  Matrix X = create_matrix(SEQ_LEN, EMBED_DIM);

  // Q, K, V ベクトル
  // Q: クエリ
  // K: キュー
  // V: バリュー
  Matrix Q = create_matrix(SEQ_LEN, EMBED_DIM);
  Matrix K = create_matrix(SEQ_LEN, EMBED_DIM);
  Matrix V = create_matrix(SEQ_LEN, EMBED_DIM);

  Matrix A_scores = create_matrix(SEQ_LEN, SEQ_LEN);
  Matrix Z = create_matrix(SEQ_LEN, EMBED_DIM);

  // ===============================================
  // === step1: Embedding (文字IDをベクトルに変換) ===
  // ===============================================
  for (int t = 0; t < SEQ_LEN; t++) {
    int id = input_ids[t];
    for (int d = 0; d < EMBED_DIM; d++) {
      X.data[t * EMBED_DIM + d] = model->token_embedding.data[id * EMBED_DIM + d];
    }
  }

  // ==========================================
  // === step2: Q, K, V の計算 (行列の掛け算) ===
  // ==========================================
  mat_mul(&X, &model->W_q, &Q);
  mat_mul(&X, &model->W_k, &K);
  mat_mul(&X, &model->W_v, &V);

  // =========================================
  // === step3: Attention (注意機構) の計算 ===
  // =========================================

  //        { A0,0 A0,1 A0,2 A0,3 } <- 'c' から見た各文字への注目度
  //        { A1,0 A1,1 A1,2 A1,3 } <- 'a' から見た各文字への注目度
  //  A =   |                     |
  //        { A2,0 A2,1 A2,2 A2,3 } <- 't' から見た各文字への注目度
  //        { A3,0 A3,1 A3,2 A3,3 } <- ' ' から見た各文字への注目度

  // Q ・ K^T = A (K の転置行列)
  mat_mul_b_trans(&Q, &K, &A_scores);

  // スケール調整とSoftmax
  for (int i = 0; i < SEQ_LEN; i++) {
    for (int j = 0; j < SEQ_LEN; j++) {
    // Q, K は4行16列の行列
    // 4行16列 x 16行4列 = 4行4列 
      A_scores.data[i * SEQ_LEN + j] /= sqrtf(EMBED_DIM); // スケール調整
    }
    // softmaxをかけて確率 (重み) にする
    softmax(&A_scores.data[i * SEQ_LEN], SEQ_LEN);
  }

  // 計算した関係性 (A_scores) を使って、V (価値)のベクトルをブレンドする
  // Z: Attention の出力 [SEQ_LEN x EMBED_DIM]
  //
  //  4行16列 = 4行4列 x 4行16列 
  //  A ・ V = Z
  mat_mul(&A_scores, &V, &Z);
  

  // ===================================================
  // === step4: 出力層 (最後の文字の予想結果だけを使う) ===
  // ===================================================

  // 今回は「最後の文字 (t = SEQ_LEN - 1)」の次の文字を予測したいので、Zの最後の行だけを計算
  //
  // 1行27列 = 1行16列 x 16列27行
  // output = last_Z ・ W_out^T (W_out の転置行列)
  //
  // W_out^T との掛け算を mat_mul_b_trans でスマートに行うために、1行のMatrixとして扱います
  Matrix last_z = { &Z.data[(SEQ_LEN - 1) * EMBED_DIM], 1, EMBED_DIM };

  // last_z ・ W_out^T = output
  mat_mul_b_trans(&last_z, &model->W_out, output_probabilties);
  softmax(output_probabilties->data, VOCAB_SIZE); // 最終的な文字の確立分布にする


  // ワークスペースの解放
  free_matrix(&X);
  free_matrix(&Q);
  free_matrix(&K);
  free_matrix(&V);
  free_matrix(&A_scores);
  free_matrix(&Z);
}


// 出力層の逆伝播関数
void backward_ouput(
  SimpleTransformer *model, 
  const Matrix *output_probabilities, // 順伝播の出力 [VOCAB_SIZE]
  int target_id,                     // 正解の文字ID
  const Matrix *Z,                    // 順伝播の途中の行列 [SEQ_LEN x EMBED_DIM]
  Matrix *dW_out,                     // W_out の勾配 [VOCAB_SIZE x EMBED_DIM]
  Matrix *dZ                           // Zへの誤差の逆流 [SEQ_LEN x EMBED_DIM]
) {
  // 1. 生スコアの誤差を計算
  float d_logits[VOCAB_SIZE];
  for (int i = 0; i < VOCAB_SIZE; i++) {
    d_logits[i] = output_probabilities->data[i];
  }
  d_logits[target_id] -= 1.0f;

  // 2. W_out の修正メーターを計算
  const float *last_z = &Z->data[(SEQ_LEN - 1) * EMBED_DIM];  // 4文字目のベクトルを指す

  for (int i = 0; i < VOCAB_SIZE; i++) {
    for (int d = 0; d < EMBED_DIM; d++) {
      // 誤差 x 通ってきた値 (last_z) を勾配に蓄積
      dW_out->data[i * EMBED_DIM + d] += d_logits[i] * last_z[d];
    }
  }

  // 3. 行列 Z の最後の行 (4行目) への誤差を計算して、上流へ引き継ぐ
  float *dz_last = &dZ->data[(SEQ_LEN - 1) * EMBED_DIM];
  for (int i = 0; i < VOCAB_SIZE; i++) {
    for (int d = 0; d < EMBED_DIM; d++) {
      dz_last[d] += d_logits[i] * model->W_out.data[i * EMBED_DIM + d];
    }
  }
}

// 相性表の誤差dA
// dA = dZ ・ V^T (Vの転置行列)
// [4 x 4] = [4 x 16] x [16 x 4]
//
// 価値の誤差dV
// dV = A^T (Aの転置行列) ・ dZ
// [4 x 16] = [4 x 4] x [4 x 16]
void backward_attention(
  const Matrix *dZ,
  const Matrix *A_scores,
  const Matrix *V,
  Matrix *dA,
  Matrix *dV
) {
  // dZ ・ V^T = dA の計算
  mat_mul_b_trans(dZ, V, dA);

  // A^T ・ dZ = dAの計算
  mat_mul_a_trans(A_scores, dZ, dV);
  
}

// ソフトマックス関数
void softmax(float *x, int size) {
  float max_val = x[0];
  for (int i = 1; i < size; i++) {
    if (x[i] > max_val) max_val = x[i];
  }

  float sum = 0.0f;
  for (int i = 0; i < size; i++) {
    x[i] = expf(x[i] - max_val);
    sum += x[i];
  }
  for (int i = 0; i < size; i++) {
    x[i] /= sum;
  }
}

// 逆伝播: Attention層のQ, K, V
void backward_attention_qkv(
    const Matrix *dA,
    const Matrix *A_prob,
    const Matrix *Q,
    const Matrix *K,
    Matrix *dQ,
    Matrix *dK
) {
    int seq_len = dA->rows;
    int embed_dim = Q->cols;

    // Softmax をかける前の生スコアへの誤差を入れるバッファ [seq_len x seq_len]
    Matrix dS = create_matrix(seq_len, seq_len);

    // =====================================================
    // Step 1 & 2: Softmaxの微分 と スケール調整 (1 / sqrt(D))
    // =====================================================
    for (int i = 0; i < seq_len; i++) {
        float sum_da_a = 0.0f;
        // 1. 行ごとの内積 (dA ・ A_prob) を計算
        for (int k = 0; k < seq_len; k++) {
            int idx = i * seq_len + k;
            sum_da_a += dA->data[idx] * A_prob->data[idx];
        }

        // 2. Softmax の微分公式を適用し、同時に√embed_dim で割る(スケール)
        for (int j = 0; j < seq_len; j++) {
            int idx = i * seq_len + j;
            float ds_val = A_prob->data[idx] * (dA->data[idx] - sum_da_a);

            // スケール調整の逆伝播
            dS.data[idx] = ds_val / sqrtf(embed_dim);
        }
    }

    // ===================================================
    // Step 3: 行列積の逆伝播 (dQ = dS ・ K, dK = dS^T ・ Q)
    // ===================================================

    // dQ = dS ・ K
    // [seq_len x seq_len] * [seq_len x embed_dim]
    // -> [seq_len x embed_dim]
    mat_mul(&dS, K, dQ);

    // dK = dS^T ・ Q
    // [seq_len x seq_len]^T * [seq_len x embed_dim]
    // -> [seq_len x embed_dim]
    mat_mul_a_trans(&dS, Q, dK);

    // ワークスペースの解放
    free_matrix(&dS);
}