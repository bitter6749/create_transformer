#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "transformer.h"
#include "attention.h"
#include "mlp.h"
#include "ops.h"

// ========================
// === モデル全体の順伝播 ===
// ========================
void forward_transformer(
    SimpleTransformer *model, 
    const int *input_ids, 
    Matrix *output_probabilities
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

  Matrix A_prob = create_matrix(SEQ_LEN, SEQ_LEN);
  Matrix Z = create_matrix(SEQ_LEN, EMBED_DIM);
  Matrix Output_Liner = create_matrix(1, VOCAB_SIZE);

  // ===============================================
  // === step1: Embedding (文字IDをベクトルに変換) ===
  // ===============================================
  for (int t = 0; t < SEQ_LEN; t++) {
    int id = input_ids[t];
    for (int d = 0; d < EMBED_DIM; d++) {
      X.data[t * EMBED_DIM + d] = model->token_embedding.data[id * EMBED_DIM + d];
    }
  }

  // =============================================
  // === step2: Attention (注意機構) の 呼び出し ===
  // ==============================================

  //        { A0,0 A0,1 A0,2 A0,3 } <- 'c' から見た各文字への注目度
  //        { A1,0 A1,1 A1,2 A1,3 } <- 'a' から見た各文字への注目度
  //  A =   |                     |
  //        { A2,0 A2,1 A2,2 A2,3 } <- 't' から見た各文字への注目度
  //        { A3,0 A3,1 A3,2 A3,3 } <- ' ' から見た各文字への注目度

  forward_attention(
    &X,
    &model->W_q, &model->W_k, &model->W_v,
    &Q, &K, &V,
    &A_prob,
    &Z
  );

  // ===================================================
  // === step4: 出力層 (最後の文字の予想結果だけを使う) ===
  // ===================================================

  // 今回は「最後の文字 (i = SEQ_LEN - 1)」の次の文字を予測したいので、Zの最後の行だけを計算
  // 1行27列 = 1行16列 x 16列27行
  // Output_Linear = last_Z ・ W_out
  Matrix last_z = create_matrix(1, EMBED_DIM);
  for (int i = 0; i < EMBED_DIM; i++) {
    last_z.data[i] = Z.data[(SEQ_LEN - 1) * EMBED_DIM + i];
  }

  // Output_Linear = last_z ・ W_out
  mat_mul(&last_z, &model->W_out, &Output_Liner);

  // 確率変換
  for (int i = 0; i < VOCAB_SIZE; i++) {
    output_probabilities->data[i] = Output_Liner.data[i];
  }
  // output_probabilities は[1x27]で0番目の行しかない
  softmax_row(output_probabilities, 0);

  // ワークスペースの解放
  free_matrix(&X);
  free_matrix(&Q);
  free_matrix(&K);
  free_matrix(&V);
  free_matrix(&A_prob);
  free_matrix(&Z);
  free_matrix(&last_z);
  free_matrix(&Output_Liner);
}

// ========================
// === モデル全体の逆伝播 ===
// ========================
void backward_transformer(
  SimpleTransformer *model,
  const int *input_ids,
  int target_id,
  // 各層の重みの勾配
  Matrix *dW_out,
  Matrix *dW_q,
  Matrix *dW_k,
  Matrix *dW_v
) {
}