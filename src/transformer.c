#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "transformer.h"
#include "attention.h"
#include "mlp.h"
#include "ops.h"

// ====================
// === 出力層の順伝播 ===
// ====================

// 入力特徴量 X_features の最後のトークンから、次に来る文字の確立分布を計算する
void forward_lm_head(
  const Matrix *X_features, // [SEQ_LEN x EMBED_DIM] (Attention または MLP の出力)
  const Matrix *W_out,      // [VOCAB_SIZE x EMBED_DIM]
  Matrix *output_probabilities  // [1 x VOCAB_SIZE] (最終出力)
) {
  int seq_len = X_features->rows;
  int embed_dim = X_features->cols;

  // 1. ワークスペースの確保
  Matrix last_z = create_matrix(1, embed_dim);

  // 2. 最後のトークン (SEQ_LEN - 1) の特徴量ベクトルを抽出
  extract_row(X_features, seq_len - 1, &last_z);
  
  // 3. 全結合層: 特徴量表現から生スコア (ロジット) へ変換 [1 x VOCAB_SIZE]
  // 1行27列 = 1行16列 x 16列27行
  // output_probabilities = last_z ・ W_out
  mat_mul(&last_z, W_out, output_probabilities);

  // 4. 確率変換: 生スコアを softmax で、0.0 ~ 1.0 の確立分布にする
  // output_probabilities は[1 x 27]で0番目の行しかない
  softmax_row(output_probabilities, 0);
  
  // ワークスペースの解放
  free_matrix(&last_z);
}

// ========================
// === モデル全体の順伝播 ===
// ========================
void forward_transformer(
    SimpleTransformer *model, 
    const int *input_ids, 
    Matrix *output_probabilities
) {
  // ==================================================
  // === STEP 1. 各種一時ワークスペース (ヒープに確保) ===
  // ==================================================
  // X: 入力文字をベクトル化したもの [SEQ_LEN x EMBED_DIM]
  //
  //      { x0,0 x0,1 ... x0,15 }  <- 一文字目 'c' のベクトル
  //      { x1,0 x1,1 ... x1,15 }  <- 二文字目 'a' のベクトル
  // X =  |                     |
  //      { x2,0 x2,1 ... x2,15 }  <- 三文字目 't' のベクトル
  //      { x3,0 x3,1 ... x3,15 }  <- 四文字目 ' ' のベクトル
  Matrix X = create_matrix(SEQ_LEN, EMBED_DIM);

  // Q, K, V ベクトル
  Matrix Q = create_matrix(SEQ_LEN, EMBED_DIM); // Q: クエリ
  Matrix K = create_matrix(SEQ_LEN, EMBED_DIM); // K: キュー
  Matrix V = create_matrix(SEQ_LEN, EMBED_DIM); // V: バリュー

  Matrix A_prob = create_matrix(SEQ_LEN, SEQ_LEN);
  Matrix Z = create_matrix(SEQ_LEN, EMBED_DIM);

  // MLP層用の一時ワークスペースを追加
  // Y_mlp: MLP の最終出力 [SEQ_LEN x EMBED_DIM]
  Matrix Y_mlp = create_matrix(SEQ_LEN, EMBED_DIM);

  // ================================================
  // === STEP 2: Embedding (文字IDをベクトルに変換) ===
  // ================================================
  // 埋め込み層 (Embedding Lookup)
  forward_embedding(
    input_ids, 
    SEQ_LEN, EMBED_DIM, 
    &model->token_embedding, 
    &X
  );

  // ==============================================
  // === STEP 3: Attention (注意機構) の 呼び出し ===
  // ===============================================
  //        { A0,0 A0,1 A0,2 A0,3 } <- 'c' から見た各文字への注目度
  //        { A1,0 A1,1 A1,2 A1,3 } <- 'a' から見た各文字への注目度
  //  A =   |                     |
  //        { A2,0 A2,1 A2,2 A2,3 } <- 't' から見た各文字への注目度
  //        { A3,0 A3,1 A3,2 A3,3 } <- ' ' から見た各文字への注目度
  // 3. 注意機構層 (Attention Layer)
  forward_attention(
    &X,
    &model->W_q, &model->W_k, &model->W_v,
    &Q, &K, &V,
    &A_prob,
    &Z
  );

  // ======================================================
  // === STEP 4: MLP(多層パーセプトロン)層の呼び出しを追加 ===
  // ======================================================
  // Attention の出力 Z を入力とし、結果を Y_mlp に格納する
  // 4. 前向型全結合層 (MLP / Feed-Forward Layer)
  forward_mlp(
    &Z,
    &model->W1,
    &model->b1,
    &model->W2,
    &model->b2,
    &model->H,
    &Y_mlp
  );

  // ====================================================
  // === STEP 5: 出力層 (最後の文字の予想結果だけを使う) ===
  // ====================================================
  // 5. 出力層 (LM Head)
  forward_lm_head(&Y_mlp, &model->W_out, output_probabilities);

  // ==================================
  // === STEP 5: ワークスペースの解放 ===
  // ==================================
  free_matrix(&X);
  free_matrix(&Q);
  free_matrix(&K);
  free_matrix(&V);
  free_matrix(&A_prob);
  free_matrix(&Z);
  free_matrix(&Y_mlp);
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