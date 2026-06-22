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

// ====================
// === 出力層の逆伝播 ===
// ====================
// 最終出力の確率分布と正解ID (target_id) から誤差を計算し、dY_mlp と dW_out を求める
void backward_lm_head(
  const Matrix *output_probabilities, // [1 x VOCAB_SIZE]
  int target_id,
  const Matrix *Y_mlp,                // [SEQ_LEN x EMBED_DIM]
  const Matrix *W_out,                // [EMBED_DIM x VOCAB_SIZE]
  Matrix *dY_mlp,                     // [SEQ_LEN x EMBED_DIM] (下流へ流す誤差)
  Matrix *dW_out                      // [EMBED_DIM x VOCAB_SIZE] (重みの勾配)
) {
  int seq_len = Y_mlp->rows;
  int embed_dim = Y_mlp->cols;

  // 1. Softmax + Cross Entropy の微分を計算
  // Softmax と Cross Entropy の損失関数の微分は、
  // (予測確率 - 正解ラベル) になる
  // 正解ラベルは確率の最大値 1 になる
  Matrix dOutput_Linear = create_matrix(1, VOCAB_SIZE);
  for (int i = 0; i < VOCAB_SIZE; i++) {
    dOutput_Linear.data[i] = output_probabilities->data[i];
  }
  dOutput_Linear.data[target_id] -= 1.0f; // (予測確率 - 正解ラベル )

  // 2. 順伝播の時と同じく、最後のトークン(SEQ_LEN - 1)の特徴量を抽出
  Matrix last_z = create_matrix(1, embed_dim);
  extract_row(Y_mlp, seq_len - 1, &last_z);

  // 3. 重み W_out への勾配を計算: 
  // dW_out = last_z^T ・ dOutput_Linear
  // [EMBED_DIM x 1] * [1 x VOCAB_SIZE] = [EMBED_DIM x VOCAB_SIZE]
  mat_mul_a_trans(&last_z, &dOutput_Linear, dW_out);

  // 4. 特徴量 (last_z) への誤差を計算: 
  // dLast_z = dOutput_Linear ・ W_out^T
  // [1 x VOCAB_SIZE] * [VOCAB_SIZE x EMBED_DIM] = [1 x EMBED_DIM]
  Matrix dLast_z = create_matrix(1, embed_dim);
  mat_mul_b_trans(&dOutput_Linear, W_out, &dLast_z);

  // 5. 全体の誤差行列 dY_mlp の最後の行に、計算した誤差を書き戻す
  for (int j = 0; j < embed_dim; j++) {
    dY_mlp->data[(seq_len - 1) * embed_dim + j] = dLast_z.data[j];
  }

  // ワークスペースの解放
  free_matrix(&dOutput_Linear);
  free_matrix(&last_z);
  free_matrix(&dLast_z);
}

// ========================
// === モデル全体の逆伝播 ===
// ========================
void backward_transformer(
  SimpleTransformer *model,
  const int *input_ids,
  int target_id,

  // 各層の重み・バイアスの勾配
  Matrix *dW_out,
  Matrix *dW1, Matrix *db1, Matrix *dW2, Matrix *db2, 
  Matrix *dW_q, Matrix *dW_k, Matrix *dW_v
) {
  // ====================================================
  // 逆伝播の計算には、順伝播時の「途中経過の行列の値」が必要
  // forwardを走らせて、各種中間バッファを揃える
  //（X, Q, K, V, A_prob, Z, Y_mlp, output_probabilities）
  // ====================================================
  Matrix X        = create_matrix(SEQ_LEN, EMBED_DIM);
  Matrix Q        = create_matrix(SEQ_LEN, EMBED_DIM);
  Matrix K        = create_matrix(SEQ_LEN, EMBED_DIM);
  Matrix V        = create_matrix(SEQ_LEN, EMBED_DIM);
  Matrix A_prob   = create_matrix(SEQ_LEN, SEQ_LEN);
  Matrix Z        = create_matrix(SEQ_LEN, EMBED_DIM);
  Matrix Y_mlp    = create_matrix(SEQ_LEN, EMBED_DIM);
  Matrix output_probabilities    = create_matrix(1, VOCAB_SIZE);

  // =============================================
  // 1. 順伝播を計算して中間データをバッファに格納する
  // =============================================
  forward_embedding(input_ids, SEQ_LEN, EMBED_DIM, &model->token_embedding, &X);
  forward_attention(&X, &model->W_q, &model->W_k, &model->W_v, &Q, &K, &V, &A_prob, &Z);
  forward_mlp(&Z, &model->W1, &model->b1, &model->W2, &model->b2, &model->H, &Y_mlp);
  forward_lm_head(&Y_mlp, &model->W_out, &output_probabilities);

  // =======================================
  // 2. 実際に逆伝播を行う ( 出力層 -> 入力層 )
  // ========================================

  // 各層を流れる誤差(勾配)を受け渡すためのバッファ
  Matrix dY_mlp = create_matrix(SEQ_LEN, EMBED_DIM); // 出力層からMLPへ流れる誤差
  Matrix dZ     = create_matrix(SEQ_LEN, EMBED_DIM); // MLPからAttentionへ流れる誤差
  Matrix dX     = create_matrix(SEQ_LEN, EMBED_DIM); // AttentionからEmbeddingへ流れる誤差

  // -----------------------------
  // --- STEP 1: 出力層の逆伝播 ---
  // -----------------------------
  // 最終出力の確率分布と正解ID (target_id) から誤差を計算し、dY_mlp と dW_out を求める
  backward_lm_head(&output_probabilities, target_id, &Y_mlp, &model->W_out, &Y_mlp, dW_out);

  // ----------------------------------------------
  // --- STEP 2: MLP(多層パーセプトロン)層の逆伝播 ---
  // -----------------------------------------------
  // 出力層から来た誤差 dY_mlp を元に、 W1, b1, W2, b2 の勾配を計算し、下流の dZ へ流す
  backward_mlp(&Y_mlp, &Z, &dZ, &model->H, &model->W1, &model->W2, dW1, db1, dW2, db2);

  // ---------------------------------------------
  // --- STEP 3: Attention (注意機構) 層の逆伝播 ---
  // ----------------------------------------------
  // MLPから来た誤差 dZ を元に、W_q, W_k, W_v の勾配を計算し、最下流の dX へ流す
  backward_attention(&dZ, &A_prob, &Q, &K, &V, &X, &model->W_q, &model->W_k, &model->W_v, &dX, dW_q, dW_k, dW_v);

  // -------------------------------------------------
  // --- STEP 4: Embedding (単語埋め込み) 層の逆伝播 ---
  // -------------------------------------------------
  // 最下流に届いた dX を使って、今回入力された文字の埋め込みパラメーターを修正する
  backward_embedding(&dX, input_ids, SEQ_LEN, EMBED_DIM, &model->token_embedding);

  // =======================
  // 3. バッファ(メモリ)の解放
  // =======================
  free_matrix(&X);
  free_matrix(&Q);
  free_matrix(&K);
  free_matrix(&V);
  free_matrix(&A_prob);
  free_matrix(&Z);
  free_matrix(&Y_mlp);
  free_matrix(&output_probabilities);
  free_matrix(&dY_mlp);
  free_matrix(&dZ);
  free_matrix(&dX);
}