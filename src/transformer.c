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
  
  // X: 現在の状態を保持するメインストリーム行列 [SEQ_LEN x EMBED_DIM]
  Matrix X = create_matrix(SEQ_LEN, EMBED_DIM);
  // 埋め込み層 (Embedding Lookup)
  forward_embedding(
    input_ids, 
    SEQ_LEN, EMBED_DIM, 
    &model->token_embedding, 
    &X
  );
  // 位置エンコーディング
  apply_positional_encoding(&X);

  // 各層の計算用一時バッファ
  Matrix X_norm = create_matrix(SEQ_LEN, EMBED_DIM);
  // Q, K, V ベクトル
  Matrix Q = create_matrix(SEQ_LEN, EMBED_DIM); // Q: クエリ
  Matrix K = create_matrix(SEQ_LEN, EMBED_DIM); // K: キュー
  Matrix V = create_matrix(SEQ_LEN, EMBED_DIM); // V: バリュー

  Matrix A_prob = create_matrix(SEQ_LEN, SEQ_LEN);
  Matrix Z = create_matrix(SEQ_LEN, EMBED_DIM);

  // MLP層用の一時ワークスペースを追加
  // Y_mlp: MLP の最終出力 [SEQ_LEN x EMBED_DIM]
  Matrix Y_mlp = create_matrix(SEQ_LEN, EMBED_DIM);

  for (int l = 0; l < NUM_LAYERS; l++) {
    // --- 1. Pre-LayerNorm -> Attention => 残差接続 ---
    forward_layernom(&X, &model->ln1_gamma[l], &model->ln1_beta[l], &X_norm);
    forward_attention(
      &X_norm,
      &model->W_q[l], &model->W_k[l], &model->W_v[l],
      &Q, &K, &V,
      &A_prob,
      &Z
    );
    mat_add(&X, &Z, &X); // 残差接続: X = X + Attention(LN(X))

    // --- 2. Pre-LayerNorm -> MLP -> 残差接続 ---
    forward_layernom(&X, &model->ln2_gamma[l], &model->ln2_beta[l], &X_norm);
    forward_mlp(
      &X_norm,
      &model->W1[l],
      &model->b1[l],
      &model->W2[l],
      &model->b2[l],
      &model->H[l],
      &Y_mlp
    );
    mat_add(&X, &Y_mlp, &X); // 残差接続: X = X + MLP(LN(X))
  } 

  // 5. 出力層 (LM Head)
  forward_lm_head(&X, &model->W_out, output_probabilities);

  // === STEP 5: ワークスペースの解放 ===
  free_matrix(&X); 
  free_matrix(&X_norm);

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
  Matrix *dln1_gamma, Matrix *dln1_beta,
  Matrix *dln2_gamma, Matrix *dln2_beta,
  Matrix *dW1, Matrix *db1, Matrix *dW2, Matrix *db2, 
  Matrix *dW_q, Matrix *dW_k, Matrix *dW_v
) {
  // ==========================================================
  // 1. 各レイヤーの「逆伝播の計算直前」の正確な状態を保持するための
  //    タイムマシン用履歴バッファを、配列として確保します。
  // ==========================================================
  Matrix X_history[NUM_LAYERS + 1]; // 各層の入り口の X
  Matrix X_norm1_history[NUM_LAYERS];
  Matrix Q_history[NUM_LAYERS];
  Matrix K_history[NUM_LAYERS];
  Matrix V_history[NUM_LAYERS];
  Matrix A_prob_history[NUM_LAYERS];
  Matrix Z_history[NUM_LAYERS];
  Matrix X_mid_history[NUM_LAYERS];   // Attention足し算後の X
  Matrix X_norm2_history[NUM_LAYERS];
  Matrix Y_mlp_history[NUM_LAYERS];

  for (int l = 0; l <= NUM_LAYERS; l++) X_history[l] = create_matrix(SEQ_LEN, EMBED_DIM);
  for (int l = 0; l < NUM_LAYERS; l++) {
    X_norm1_history[l] = create_matrix(SEQ_LEN, EMBED_DIM);
    Q_history[l]       = create_matrix(SEQ_LEN, EMBED_DIM);
    K_history[l]       = create_matrix(SEQ_LEN, EMBED_DIM);
    V_history[l]       = create_matrix(SEQ_LEN, EMBED_DIM);
    A_prob_history[l]  = create_matrix(SEQ_LEN, SEQ_LEN);
    Z_history[l]       = create_matrix(SEQ_LEN, EMBED_DIM);
    X_mid_history[l]   = create_matrix(SEQ_LEN, EMBED_DIM);
    X_norm2_history[l] = create_matrix(SEQ_LEN, EMBED_DIM);
    Y_mlp_history[l]   = create_matrix(SEQ_LEN, EMBED_DIM);
  }
  Matrix output_probabilities = create_matrix(1, VOCAB_SIZE);

  // 履歴バッファに順伝播の前期積を記録
  forward_embedding(input_ids, SEQ_LEN, EMBED_DIM, &model->token_embedding, &X_history[0]);
  apply_positional_encoding(&X_history[0]);

  for (int l = 0; l < NUM_LAYERS; l++) {
    forward_layernom(&X_history[l], &model->ln1_gamma[l], &model->ln1_beta[l], &X_norm1_history[l]);
    forward_attention(
      &X_norm1_history[l], 
      &model->W_q[l], &model->W_k[l], &model->W_v[l], 
      &Q_history[l], &K_history[l], &V_history[l], 
      &A_prob_history[l], &Z_history[l]
    );
    mat_add(&X_history[l], &Z_history[l], &X_mid_history[l]);

    forward_layernom(&X_mid_history[l], &model->ln2_gamma[l], &model->ln2_beta[l], &X_norm2_history[l]);
    forward_mlp(
      &X_norm2_history[l], 
      &model->W1[l], &model->b1[l], 
      &model->W2[l], &model->b2[l], 
      &model->H[l], &Y_mlp_history[l]
    );
    mat_add(&X_mid_history[l], &Y_mlp_history[l], &X_history[l + 1]);
  }
  forward_lm_head(&X_history[NUM_LAYERS], &model->W_out, &output_probabilities);

  // =======================================
  // 2. 逆伝播の実行 ( 出力層 -> 入力層 )
  // ========================================
  Matrix dX_stream = create_matrix(SEQ_LEN, EMBED_DIM); // 誤差が逆流するメインストリーム

  // 出力層の逆伝播
  backward_lm_head(&output_probabilities, target_id, &X_history[NUM_LAYERS], &model->W_out, &dX_stream, dW_out);

  // 各ブロックの一時誤差バッファ
  Matrix dBlock_out = create_matrix(SEQ_LEN, EMBED_DIM);
  Matrix dLN_in     = create_matrix(SEQ_LEN, EMBED_DIM);

  // NUM_LAYERS から 0 に向かって、逆順に誤差を逆流させる
  for (int l = NUM_LAYERS - 1; l >= 0; l--) {
    // 2. MLP ブロックの逆伝播
    // 残差接続の微分は、定数なので無視
    // メインストリームの誤差 dX_stream が、そのまま MLP への誤差 (dBlock_out) になる
    for (int i = 0; i < SEQ_LEN * EMBED_DIM; i++) dBlock_out.data[i] = dX_stream.data[i];

    // MLP 本体の逆伝播
    backward_mlp(
      &dBlock_out, 
      &X_norm2_history[l], 
      &model->W1[l], 
      &model->H[l], 
      &model->W2[l],
      &dLN_in, &dW1[l], 
      &db1[l], &dW2[l], &db2[l]
    );
    
    // LayerNorm の逆伝播
    backward_layernorm(
      &dLN_in,
      &X_mid_history[l],
      &model->ln2_gamma[l], 
      &dBlock_out, &dln2_gamma[l], &dln2_beta[l]
    );

    // メインストリームへの合流
    mat_add(&dX_stream, &dBlock_out, &dX_stream);

    // Attention ブロックの逆伝播
    for (int i = 0; i < SEQ_LEN * EMBED_DIM; i++) dBlock_out.data[i] = dX_stream.data[i];

    // Attention 本体の逆伝播
    backward_attention(
      &dBlock_out, &A_prob_history[l], 
      &Q_history[l], &K_history[l], &V_history[l], 
      &X_norm1_history[l], 
      &model->W_q[l], &model->W_k[l], &model->W_v[l], 
      &dLN_in, &dW_q[l], &dW_k[l], &dW_v[l]
    );

    // LayerNorm の逆伝播
    backward_layernorm(
      &dLN_in,
      &X_history[l],
      &model->ln1_gamma[l], 
      &dBlock_out, &dln1_gamma[l], &dln1_beta[l]
    );

    // メインストリームへの合流
    mat_add(&dX_stream, &dBlock_out, &dX_stream);
  }

  // 最下流に届いた dX_stream を使って、Embedding を更新
  backward_embedding(&dX_stream, input_ids, SEQ_LEN, EMBED_DIM, &model->token_embedding);

  // =======================
  // 3. バッファ(メモリ)の解放
  // =======================
  for (int l = 0; l <= NUM_LAYERS; l++) free_matrix(&X_history[l]);
    for (int l = 0; l < NUM_LAYERS; l++) {
      free_matrix(&X_norm1_history[l]); free_matrix(&Q_history[l]); free_matrix(&K_history[l]);
      free_matrix(&V_history[l]);       free_matrix(&A_prob_history[l]); free_matrix(&Z_history[l]);
      free_matrix(&X_mid_history[l]);   free_matrix(&X_norm2_history[l]); free_matrix(&Y_mlp_history[l]);
    }
  free_matrix(&output_probabilities);
  free_matrix(&dX_stream);
  free_matrix(&dBlock_out);
  free_matrix(&dLN_in);
}