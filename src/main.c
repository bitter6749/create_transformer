#include <stdio.h>
#include <stdlib.h>
#include "model.h"
#include "matrix.h"
#include "transformer.h"
#include "mlp.h"
#include "ops.h"

// 指定した範囲 (-scale から +scale) のランダムな少数を生成する関数
float rand_weight(float scale) {
  // 0.0 ~ 1.0 の乱数を作り、-1.0 ~ 1.0 に変換してから scale をかける
  float r = ((float)rand() / (float)RAND_MAX);
  return (r * 2.0f - 1.0f) * scale;
}

int main() {
  // seed を固定 (実行するたび同じ値になるようにする)
  srand(42);

  // 1. モデルの宣言とメモリ確保
  SimpleTransformer model;
  init_model(&model);

  // ==============================================
  //  すべての重みとバイアスをランダムな値で初期化する
  // ==============================================

  // 埋め込み(Embedding)層
  for (int i=0; i < model.token_embedding.rows * model.token_embedding.cols; i++) {
    model.token_embedding.data[i] = rand_weight(0.1f);
  }

  for (int l = 0; l < NUM_LAYERS; l++) {
    // LayerNorm (gammaは1.0, betaは0.0)

    // LayerNorm1 (Attention手前用) 
    for (int j = 0; j < EMBED_DIM; j++) {
      model.ln1_gamma[l].data[j] = 1.0f;
      model.ln1_beta[l].data[j] = 0.0f;
    }
    // LayerNorm2 (MLP手前用)
    for (int j = 0; j < EMBED_DIM; j++) {
      model.ln2_gamma[l].data[j] = 1.0f;
      model.ln2_beta[l].data[j] = 0.0f;
    }

    // Attention層
    for (int i = 0; i < model.W_q[l].rows * model.W_q[l].cols; i++) model.W_q[l].data[i] = rand_weight(0.05f);
    for (int i = 0; i < model.W_k[l].rows * model.W_k[l].cols; i++) model.W_k[l].data[i] = rand_weight(0.05f);
    for (int i = 0; i < model.W_v[l].rows * model.W_v[l].cols; i++) model.W_v[l].data[i] = rand_weight(0.05f);

    // MLP層
    for (int i = 0; i < model.W1[l].rows * model.W1[l].cols; i++) model.W1[l].data[i] = rand_weight(0.05f);
    for (int i = 0; i < model.b1[l].rows * model.b1[l].cols; i++) model.b1[l].data[i] = 0.0f;

    for (int i = 0; i < model.W2[l].rows * model.W2[l].cols; i++) model.W2[l].data[i] = rand_weight(0.05f);
    for (int i = 0; i < model.b2[l].rows * model.b2[l].cols; i++) model.b2[l].data[i] = 0.0f;
  }

  // 出力層
  for (int i=0; i < model.W_out.rows * model.W_out.cols; i++) {
    model.W_out.data[i] = rand_weight(0.1f);
  }

  // 2. 入力データ ("cat " を表すIDの配列)
  // c=2, a=0, t=19, space=26
  int input_ids[SEQ_LEN] = {2, 0, 19, 26};
  // 正解 (ターゲット) : 次に来る文字を 'c' (ID: 2) と仮定
  int target_id = 2;

  // 3. 勾配 (グラジエント) を格納するためのバッファを確保 (モデルの重みとまったく同じサイズ)
  Matrix dW_out = create_matrix(EMBED_DIM, VOCAB_SIZE);

  Matrix dln1_gamma[NUM_LAYERS]; 
  Matrix dln1_beta[NUM_LAYERS];
  Matrix dln2_gamma[NUM_LAYERS]; 
  Matrix dln2_beta[NUM_LAYERS];
  Matrix dW1[NUM_LAYERS];
  Matrix db1[NUM_LAYERS]; 
  Matrix dW2[NUM_LAYERS];
  Matrix db2[NUM_LAYERS];
  Matrix dW_q[NUM_LAYERS];
  Matrix dW_k[NUM_LAYERS];
  Matrix dW_v[NUM_LAYERS];

  for (int l = 0; l < NUM_LAYERS; l++) {
    dln1_gamma[l] = create_matrix(1, EMBED_DIM);
    dln1_beta[l]  = create_matrix(1, EMBED_DIM);
    dln2_gamma[l] = create_matrix(1, EMBED_DIM);
    dln2_beta[l]  = create_matrix(1, EMBED_DIM);

    dW1[l]        = create_matrix(EMBED_DIM, MLP_HIDDEN_DIM);
    db1[l]        = create_matrix(1, MLP_HIDDEN_DIM);
    dW2[l]        = create_matrix(MLP_HIDDEN_DIM, EMBED_DIM);
    db2[l]        = create_matrix(1, EMBED_DIM);  

    dW_q[l]       = create_matrix(EMBED_DIM, EMBED_DIM);
    dW_k[l]       = create_matrix(EMBED_DIM, EMBED_DIM);
    dW_v[l]       = create_matrix(EMBED_DIM, EMBED_DIM);
  }

  // 次の一文字の予測確率 (27文字分) を受け取るバッファ
  Matrix output_probabilities = create_matrix(1, VOCAB_SIZE);

  // ハイパーパラメータ
  float learning_rate = 0.25f; // 学習率 (lr)
  int epochs = 600;           // 100回繰り返し学習する

  printf("=== Transformerの学習を開始します ===\n");

  // 4. トレーニングループ
  for (int epoch = 1; epoch <= epochs; epoch++) {
    // 新しいエポックの計算を始める前に、すべての勾配箱の中身を 0.0f にクリアする
    for (int i = 0; i < dW_out.rows * dW_out.cols; i++) dW_out.data[i] = 0.0f;
    for (int l = 0; l < NUM_LAYERS; l++) {
      for (int i = 0; i < dln1_gamma[l].rows * dln1_gamma[l].cols; i++) {
        dln1_gamma[l].data[i] = 0.0f; dln1_beta[l].data[i] = 0.0f;
        dln2_gamma[l].data[i] = 0.0f; dln2_beta[l].data[i] = 0.0f;
      }
      for (int i = 0; i < dW_q[l].rows * dW_q[l].cols; i++) dW_q[l].data[i] = 0.0f;
      for (int i = 0; i < dW_k[l].rows * dW_k[l].cols; i++) dW_k[l].data[i] = 0.0f;
      for (int i = 0; i < dW_v[l].rows * dW_v[l].cols; i++) dW_v[l].data[i] = 0.0f;
      for (int i = 0; i < dW1[l].rows * dW1[l].cols; i++)   dW1[l].data[i] = 0.0f;
      for (int i = 0; i < db1[l].rows * db1[l].cols; i++)   db1[l].data[i] = 0.0f;
      for (int i = 0; i < dW2[l].rows * dW2[l].cols; i++)   dW2[l].data[i] = 0.0f;
      for (int i = 0; i < db2[l].rows * db2[l].cols; i++)   db2[l].data[i] = 0.0f;
    }

    // STEP 4-1: 逆伝播関数を呼び出し、全レイヤーの勾配を計算
    backward_transformer(
      &model, input_ids, target_id,
      &dW_out, dln1_gamma, dln1_beta, dln2_gamma, dln2_beta,
      dW1, db1, dW2, db2, dW_q, dW_k, dW_v
    );

    // STEP 4-2: 算出された勾配を使って、各パラメーターを更新 (SGD)
    gradient_descent_update(&model.W_out, &dW_out, learning_rate);

    // 全レイヤーのパラメータを一括ループ更新
    for (int l = 0; l < NUM_LAYERS; l++) {
      gradient_descent_update(&model.ln1_gamma[l], &dln1_gamma[l], learning_rate);
      gradient_descent_update(&model.ln1_beta[l],  &dln1_beta[l],  learning_rate);
      gradient_descent_update(&model.ln2_gamma[l], &dln2_gamma[l], learning_rate);
      gradient_descent_update(&model.ln2_beta[l],  &dln2_beta[l],  learning_rate);
      
      gradient_descent_update(&model.W_q[l],   &dW_q[l],   learning_rate);
      gradient_descent_update(&model.W_k[l],   &dW_k[l],   learning_rate);
      gradient_descent_update(&model.W_v[l],   &dW_v[l],   learning_rate);
      
      gradient_descent_update(&model.W1[l],    &dW1[l],    learning_rate);
      gradient_descent_update(&model.b1[l],    &db1[l],    learning_rate);
      gradient_descent_update(&model.W2[l],    &dW2[l],    learning_rate);
      gradient_descent_update(&model.b2[l],    &db2[l],    learning_rate);
    }
    // 定期的に進捗 (正解文字 'c' の予測率の変化) を表示
    if (epoch == 1 || epoch % 20 == 0) {
      forward_transformer(&model, input_ids, &output_probabilities);
      printf("Epoch %3d: 正解文字(ID: %d)の予測確率: %.2f%%\n",
              epoch, target_id, output_probabilities.data[target_id] * 100.0f);
    }
  }

  printf("=== 学習完了 ===\n");

  // メモリの開放
  free_model(&model);
  free_matrix(&output_probabilities);
  free_matrix(&dW_out); 
  for (int l = 0; l < NUM_LAYERS; l++) {
    free_matrix(&dln1_gamma[l]);
    free_matrix(&dln1_beta[l]);

    free_matrix(&dln2_gamma[l]); 
    free_matrix(&dln2_beta[l]);

    free_matrix(&dW_q[l]); 
    free_matrix(&dW_k[l]); 
    free_matrix(&dW_v[l]);

    free_matrix(&dW1[l]); 
    free_matrix(&db1[l]);

    free_matrix(&dW2[l]);
    free_matrix(&db2[l]);
  }

  return 0;
}
