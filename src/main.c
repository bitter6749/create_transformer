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

  // Attention層
  for (int i=0; i < model.W_q.rows * model.W_q.cols; i++) {
    model.W_q.data[i] = rand_weight(0.05f);
    model.W_k.data[i] = rand_weight(0.05f);
    model.W_v.data[i] = rand_weight(0.05f);
  }

  // MLP層
  for (int i=0; i < model.W1.rows * model.W1.cols; i++) model.W1.data[i] = rand_weight(0.05f);
  for (int i=0; i < model.b1.rows * model.b1.cols; i++) model.b1.data[i] = 0.0f;

  for (int i=0; i < model.W2.rows * model.W2.cols; i++) model.W2.data[i] = rand_weight(0.05f);
  for (int i=0; i < model.b2.rows * model.b2.cols; i++) model.b2.data[i] = 0.0f;
  
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
  Matrix dW1    = create_matrix(EMBED_DIM, MLP_HIDDEN_DIM);
  Matrix db1    = create_matrix(1, MLP_HIDDEN_DIM);
  Matrix dW2    = create_matrix(MLP_HIDDEN_DIM, EMBED_DIM);
  Matrix db2    = create_matrix(1, EMBED_DIM);  
  Matrix dW_q   = create_matrix(EMBED_DIM, EMBED_DIM);
  Matrix dW_k   = create_matrix(EMBED_DIM, EMBED_DIM);
  Matrix dW_v   = create_matrix(EMBED_DIM, EMBED_DIM);

  // 次の一文字の予測確率 (27文字分) を受け取るバッファ
  Matrix output_probabilities = create_matrix(1, VOCAB_SIZE);

  // ハイパーパラメータ
  float learning_rate = 0.1f; // 学習率 (lr)
  int epochs = 100;           // 100回繰り返し学習する

  printf("=== Transformerの学習を開始します ===\n");

  // 4. トレーニングループ
  for (int epoch = 1; epoch <= epochs; epoch++) {

    // STEP 4-1: 逆伝播関数を呼び出し、全レイヤーの勾配を計算
    backward_transformer(
      &model, input_ids, target_id,
      &dW_out, &dW1, &db1, &dW2, &db2, &dW_q, &dW_k, &dW_v
    );

    // STEP 4-2: 算出された勾配を使って、各パラメーターを更新 (SGD)
    gradient_descent_update(&model.W_out, &dW_out, learning_rate);
    gradient_descent_update(&model.W1, &dW1, learning_rate);
    gradient_descent_update(&model.b1, &db1, learning_rate);
    gradient_descent_update(&model.W2, &dW2, learning_rate);
    gradient_descent_update(&model.b2, &db2, learning_rate);
    gradient_descent_update(&model.W_q, &dW_q, learning_rate);
    gradient_descent_update(&model.W_k, &dW_k, learning_rate);
    gradient_descent_update(&model.W_v, &dW_v, learning_rate);

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
  free_matrix(&dW_out); free_matrix(&dW1); free_matrix(&db1);
  free_matrix(&dW2); free_matrix(&db2);
  free_matrix(&dW_q); free_matrix(&dW_k); free_matrix(&dW_v);

  return 0;
}
