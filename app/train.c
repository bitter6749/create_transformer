#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "model.h"
#include "matrix.h"
#include "transformer.h"
#include "mlp.h"
#include "ops.h"
#include "tokenizer.h"

// 指定した範囲 (-scale から +scale) のランダムな少数を生成する関数
float rand_weight(float scale) {
  // 0.0 ~ 1.0 の乱数を作り、-1.0 ~ 1.0 に変換してから scale をかける
  float r = ((float)rand() / (float)RAND_MAX);
  return (r * 2.0f - 1.0f) * scale;
}

// テキストファイルからすべての単語を読み込んでID配列にする関数
int load_and_tokenize_file(const char *filename, int *tokens_out, int max_tokens) {
  FILE *f = fopen(filename, "r");
  if (f == NULL) {
    fprintf(stderr, "Error: %s を開けませんでした。\n", filename);
    return 0;
  }

  char word[MAX_TOKEN_LEN];
  int count = 0;

  // ファイルからスペースまたは改行区切りで1単語ずつ読み込む
  while (fscanf(f, "%31s", word) != EOF && count < max_tokens) {
    int matched_id = 9; // デフォルトは [UNK]

    // 辞書と総当たりで比較してIDに変換
    for (int id = 0; id < VOCAB_SIZE_NEW; id++) {
      if (strcmp(word, vocab_reverse_dict[id]) == 0) {
        matched_id = id;
        break;
      }
    }
    tokens_out[count] = matched_id;
    count++;
  }

  fclose(f);
  return count; // 読み込んだ総単語数を返す
}

int main() {
  // seed を固定 (実行するたび同じ値になるようにする)
  srand(42);

  // 外部ファイルを読み込んでハッシュテーブルを構築 
  if (!init_tokenizer_hash("vocab.txt")) {
    return EXIT_FAILURE;
  }

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

  // 2. テキストファイルを読み込んでメモリ上でトークナイズ
  FILE *f_txt = fopen("data_input_1.txt", "r");
  if (f_txt == NULL) {
    fprintf(stderr, "Error: 'data_input.txt' が見つかりません。テキストファイルを用意してください。\n");
    free_model(&model);
    return EXIT_FAILURE;
  }

  // ファイルサイズを計測して動的にバッファを確保
  fseek(f_txt, 0, SEEK_END);
  long file_size = ftell(f_txt);
  fseek(f_txt, 0, SEEK_SET);

  char *raw_text = (char *)malloc(file_size + 1);
  size_t read_bytes = fread(raw_text, 1, file_size, f_txt);
  raw_text[read_bytes] = '\0';
  fclose(f_txt);

  // トークンを格納する十分な大きさの配列を確保（文字数より多くなることはない）
  int *full_tokens = (int *)malloc(sizeof(int) * read_bytes);
  
  // 配列全体を処理するため、一時的に巨大なseq_lenとしてtokenizeを呼び出す
  printf(">>> ハッシュテーブルを使用してリアルタイムWordPieceトークナイズを実行中...\n");

  // 実際に詰め込まれた有効なトークン数をカウント
  int total_words = tokenize(raw_text, full_tokens, read_bytes);
  free(raw_text); // 原文テキストバッファはもう不要なので解放

  printf(">>> WordPiece分解完了: 総トークン数 %d 個をメモリに展開しました。\n", total_words);

  if (total_words < SEQ_LEN + 1) {
    fprintf(stderr, "Error: トークン数が少なすぎます。終了します。\n");
    free(full_tokens);
    free_model(&model);
    return EXIT_FAILURE;
  }


  
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

  // バッチ全体の勾配を「蓄積（合計）」していくための巨大な箱
  Matrix acc_dW_out = create_matrix(EMBED_DIM, VOCAB_SIZE);
  Matrix acc_dln1_gamma[NUM_LAYERS], acc_dln1_beta[NUM_LAYERS], acc_dln2_gamma[NUM_LAYERS], acc_dln2_beta[NUM_LAYERS];
  Matrix acc_dW1[NUM_LAYERS], acc_db1[NUM_LAYERS], acc_dW2[NUM_LAYERS], acc_db2[NUM_LAYERS];
  Matrix acc_dW_q[NUM_LAYERS], acc_dW_k[NUM_LAYERS], acc_dW_v[NUM_LAYERS];

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

    // 蓄積用の箱のメモリ確保
    acc_dln1_gamma[l] = create_matrix(1, EMBED_DIM); 
    acc_dln1_beta[l]  = create_matrix(1, EMBED_DIM);
    acc_dln2_gamma[l] = create_matrix(1, EMBED_DIM); 
    acc_dln2_beta[l]  = create_matrix(1, EMBED_DIM);

    acc_dW1[l]        = create_matrix(EMBED_DIM, MLP_HIDDEN_DIM); 
    acc_db1[l]        = create_matrix(1, MLP_HIDDEN_DIM);
    acc_dW2[l]        = create_matrix(MLP_HIDDEN_DIM, EMBED_DIM); 
    acc_db2[l]        = create_matrix(1, EMBED_DIM); 

    acc_dW_q[l]       = create_matrix(EMBED_DIM, EMBED_DIM); 
    acc_dW_k[l]       = create_matrix(EMBED_DIM, EMBED_DIM);
    acc_dW_v[l]       = create_matrix(EMBED_DIM, EMBED_DIM);
  }

  // 次の一文字の予測確率 (27文字分) を受け取るバッファ
  Matrix output_probabilities = create_matrix(1, VOCAB_SIZE);

  // ハイパーパラメータ
  float learning_rate = 0.02f; // 学習率 (lr)
  int batch_size = 64;          // 64個のサンプルごとに1回更新する
  int epochs = 100;           // 100回繰り返し学習する

  printf("=== Transformerの学習を開始します ===\n");

  // 4. トレーニングループ
  for (int epoch = 1; epoch <= epochs; epoch++) {
    float epoch_loss_sum = 0.0f;
    int step_count = 0;
    int batch_accumulator_count = 0;

    // バッチ全体の蓄積箱を 0.0f でクリア
    for (int i = 0; i < acc_dW_out.rows * acc_dW_out.cols; i++) acc_dW_out.data[i] = 0.0f;
    for (int l = 0; l < NUM_LAYERS; l++) {
      for (int i = 0; i < acc_dln1_gamma[l].rows * acc_dln1_gamma[l].cols; i++) { 
        acc_dln1_gamma[l].data[i] = 0.0f; 
        acc_dln1_beta[l].data[i] = 0.0f; 
        acc_dln2_gamma[l].data[i] = 0.0f; 
        acc_dln2_beta[l].data[i] = 0.0f; 
      }
      for (int i = 0; i < acc_dW_q[l].rows * acc_dW_q[l].cols; i++) acc_dW_q[l].data[i] = 0.0f;
      for (int i = 0; i < acc_dW_k[l].rows * acc_dW_k[l].cols; i++) acc_dW_k[l].data[i] = 0.0f;
      for (int i = 0; i < acc_dW_v[l].rows * acc_dW_v[l].cols; i++) acc_dW_v[l].data[i] = 0.0f;
      for (int i = 0; i < acc_dW1[l].rows * acc_dW1[l].cols; i++)   acc_dW1[l].data[i] = 0.0f;
      for (int i = 0; i < acc_db1[l].rows * acc_db1[l].cols; i++)   acc_db1[l].data[i] = 0.0f;
      for (int i = 0; i < acc_dW2[l].rows * acc_dW2[l].cols; i++)   acc_dW2[l].data[i] = 0.0f;
      for (int i = 0; i < acc_db2[l].rows * acc_db2[l].cols; i++)   acc_db2[l].data[i] = 0.0f;
    }

    int input_ids[SEQ_LEN]; // PAD(0) で初期化
    for (int p = 0; p < total_words - 1; p++) {
      memset(input_ids, 0 , sizeof(input_ids));

      // 窓の中に過去の単語を詰め込む（最大4つ）
      int idx = 3;
      for (int k = p; k >= 0 && idx >= 0; k--) {
        input_ids[idx] = full_tokens[k];
        idx--;
      }

      // 今回予測すべき正解ターゲット（次の1単語）
      int target_id = full_tokens[p + 1];
      
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

      // 計算した勾配（dW）を、バッチ蓄積用の箱（acc_dW）にひたすら足し算する
      for (int i = 0; i < dW_out.rows * dW_out.cols; i++) acc_dW_out.data[i] += dW_out.data[i];
      for (int l = 0; l < NUM_LAYERS; l++) {
        for (int i = 0; i < dln1_gamma[l].rows * dln1_gamma[l].cols; i++) { 
          acc_dln1_gamma[l].data[i] += dln1_gamma[l].data[i];
          acc_dln1_beta[l].data[i] += dln1_beta[l].data[i]; 
          acc_dln2_gamma[l].data[i] += dln2_gamma[l].data[i]; 
          acc_dln2_beta[l].data[i] += dln2_beta[l].data[i]; 
        }
        for (int i = 0; i < dW_q[l].rows * dW_q[l].cols; i++) acc_dW_q[l].data[i] += dW_q[l].data[i];
        for (int i = 0; i < dW_k[l].rows * dW_k[l].cols; i++) acc_dW_k[l].data[i] += dW_k[l].data[i];
        for (int i = 0; i < dW_v[l].rows * dW_v[l].cols; i++) acc_dW_v[l].data[i] += dW_v[l].data[i];
        for (int i = 0; i < dW1[l].rows * dW1[l].cols; i++)   acc_dW1[l].data[i] += dW1[l].data[i];
        for (int i = 0; i < db1[l].rows * db1[l].cols; i++)   acc_db1[l].data[i] += db1[l].data[i];
        for (int i = 0; i < dW2[l].rows * dW2[l].cols; i++)   acc_dW2[l].data[i] += dW2[l].data[i];
        for (int i = 0; i < db2[l].rows * db2[l].cols; i++)   acc_db2[l].data[i] += db2[l].data[i];
      }

      batch_accumulator_count++;

      // バッチサイズ分溜まったら、一括して平均化・クリッピング・更新を行う
      if (batch_accumulator_count == batch_size || p == total_words - 2) {
        
        // 1. 勾配をサンプル数で割って平均化する (パラメータ更新はここでは絶対にしない)
        float inv_b = 1.0f / (float)batch_accumulator_count;
        for (int i = 0; i < acc_dW_out.rows * acc_dW_out.cols; i++) acc_dW_out.data[i] *= inv_b;
        for (int l = 0; l < NUM_LAYERS; l++) {
          for (int i = 0; i < acc_dln1_gamma[l].rows * acc_dln1_gamma[l].cols; i++) {
            acc_dln1_gamma[l].data[i] *= inv_b; acc_dln1_beta[l].data[i] *= inv_b;
            acc_dln2_gamma[l].data[i] *= inv_b; acc_dln2_beta[l].data[i] *= inv_b;
          }
          for (int i = 0; i < acc_dW_q[l].rows * acc_dW_q[l].cols; i++) acc_dW_q[l].data[i] *= inv_b;
          for (int i = 0; i < acc_dW_k[l].rows * acc_dW_k[l].cols; i++) acc_dW_k[l].data[i] *= inv_b;
          for (int i = 0; i < acc_dW_v[l].rows * acc_dW_v[l].cols; i++) acc_dW_v[l].data[i] *= inv_b;
          for (int i = 0; i < acc_dW1[l].rows * acc_dW1[l].cols; i++)   acc_dW1[l].data[i] *= inv_b;
          for (int i = 0; i < acc_db1[l].rows * acc_db1[l].cols; i++)   acc_db1[l].data[i] *= inv_b;
          for (int i = 0; i < acc_dW2[l].rows * acc_dW2[l].cols; i++)   acc_dW2[l].data[i] *= inv_b;
          for (int i = 0; i < acc_db2[l].rows * acc_db2[l].cols; i++)   acc_db2[l].data[i] *= inv_b;
        }

        // 2. 勾配クリッピング (暴走を抑える安全弁：0.0fにするのではなく、上限値 clip_val に丸める)
        float clip_val = 1.0f;
        for (int i = 0; i < acc_dW_out.rows * acc_dW_out.cols; i++) {
          if (acc_dW_out.data[i] > clip_val)  acc_dW_out.data[i] = clip_val;
          if (acc_dW_out.data[i] < -clip_val) acc_dW_out.data[i] = -clip_val;
        }
        for (int l = 0; l < NUM_LAYERS; l++) {
          for (int i = 0; i < acc_dln1_gamma[l].rows * acc_dln1_gamma[l].cols; i++) { 
            if (acc_dln1_gamma[l].data[i] > clip_val)  acc_dln1_gamma[l].data[i] = clip_val;
            if (acc_dln1_gamma[l].data[i] < -clip_val) acc_dln1_gamma[l].data[i] = -clip_val;
            if (acc_dln1_beta[l].data[i] > clip_val)   acc_dln1_beta[l].data[i] = clip_val;
            if (acc_dln1_beta[l].data[i] < -clip_val)  acc_dln1_beta[l].data[i] = -clip_val;
            if (acc_dln2_gamma[l].data[i] > clip_val)  acc_dln2_gamma[l].data[i] = clip_val;
            if (acc_dln2_gamma[l].data[i] < -clip_val) acc_dln2_gamma[l].data[i] = -clip_val;
            if (acc_dln2_beta[l].data[i] > clip_val)   acc_dln2_beta[l].data[i] = clip_val;
            if (acc_dln2_beta[l].data[i] < -clip_val)  acc_dln2_beta[l].data[i] = -clip_val;
          }
          for (int i = 0; i < acc_dW_q[l].rows * acc_dW_q[l].cols; i++) { if (acc_dW_q[l].data[i] > clip_val) acc_dW_q[l].data[i] = clip_val; if (acc_dW_q[l].data[i] < -clip_val) acc_dW_q[l].data[i] = -clip_val; }
          for (int i = 0; i < acc_dW_k[l].rows * acc_dW_k[l].cols; i++) { if (acc_dW_k[l].data[i] > clip_val) acc_dW_k[l].data[i] = clip_val; if (acc_dW_k[l].data[i] < -clip_val) acc_dW_k[l].data[i] = -clip_val; }
          for (int i = 0; i < acc_dW_v[l].rows * acc_dW_v[l].cols; i++) { if (acc_dW_v[l].data[i] > clip_val) acc_dW_v[l].data[i] = clip_val; if (acc_dW_v[l].data[i] < -clip_val) acc_dW_v[l].data[i] = -clip_val; }
          for (int i = 0; i < acc_dW1[l].rows * acc_dW1[l].cols; i++)   { if (acc_dW1[l].data[i] > clip_val) acc_dW1[l].data[i] = clip_val; if (acc_dW1[l].data[i] < -clip_val) acc_dW1[l].data[i] = -clip_val; }
          for (int i = 0; i < acc_db1[l].rows * acc_db1[l].cols; i++)   { if (acc_db1[l].data[i] > clip_val) acc_db1[l].data[i] = clip_val; if (acc_db1[l].data[i] < -clip_val) acc_db1[l].data[i] = -clip_val; }
          for (int i = 0; i < acc_dW2[l].rows * acc_dW2[l].cols; i++)   { if (acc_dW2[l].data[i] > clip_val) acc_dW2[l].data[i] = clip_val; if (acc_dW2[l].data[i] < -clip_val) acc_dW2[l].data[i] = -clip_val; }
          for (int i = 0; i < acc_db2[l].rows * acc_db2[l].cols; i++)   { if (acc_db2[l].data[i] > clip_val) acc_db2[l].data[i] = clip_val; if (acc_db2[l].data[i] < -clip_val) acc_db2[l].data[i] = -clip_val; }
        }

        // 3. 平均化＋クリッピングされた安全な勾配を使って、パラメータを一括更新 (SGD)
        gradient_descent_update(&model.W_out, &acc_dW_out, learning_rate);
        for (int l = 0; l < NUM_LAYERS; l++) {
          gradient_descent_update(&model.ln1_gamma[l], &acc_dln1_gamma[l], learning_rate);
          gradient_descent_update(&model.ln1_beta[l],  &acc_dln1_beta[l],  learning_rate);
          gradient_descent_update(&model.ln2_gamma[l], &acc_dln2_gamma[l], learning_rate);
          gradient_descent_update(&model.ln2_beta[l],  &acc_dln2_beta[l],  learning_rate);

          gradient_descent_update(&model.W_q[l], &acc_dW_q[l], learning_rate);
          gradient_descent_update(&model.W_k[l], &acc_dW_k[l], learning_rate);
          gradient_descent_update(&model.W_v[l], &acc_dW_v[l], learning_rate);

          gradient_descent_update(&model.W1[l],  &acc_dW1[l],  learning_rate);
          gradient_descent_update(&model.b1[l],  &acc_db1[l],  learning_rate);
          gradient_descent_update(&model.W2[l],  &acc_dW2[l],  learning_rate);
          gradient_descent_update(&model.b2[l],  &acc_db2[l],  learning_rate);
        }

        // 4. 次のバッチのために蓄積カウンタと蓄積箱を 0.0f に完全リセット
        batch_accumulator_count = 0;
        for (int i = 0; i < acc_dW_out.rows * acc_dW_out.cols; i++) acc_dW_out.data[i] = 0.0f;
        for (int l = 0; l < NUM_LAYERS; l++) {
          for (int i = 0; i < acc_dln1_gamma[l].rows * acc_dln1_gamma[l].cols; i++) { 
            acc_dln1_gamma[l].data[i] = 0.0f; acc_dln1_beta[l].data[i] = 0.0f; 
            acc_dln2_gamma[l].data[i] = 0.0f; acc_dln2_beta[l].data[i] = 0.0f; 
          }
          for (int i = 0; i < acc_dW_q[l].rows * acc_dW_q[l].cols; i++) acc_dW_q[l].data[i] = 0.0f;
          for (int i = 0; i < acc_dW_k[l].rows * acc_dW_k[l].cols; i++) acc_dW_k[l].data[i] = 0.0f;
          for (int i = 0; i < acc_dW_v[l].rows * acc_dW_v[l].cols; i++) acc_dW_v[l].data[i] = 0.0f;
          for (int i = 0; i < acc_dW1[l].rows * acc_dW1[l].cols; i++)   acc_dW1[l].data[i] = 0.0f;
          for (int i = 0; i < acc_db1[l].rows * acc_db1[l].cols; i++)   acc_db1[l].data[i] = 0.0f;
          for (int i = 0; i < acc_dW2[l].rows * acc_dW2[l].cols; i++)   acc_dW2[l].data[i] = 0.0f;
          for (int i = 0; i < acc_db2[l].rows * acc_db2[l].cols; i++)   acc_db2[l].data[i] = 0.0f;
        }
      } 

      // 損失の簡易計算と進捗ログ
      forward_transformer(&model, input_ids, &output_probabilities);

      float prob = output_probabilities.data[target_id];
      if (prob < 1e-5f) prob = 1e-5f;
      if (prob > 1.0f) prob = 1.0f;

      epoch_loss_sum += -logf(prob);
      step_count++;

      // 100 ステップごとに現在の平均Lossを出力して進捗を可視化
      if (step_count % 100 == 0) {
        printf("  [Step %d / %d] 現在の暫定平均Loss: %.4f\n", step_count, total_words, epoch_loss_sum / (float)step_count);
      }
    }

    printf("-- Epoch %3d: 完了時の全体平均Loss: %.4f --\n", epoch, epoch_loss_sum / (float)step_count);
  }

  printf("=== 学習完了 ===\n");

  save_model_checkpoint(&model, "build/checkpoint_novel");

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

  // ハッシュテーブルの逆引き辞書メモリを解放
  if (vocab_reverse_dict != NULL) {
    for (int i = 0; i < VOCAB_SIZE_NEW; i++) {
      free(vocab_reverse_dict[i]);
    }
    free(vocab_reverse_dict);
  }

  return 0;
}
