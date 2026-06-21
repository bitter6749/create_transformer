#include <stdio.h>
#include <stdlib.h>
#include "model.h"
#include "matrix.h"
#include "transformer.h"



int main() {
  // 1. モデルの宣言とメモリ確保
  SimpleTransformer model;
  init_model(&model);

  // すべての値に適当初期値を入れる
  for (int i=0; i < VOCAB_SIZE * EMBED_DIM; i++) model.token_embedding.data[i] = 0.1f;
  for (int i=0; i < EMBED_DIM * EMBED_DIM; i++) {
    model.W_q.data[i] = 0.5f;
    model.W_k.data[i] = 0.5f;
    model.W_v.data[i] = 0.5f;
  }
  for (int i=0; i < VOCAB_SIZE * EMBED_DIM; i++) model.W_out.data[i] = 0.1f;

  // 2. 入力データ ("cat " を表すIDの配列)
  // c=2, a=0, t=19, space=26
  int input_ids[SEQ_LEN] = {2, 0, 19, 26};
  // 次の一文字の予測確率 (27文字分) を受け取るバッファ
  Matrix output_probabilties = create_matrix(1, VOCAB_SIZE);

  // 3. 予測の実行
  printf("Transformerで次の文字を予測中...\n");
  forward_transformer(&model, input_ids, &output_probabilties);

  // 4. 結果の表示 (最も確率が高い文字のIDを探す)
  int max_id = 0;
  float max_prob = output_probabilties.data[0];
  for (int i = 1; i < VOCAB_SIZE; i++) {
    if (output_probabilties.data[i] > max_prob) {
      max_prob = output_probabilties.data[i];
      max_id = i;
    }
  }
  printf("予測された次の文字のID: %d (確率: %.2f%%)\n", max_id, max_prob * 100.0f);

  // メモリの開放
  free_model(&model);
  free_matrix(&output_probabilties);
  return 0;
}
