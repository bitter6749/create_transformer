#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "model.h"
#include "matrix.h"
#include "transformer.h"
#include "ops.h"
#include "tokenizer.h"

int main() {
  // 1. モデルの宣言とメモリ確保
  SimpleTransformer model;
  init_model(&model);

  // =========================================================
  //  2. ハードディスクから学習済みの重みをロード（復元）
  // =========================================================
  load_model_checkpoint(&model, "build/checkpoint_novel");

  // =========================================================
  //  3. 人間からのプロンプト（入力文）の設定
  // =========================================================
  // 新しい小説（novel.txt）の冒頭部分を入力にします
  const char *prompt_text = "僕 歩く 。"; 
  
  printf("\n=== 文章の自動生成を開始します ===\n");
  printf("入力（プロンプト）: \"%s\"\n\n", prompt_text);

  // 入力されたテキストをトークンIDの配列に変換
  int current_tokens[SEQ_LEN] = {0, 0, 0, 0};
  tokenize(prompt_text, current_tokens, SEQ_LEN);

  printf("生成された続き: ");
  
  // 確率を受け取るバッファ
  Matrix output_probabilities = create_matrix(1, VOCAB_SIZE);

  // =========================================================
  //  4. 自動生成ループ（1語ずつ未来を予測して付け足していく）
  // =========================================================
  for (int step = 0; step < 5; step++) {
    
    // 現在の入力（4単語）から、次に来る単語の確率分布を順伝播で計算
    forward_transformer(&model, current_tokens, &output_probabilities);

    // 最も確率（確信度）が高い単語のIDを探す（貪欲法 / Greedy Search）
    int next_word_id = 0;
    float max_prob = output_probabilities.data[0];
    for (int i = 1; i < VOCAB_SIZE; i++) {
      if (output_probabilities.data[i] > max_prob) {
        max_prob = output_probabilities.data[i];
        next_word_id = i;
      }
    }

    // もし [PAD] や [UNK] が予測されたら生成を終了する
    if (next_word_id == 0 || next_word_id == 9) {
      break;
    }

    // 予測された単語を画面に表示！
    printf("%s ", VOCAB_DICT[next_word_id]);

    //  【修正】トークナイザーの格納順（先頭詰め）に合わせた正しいスライディング・ウィンドウ
    // 1マスずつ左にずらして、空いた最後の場所に次の単語を正しく配置します
    for (int i = 0; i < SEQ_LEN - 1; i++) {
      current_tokens[i] = current_tokens[i + 1];
    }
    current_tokens[SEQ_LEN - 1] = next_word_id;
  }

  printf("\n\n=== 生成終了 ===\n");

  // メモリの解放
  free_model(&model);
  free_matrix(&output_probabilities);

  return 0;
}