#include <string.h>
#include <stdio.h>
#include "tokenizer.h"

// テキスト (スペース区切り) をトークンIDの配列に変換する
// 安全のため、入りきらない分は [PAD] で埋める
void tokenize(const char *text, int *output_ids, int seq_len) {
    // まずは出力配列をすべて [PAD] で初期化
    for (int i = 0; i < seq_len; i++) {
        output_ids[i] = 0;
    }

    // 元の文字列を破壊しないようにコピーを作成
    char text_copy[512];
    strncpy(text_copy, text, sizeof(text_copy) - 1);
    text_copy[sizeof(text_copy) - 1] = '\0';

    int token_count = 0;
    // スペース (" ") を区切り文字として単語を抽出
    char *token = strtok(text_copy, " ");

    while (token != NULL && token_count < seq_len) {
        int matched_id = 9; // デフォルトは [UNK] (未知の単語)

        // 辞書と総当たりで比較
        for (int id = 0; id < VOCAB_SIZE_NEW; id++) {
            if (strcmp(token, VOCAB_DICT[id]) == 0) {
                matched_id = id;
                break;
            }
        }

        output_ids[token_count] = matched_id;
        token_count++;
        token = strtok(NULL, " ");
    }
}

// トークンIDの配列を人間の読めるテキストに戻す
void detokenize(const int *token_ids, int seq_len, char *output_text) {
    output_text[0] = '\0'; // 空文字で初期化

    for (int i = 0; i < seq_len; i++) {
        int id = token_ids[i];
        // [PAD] は人間向けには表示しない
        if (id == 0) continue;

        // 安全に文字列を結合
        strcat(output_text, VOCAB_DICT[id]);
        strcat(output_text, " "); // 単語区切りスペース
    }
}