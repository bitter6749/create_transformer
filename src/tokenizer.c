#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include "tokenizer.h"

typedef struct {
    char word[MAX_TOKEN_LEN];
    int id;
    bool is_occupied;
} HashEntry;

// ハッシュテーブル本体
static HashEntry hash_table[HASH_TABLE_SIZE];

// IDから文字列を逆引きするための動的配列
char **vocab_reverse_dict = NULL;

// 文字列ハッシュ関数 (DJB2)
static unsigned long djb2_hash(const char *str) {
    unsigned long hash = 5381;
    int c;
    while ((c = (unsigned char)*str++)) {
        hash = ((hash << 5) + hash) + c; // hash * 33 + c
    }
    return hash % HASH_TABLE_SIZE;
}

// ハッシュテーブルへの挿入関数
static void insert_hash(const char *word, int id) {
    unsigned long index = djb2_hash(word);

    // 衝突したら隣へスライド
    while (hash_table[index].is_occupied) {
        index = (index + 1) % HASH_TABLE_SIZE;
    }

    strncpy(hash_table[index].word, word, MAX_TOKEN_LEN - 1);
    hash_table[index].id = id;
    hash_table[index].is_occupied = true;
}

// ハッシュテーブルから検索関数 
static int search_hash(const char *word) {
    unsigned long index = djb2_hash(word);
    unsigned long start_index = index;

    // 衝突したら隣へスライド
    while (hash_table[index].is_occupied)
    {
        if (strcmp(hash_table[index].word, word) == 0) {
            return hash_table[index].id; 
        }
        index = (index + 1) % HASH_TABLE_SIZE; // 衝突の可能性を考慮して隣を確認
        if (index == start_index) break;    // テーブルを1周したら修了
    }

    return -1; // 辞書に存在しない
}

// ハッシュテーブルを構築する関数
int init_tokenizer_hash(const char *vocab_file_path) {
    FILE *f = fopen(vocab_file_path, "r");
    if (f == NULL) {
        fprintf(stderr, "Error: トークナイザー辞書ファイル %s を開けませんでした。\n", vocab_file_path);
        return 0;
    }

    // ハッシュテーブルの初期化
    for (int i = 0; i < HASH_TABLE_SIZE; i++) {
        hash_table[i].is_occupied = false;
    }

    // 逆引き用配列のメモリ確保
    vocab_reverse_dict = (char **)malloc(sizeof(char *) * VOCAB_SIZE_NEW);
    for (int i = 0; i < VOCAB_SIZE_NEW; i++) {
        vocab_reverse_dict[i] = (char *)malloc(sizeof(char) * MAX_TOKEN_LEN);
    }

    char line[MAX_TOKEN_LEN];
    int id = 0;

    // 1行ずつ単語を読み込んでハッシュテーブルと逆引き配列に格納
    while (fgets(line, sizeof(line), f) != NULL && id < VOCAB_SIZE_NEW) {
        // 改行文字を削除
        line[strcspn(line, "\r\n")] = 0;

        if (strlen(line) > 0) {
            insert_hash(line, id);
            strncpy(vocab_reverse_dict[id], line, MAX_TOKEN_LEN - 1);
            id++;
        }
    }

    fclose(f);
    printf(">>> トークナイザー初期化完了: %d 語の辞書をハッシュテーブルに格納しました。\n", id);
    return 1;
}

// テキスト (スペース区切り) をトークンIDの配列に変換する
// 安全のため、入りきらない分は [PAD] で埋める
int tokenize(const char *text, int *output_ids, int seq_len) {
    for (int i = 0; i < seq_len; i++) output_ids[i] = 0; // PAD(0) 初期化

    size_t text_len = strlen(text);
    char *editable_text = (char *)malloc(text_len + 1);
    if (editable_text == NULL) {
        fprintf(stderr, "Error: トークナイズ用のメモリ確保に失敗しました。\n");
        return 0;
    }
    strcpy(editable_text, text);

    // 区切り文字を" \t\r\n" に統一
    char *token = strtok(editable_text, " \t\r\n");
    int token_count = 0;

    while (token != NULL && token_count < seq_len) {
        int len = strlen(token);
        int start = 0;
        bool is_bad = false;

        // 1つの単語をWordPieceに分解するループ
        while (start < len) {
            int end = len;
            int matched_id = -1;
            char sub_word[MAX_TOKEN_LEN];

            while (start < end) {
                // 切り出す文字列の長さを計算
                int sub_len = end - start;

                // 2つ目以降のサブワードには頭に "##" を付与する
                if (start > 0) {
                    snprintf(sub_word, sizeof(sub_word), "##%.*s", sub_len, &token[start]);
                } else {
                    snprintf(sub_word, sizeof(sub_word), "%.*s", sub_len, &token[start]);
                }

                // 自作ハッシュテーブルから検索
                int id = search_hash(sub_word);
                if (id != -1) {
                    matched_id = id;
                    break;
                }
                end--; // 見つからなければ後ろを1文字削る
            }

            // 1文字もマッチしなかった場合は、この単語は未知語 [UNK]
            if (matched_id == -1) {
                is_bad = true;
                break;
            }

            // マッチしたトークンIDを登録
            output_ids[token_count++] = matched_id;
            if (token_count >= seq_len) break;

            // 次のパーツの開始位置を進める
            start = end;
        }

        // WordPiece分解に失敗した単語だった場合は [UNK] (ID: 100) を入れる
        if (is_bad) {
            // 巻き戻して [UNK] に置き換え
            if (token_count > 0 && start > 0) {
                // 既に途中まで入ってしまったサブワードをクリア
                token_count--;
            }
            output_ids[token_count++] = 100; // BERT の [UNK] ID
        }

        if (token_count >= seq_len) break;
        token = strtok(NULL, " \t\r\n");
    }

    // バッファを解放
    free(editable_text);

    return token_count;
}

// トークンIDの配列を人間の読めるテキストに戻す
void detokenize(const int *token_ids, int seq_len, char *output_text) {
    output_text[0] = '\0'; // 空文字で初期化
    if (vocab_reverse_dict == NULL) return;

    for (int i = 0; i < seq_len; i++) {
        int id = token_ids[i];

        if (id >= 0 && id < VOCAB_SIZE_NEW) {
            strcat(output_text, vocab_reverse_dict[id]);
            strcat(output_text, " "); // 単語区切りスペース
        }
    }
}