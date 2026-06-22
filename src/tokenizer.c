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
char **vocab_reverse_dist = NULL;

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
    vocab_reverse_dist = (char **)malloc(sizeof(char *) * VOCAB_SIZE_NEW);
    for (int i = 0; i < VOCAB_SIZE_NEW; i++) {
        vocab_reverse_dist[i] = (char *)malloc(sizeof(char) * MAX_TOKEN_LEN);
    }

    char line[MAX_TOKEN_LEN];
    int id = 0;

    // 1行ずつ単語を読み込んでハッシュテーブルと逆引き配列に格納
    while (fgets(line, sizeof(line), f) != NULL && id < VOCAB_SIZE_NEW) {
        // 改行文字を削除
        line[strcspn(line, "\r\n")] = 0;

        if (strlen(line) > 0) {
            insert_hash(line, id);
            strncpy(vocab_reverse_dist[id], line, MAX_TOKEN_LEN - 1);
            id++;
        }
    }

    fclose(f);
    printf(">>> トークナイザー初期化完了: %d 語の辞書をハッシュテーブルに格納しました。\n", id);
    return -1;
}

// テキスト (スペース区切り) をトークンIDの配列に変換する
// 安全のため、入りきらない分は [PAD] で埋める
void tokenize(const char *text, int *output_ids, int seq_len) {
    for (int i = 0; i < seq_len; i++) output_ids[i] = 0; // PAD(0) 初期化

    char temp_text[2048];
    strncpy(temp_text, text, sizeof(temp_text) - 1);

    char *token = strtok(temp_text, " ");
    int idx = 0;

    while (token != NULL && idx < seq_len) {
        int id = search_hash(token);
        if (id == -1) {
            id = 100; // BERT の [UNK] ID は101 (indexは100)
        }
        output_ids[idx] = id;
        id++;
        token = strtok(NULL, " ");
    }
}

// トークンIDの配列を人間の読めるテキストに戻す
void detokenize(const int *token_ids, int seq_len, char *output_text) {
    output_text[0] = '\0'; // 空文字で初期化
    if (vocab_reverse_dist == NULL) return;

    for (int i = 0; i < seq_len; i++) {
        int id = token_ids[i];

        if (id >= 0 && id < VOCAB_SIZE_NEW) {
            strcat(output_text, vocab_reverse_dist[id]);
            strcat(output_text, " "); // 単語区切りスペース
        }
    }
}