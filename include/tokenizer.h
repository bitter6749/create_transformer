#ifndef TOKENIZER_H
#define TOKENIZER_H

#define MAX_TOKEN_LEN 64 
// BERTの保持するご語彙数
#define VOCAB_SIZE_NEW 30522
// 衝突を避けるため、語彙数 (30522) の約2倍の素数・部屋数を確保
#define HASH_TABLE_SIZE 65536

// ハッシュテーブル初期化関数 (外部の vocab.txtを読み込む)
int init_tokenizer_hash(const char *vocab_file_path);

// 外部ファイルからIDに対応する単語文字列を逆引きするための配列ポインタ
extern char **vocab_reverse_dict;

// トークナイズ・デトークナイズ用の関数
int tokenize(const char *text, int *output_ids, int seq_len);
void detokenize(const int *token_ids, int seq_len, char *output_text);

#endif // TOKENIZER_H