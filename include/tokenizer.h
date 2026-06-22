#ifndef TOKENIZER_H
#define TOKENIZER_H

#define MAX_TOKEN_LEN 32
// 今回のミニ辞書に登録する単語数
#define VOCAB_SIZE_NEW 20

// 辞書の単語リスト
static const char *VOCAB_DICT[VOCAB_SIZE_NEW] = {
  "[PAD]",       // ID 0
  "おはよ",     // ID 1
  "おやすみ",   // ID 2
  "元気",       // ID 3
  "？",         // ID 4
  "！",         // ID 5
  "うれしい",   // ID 6
  "かなしい",   // ID 7
  "バイバイ",   // ID 8
  "[UNK]",       // ID 9
  "僕",         // ID 10
  "君",         // ID 11
  "空",         // ID 12
  "星",         // ID 13
  "見る",       // ID 14
  "笑う",       // ID 15
  "歩く",       // ID 16
  "綺麗",       // ID 17
  "夜",         // ID 18
  "静か"        // ID 19
};

// テキスト (スペース区切り) をトークンIDの配列に変換する
// 例: "おはよ !" -> [1, 5, 0, 0]
void tokenize(const char *text, int *output_ids, int seq_len);

// トークンIDの配列を人間の読めるテキストに戻す
// 例: [1, 5, 0, 0] -> "おはよ ! "
void detokenize(const int *token_ids, int seq_len, char *output_text);

#endif // TOKENIZER_H