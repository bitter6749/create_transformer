#ifndef TOKENIZER_H
#define TOKENIZER_H

#define MAX_TOKEN_LEN 32
// 今回のミニ辞書に登録する単語数
#define VOCAB_SIZE_NEW 10

// 辞書の単語リスト
static const char *VOCAB_DICT[VOCAB_SIZE_NEW] = {
  "[PAD]",       // ID 0: 空欄を埋める特殊トークン
  "おはよ",     // ID 1
  "おやすみ",   // ID 2
  "元気",       // ID 3
  "？",         // ID 4
  "！",         // ID 5
  "うれしい",   // ID 6
  "かなしい",   // ID 7
  "バイバイ",   // ID 8
  "[UNK]"        // ID 9: 辞書にない未知の単語用
};

// テキスト (スペース区切り) をトークンIDの配列に変換する
// 例: "おはよ !" -> [1, 5, 0, 0]
void tokenize(const char *text, int *output_ids, int seq_len);

// トークンIDの配列を人間の読めるテキストに戻す
// 例: [1, 5, 0, 0] -> "おはよ ! "
void detokenize(const int *token_ids, int seq_len, char *output_text);

#endif // TOKENIZER_H