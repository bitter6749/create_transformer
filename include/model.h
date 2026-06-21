#ifndef MODEL_H
#define MODEL_H

#include "matrix.h"

typedef struct {
  Matrix token_embedding; // 文字をベクトルに変換する表 [VOCAB_SIZE x EMBED_DIM]
  // 【token_embedding [27 × 16] の中身のイメージ】
  // [ 0 〜 15マス目 ] : ID=0 ('a') を表す 16個の数字
  // [16 〜 31マス目 ] : ID=1 ('b') を表す 16個の数字
  // [32 〜 47マス目 ] : ID=2 ('c') を表す 16個の数字
  // ...
  // [416〜431マス目 ] : ID=26 (' ') を表す 16個の数字
  // ID=1 の 3マス目: 1 * 16 + 2
  // 一般化すると、 ID * EMBED_DIM + index

  // Attention (注意機構) 用の重み 
  // (Q, K, V) という3つの役割の行列
  Matrix W_q; // [EMBED_DIM x EMBED_DIM]
  Matrix W_k; // [EMBED_DIM x EMBED_DIM]
  Matrix W_v; // [EMBED_DIM x EMBED_DIM]

  // MLP(多層パーセプトロン)用の重みとバイアス
  Matrix W1;  // [EMBED_DIM x MLP_HIDDEN_DIM] (次元を広げる)
  Matrix b1;  // [1 x MLP_HIDDEN_DIM]         (バイアスは1行)
  Matrix W2;  // [MLP_HIDDEN_DIM x EMBED_DIM] (次元を元に戻す)
  Matrix b2;  // [1 x EMBED_DIM]              (バイアスは１行)

  // 逆伝播 (ReLUの微分) のために順伝播の途中の値を保存するバッファ
  Matrix H;   // [SEQ_LEN x MLP_HIDDEN_DIM]

  // 出力層の重み
  // (ベクトルから文字の確立に戻す行列)
  Matrix W_out;   // [VOCAB_SIZE x EMBED_DIM] 
} SimpleTransformer;

// --- モデルのメモリ管理関数 ---
void init_model(SimpleTransformer *model);
void free_model(SimpleTransformer *model);

#endif // MODEL_H