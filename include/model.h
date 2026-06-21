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

  // Attention (注意機構) 用の重み (Q, K, V) という3つの役割の行列
  Matrix W_q; // [EMBED_DIM x EMBED_DIM]
  Matrix W_k; // [EMBED_DIM x EMBED_DIM]
  Matrix W_v; // [EMBED_DIM x EMBED_DIM]

  // 出力層の重み
  Matrix W_out;   // ベクトルから文字の確立に戻す行列 [VOCAB_SIZE x EMBED_DIM]
} SimpleTransformer;

// --- モデルのメモリ管理関数 ---
void init_model(SimpleTransformer *model);
void free_model(SimpleTransformer *model);

#endif // MODEL_H