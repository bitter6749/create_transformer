#ifndef MODEL_H
#define MODEL_H

#include "matrix.h"

// =====================================
// === ハイパーパラメーターのマクロ定義 ===
// =====================================

#define VOCAB_SIZE 27 // a~z + space
#define SEQ_LEN    4  // 入力する文字数(例: "cat " の四文字)
#define EMBED_DIM  16  // 文字を表現するベクトルの次元数 (RNN の HIDDENに相当)

// 隠れ層の次元数を設定 (EMBED_DIM の 4倍が一般的)
#define MLP_HIDDEN_DIM    (EMBED_DIM * 4) // 64
#define NUM_LAYERS        2 

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

  // LayerNorm (Attention層の手前用と、MLP層の手前用の2面 x 各層)
  Matrix ln1_gamma[NUM_LAYERS]; Matrix ln1_beta[NUM_LAYERS];
  Matrix ln2_gamma[NUM_LAYERS]; Matrix ln2_beta[NUM_LAYERS];

  // Attention (注意機構) 用の重み 
  // (Q, K, V) という3つの役割の行列
  Matrix W_q[NUM_LAYERS]; // [EMBED_DIM x EMBED_DIM]
  Matrix W_k[NUM_LAYERS]; // [EMBED_DIM x EMBED_DIM]
  Matrix W_v[NUM_LAYERS]; // [EMBED_DIM x EMBED_DIM]

  // MLP(多層パーセプトロン)用の重みとバイアス
  Matrix W1[NUM_LAYERS];  // [EMBED_DIM x MLP_HIDDEN_DIM] (次元を広げる)
  Matrix b1[NUM_LAYERS];  // [1 x MLP_HIDDEN_DIM]         (バイアスは1行)
  Matrix W2[NUM_LAYERS];  // [MLP_HIDDEN_DIM x EMBED_DIM] (次元を元に戻す)
  Matrix b2[NUM_LAYERS];  // [1 x EMBED_DIM]              (バイアスは１行)

  // 逆伝播 (ReLUの微分) のために順伝播の途中の値を保存するバッファ
  Matrix H[NUM_LAYERS];   // [SEQ_LEN x MLP_HIDDEN_DIM]

  // 出力層の重み
  // (ベクトルから文字の確立に戻す行列)
  Matrix W_out;   // [VOCAB_SIZE x EMBED_DIM] 
} SimpleTransformer;

// --- モデルのメモリ管理関数 ---
void init_model(SimpleTransformer *model);
void free_model(SimpleTransformer *model);

#endif // MODEL_H