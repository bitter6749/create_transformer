#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// 設定
#define VOCAB_SIZE 27 // a~z + space
#define SEQ_LEN    4  // 入力する文字数(例: "cat " の四文字)
#define EMBED_DIM  16  // 文字を表現するベクトルの次元数 (RNN の HIDDENに相当)

// 行列の情報を表す構造体
typedef struct {
  float *data;  // ヒープ領域へのポインタ
  int rows;     // 行数 (縦のサイズ)
  int cols;     // 列数 (横のサイズ)
} Matrix;

typedef struct {
  float *token_embedding; // 文字をベクトルに変換する表 [VOCAB_SIZE x EMBED_DIM]
  // 【token_embedding [27 × 16] の中身のイメージ】
  // [ 0 〜 15マス目 ] : ID=0 ('a') を表す 16個の数字
  // [16 〜 31マス目 ] : ID=1 ('b') を表す 16個の数字
  // [32 〜 47マス目 ] : ID=2 ('c') を表す 16個の数字
  // ...
  // [416〜431マス目 ] : ID=26 (' ') を表す 16個の数字
  // ID=1 の 3マス目: 1 * 16 + 2
  // 一般化すると、 ID * EMBED_DIM + index

  // Attention (注意機構) 用の重み (Q, K, V) という3つの役割の行列
  float *W_q; // [EMBED_DIM x EMBED_DIM]
  float *W_k; // [EMBED_DIM x EMBED_DIM]
  float *W_v; // [EMBED_DIM x EMBED_DIM]

  // 出力層の重み
  float *W_out;   // ベクトルから文字の確立に戻す行列 [VOCAB_SIZE x EMBED_DIM]
} SimpleTransformer;

// ============================
// === 行列計算のヘルパー関数 ===
// ============================

// 行列のメモリを確保し、0で初期化する関数
Matrix create_matrix(int rows, int cols) {
  Matrix M;
  M.rows = rows;
  M.cols = cols;

  // calloc: メモリの確保と初期化を同時に行う
  M.data = (float *)calloc(rows * cols, sizeof(float));

  if (M.data == NULL) {
    fprintf(stderr, "Error: 行列のメモリの確保に失敗しました。");
    exit(EXIT_FAILURE);
  }

  return M;
}

// 行列のメモリを安全に開放する関数
void free_matrix(Matrix *M) {
  if (M->data != NULL) {
    free(M->data);
    M->data = NULL; // 二重開放 (Double Free) を防ぐ
  }

  M->rows = 0;
  M->cols = 0;
}
// 通常の行列掛け算 C = A ・ B
void mat_mul(const Matrix *A, const Matrix *B, Matrix *C) {
  // 行列のサイズから、計算可能かチェック
  if (A->cols != B->rows || C->rows != A->rows || C->cols != B->cols) {
    fprintf(stderr, "Error: 行列のサイズが不一致です。\n");
    exit(EXIT_FAILURE);
  }

  for (int A_row = 0; A_row < A->rows; A_row++) {
    for (int B_col = 0; B_col < B->cols; B_col++) {
      // 各行と列の内積
      float inner_product = 0.0f;

      for (int A_col = 0; A_col < A->cols; A_col++) {
        float a = A->data[A_row * A->cols + A_col];
        float b = B->data[A_col * B->cols + B_col];

        inner_product += a * b;
      }
      C->data[A_row * C->cols + B_col] = inner_product;
    }
  }
}

// 右側転置の行列掛け算 C = A ・ B^T
void mat_mul_b_trans(const Matrix *A, const Matrix *B, Matrix *C) {
  // Bは転置されるので、Aの列数とBの列数が一致すべき
  if (A->cols != B->cols || C->rows != A-> rows || C->cols != B->rows) {
    fprintf(stderr, "Error: 行列のサイズが不一致です。\n");
    exit(EXIT_FAILURE);
  }

  for (int A_row = 0; A_row < A->rows; A_row++) {
    for (int B_row = 0; B_row < B->rows; B_row++) {
      // 各行と列の内積
      float inner_product = 0.0f;

      for (int A_col = 0; A_col < A->cols; A_col++) {
        float a = A->data[A_row * A->cols + A_col];
        float b = B->data[B_row * B->cols + A_col];
        
        inner_product += a * b;
      }
      C->data[A_row * C->cols + B_row] = inner_product;
    }
  }
}

// 左側転置の行列掛け算 C = A^T ・ B
void mat_mul_a_trans(const Matrix *A, const Matrix *B, Matrix *C) {
  // Aは転置されるので、Aの行数とBの行数が一致すべき
  if (A->rows != B->rows || C->rows != A->cols || C->cols != B->cols) {
    fprintf(stderr, "Error: 行列のサイズが不一致です。\n");
    exit(EXIT_FAILURE);
  }

  for (int A_col; A_col < A->cols; A_col++) {
    for (int B_col; B_col < B->cols; B_col++) {
      // 各行の列と内積
      float inner_product = 0.0f;

      for (int A_row; A_row < A->rows; A_row++) {
        float a = A->data[A_row * A->cols + A_col];
        float b = B->data[A_row * B->cols + B_col];

        inner_product += a * b;
      }
      C->data[A_col * C->cols + B_col] = inner_product;
    }
  }
}


// プロトタイプ宣言
void forward_transformer(
  SimpleTransformer *model, 
  const int *input_ids, 
  float *output_probabilties
);
void backward_ouput(
  SimpleTransformer *model, 
  const float *output_probabilities, 
  int target_id, 
  const float *Z, 
  float *dW_out, 
  float*dZ
);
void backward_attention(
  const float *dZ,
  const float *A_scores,
  const float *V,
  float *dA,
  float *dV
);
void softmax(float *x, int size);

int main() {
  // 1. モデルの宣言とメモリ確保
  SimpleTransformer model;
  model.token_embedding = (float *)malloc(VOCAB_SIZE * EMBED_DIM * sizeof(float));
  model.W_q = (float *)malloc(EMBED_DIM * EMBED_DIM * sizeof(float));
  model.W_k = (float *)malloc(EMBED_DIM * EMBED_DIM * sizeof(float));
  model.W_v = (float *)malloc(EMBED_DIM * EMBED_DIM * sizeof(float));
  model.W_out = (float *)malloc(VOCAB_SIZE * EMBED_DIM * sizeof(float));

  // すべての値に適当初期値を入れる
  for (int i=0; i < VOCAB_SIZE * EMBED_DIM; i++) model.token_embedding[i] = 0.1f;
  for (int i=0; i < EMBED_DIM * EMBED_DIM; i++) {
    model.W_q[i] = 0.5f;
    model.W_k[i] = 0.5f;
    model.W_v[i] = 0.5f;
  }
  for (int i=0; i < VOCAB_SIZE * EMBED_DIM; i++) model.W_out[i] = 0.1f;

  // 2. 入力データ ("cat " を表すIDの配列)
  // c=2, a=0, t=19, space=26
  int input_ids[SEQ_LEN] = {2, 0, 19, 26};

  // 次の一文字の予測確率 (27文字分) を受け取るバッファ
  float *output_probabilties = (float *)calloc(VOCAB_SIZE, sizeof(float));

  // 3. 予測の実行
  printf("Transformerで次の文字を予測中...\n");
  forward_transformer(&model, input_ids, output_probabilties);

  // 4. 結果の表示 (最も確率が高い文字のIDを探す)
  int max_id = 0;
  float max_prob = output_probabilties[0];
  for (int i = 1; i < VOCAB_SIZE; i++) {
    if (output_probabilties[i] > max_prob) {
      max_prob = output_probabilties[i];
      max_id = i;
    }
  }
  printf("予測された次の文字のID: %d (確率: %.2f%%)\n", max_id, max_prob * 100.0f);

  // メモリの開放
  free(model.token_embedding); 
  free(model.W_q);
  free(model.W_k);
  free(model.W_v);
  free(model.W_out);
  free(output_probabilties);

  return 0;
}

void forward_transformer(SimpleTransformer *model, const int *input_ids, float *output_probabilties) {
  // 一時的なワークスペース (ヒープに確保)
  // X: 入力文字をベクトル化したもの [SEQ_LEN x EMBED_DIM]
  //
  //      { x0,0 x0,1 ... x0,15 }  <- 一文字目 'c' のベクトル
  //      { x1,0 x1,1 ... x1,15 }  <- 二文字目 'a' のベクトル
  // X =  |                     |
  //      { x2,0 x2,1 ... x2,15 }  <- 三文字目 't' のベクトル
  //      { x3,0 x3,1 ... x3,15 }  <- 四文字目 ' ' のベクトル
  float *X = (float *)malloc(SEQ_LEN * EMBED_DIM * sizeof(float));

  // Q, K, V ベクトル
  // Q: クエリ
  // K: キュー
  // V: バリュー
  float *Q = (float *)malloc(SEQ_LEN * EMBED_DIM * sizeof(float));
  float *K = (float *)malloc(SEQ_LEN * EMBED_DIM * sizeof(float));
  float *V = (float *)malloc(SEQ_LEN * EMBED_DIM * sizeof(float));

  // step1: Embedding (文字IDをベクトルに変換)
  for (int t = 0; t < SEQ_LEN; t++) {
    int id = input_ids[t];
    for (int d = 0; d < EMBED_DIM; d++) {
      X[t * EMBED_DIM + d] = model->token_embedding[id * EMBED_DIM + d];
    }
  }

  // step2: Q, K, V の計算 (行列の掛け算)
  for (int t = 0; t < SEQ_LEN; t++) {
    for (int i = 0; i < EMBED_DIM; i++) {
      float q_val = 0.0f, k_val = 0.0f, v_val = 0.0f;

      for (int j = 0; j < EMBED_DIM; j++) {
        float x = X[t * EMBED_DIM + j];
        q_val += x * model->W_q[i * EMBED_DIM + j];
        k_val += x * model->W_k[i * EMBED_DIM + j];
        v_val += x * model->W_v[i * EMBED_DIM + j];
      }
      Q[t * EMBED_DIM + i] = q_val;
      K[t * EMBED_DIM + i] = k_val;
      V[t * EMBED_DIM + i] = v_val;
    }
  }

  // step3: Attention (注意機構) の計算
  //
  //        { A0,0 A0,1 A0,2 A0,3 } <- 'c' から見た各文字への注目度
  //        { A1,0 A1,1 A1,2 A1,3 } <- 'a' から見た各文字への注目度
  //  A =   |                     |
  //        { A2,0 A2,1 A2,2 A2,3 } <- 't' から見た各文字への注目度
  //        { A3,0 A3,1 A3,2 A3,3 } <- ' ' から見た各文字への注目度
  //
  float A_scores[SEQ_LEN * SEQ_LEN] = {0};
  for (int i = 0; i < SEQ_LEN; i++) {
    for (int j = 0; j < SEQ_LEN; j++) {
      float score = 0.0f;
      for (int d = 0; d < EMBED_DIM; d++) {
        // Q, K は4行16列の行列
        // score は 4行4列
        // 4行16列 x 16行4列 = 4行4列 
        // score = Q ・ K^T (K の転置行列)
        // K はd行j列目とかけたいが、配列だとj行d列目と同じになる
         score += Q[i * EMBED_DIM + d] * K[j * EMBED_DIM + d]; // Q と K の内積
      }
      A_scores[i * SEQ_LEN + j] = score / sqrtf(EMBED_DIM); // スケール調整
    }
    // softmaxをかけて確率 (重み) にする
    softmax(&A_scores[i * SEQ_LEN], SEQ_LEN);
  }

  // 計算した関係性 (A_scores) を使って、V (価値)のベクトルをブレンドする
  // Z: Attention の出力 [SEQ_LEN x EMBED_DIM]
  //
  //  4行16列 = 4行4列 x 4行16列 
  //  Z = A ・ V
  //
  float *Z = (float *)calloc(SEQ_LEN * EMBED_DIM, sizeof(float));
  for (int i = 0; i < SEQ_LEN; i++) {
    for (int d = 0; d < EMBED_DIM; d++) {
      for (int j = 0; j < SEQ_LEN; j++) {
        float a = A_scores[i * SEQ_LEN + j];
        float v = V[j * EMBED_DIM + d];
        Z[i * EMBED_DIM + d] += a * v;
      }
    }
  }

  // step4: 出力層 (最後の文字の予想結果だけを使う)
  // 今回は「最後の文字 (t = SEQ_LEN - 1)」の次の文字を予測したいので、Zの最後の行だけを計算
  //
  // 1行27列 = 1行16列 x 16列27行
  // output = Z ・ W_out^T (W_out の転置行列)
  //
  // 全結合層
  float *last_z = &Z[(SEQ_LEN - 1) * EMBED_DIM];
  for (int i = 0; i < VOCAB_SIZE; i++) {
    float val = 0.0f;
    for (int d = 0; d < EMBED_DIM; d++) {
      float z = last_z[d];
      float w = model->W_out[i * EMBED_DIM + d];
      val += z * w;
    }

    output_probabilties[i] = val;
  }
  softmax(output_probabilties, VOCAB_SIZE); // 最終的な文字の確立分布にする


  // ワークスペースの解放
  free(X);
  free(Q);
  free(K);
  free(V);
  free(Z);
}

// ソフトマックス関数
void softmax(float *x, int size) {
  float max_val = x[0];
  for (int i = 1; i < size; i++) {
    if (x[i] > max_val) max_val = x[i];
  }

  float sum = 0.0f;
  for (int i = 0; i < size; i++) {
    x[i] = expf(x[i] - max_val);
    sum += x[i];
  }
  for (int i = 0; i < size; i++) {
    x[i] /= sum;
  }

}

// 出力層の逆伝播関数
void backward_ouput(
  SimpleTransformer *model, 
  const float *output_probabilities, // 順伝播の出力 [VOCAB_SIZE]
  int target_id,                     // 正解の文字ID
  const float *Z,                    // 順伝播の途中の行列 [SEQ_LEN x EMBED_DIM]
  float *dW_out,                     // W_out の勾配 [VOCAB_SIZE x EMBED_DIM]
  float*dZ                           // Zへの誤差の逆流 [SEQ_LEN x EMBED_DIM]
) {
  // 1. 生スコアの誤差を計算
  float d_logits[VOCAB_SIZE];
  for (int i = 0; i < VOCAB_SIZE; i++) {
    d_logits[i] = output_probabilities[i];
  }
  d_logits[target_id] -= 1.0f;

  // 2. W_out の修正メーターを計算
  const float *last_z = &Z[(SEQ_LEN - 1) * EMBED_DIM];  // 4文字目のベクトルを指す

  for (int i = 0; i < VOCAB_SIZE; i++) {
    for (int d = 0; d < EMBED_DIM; d++) {
      // 誤差 x 通ってきた値 (last_z) を勾配に蓄積
      dW_out[i * EMBED_DIM + d] += d_logits[i] * last_z[d];
    }
  }

  // 3. 行列 Z の最後の行 (4行目) への誤差を計算して、上流へ引き継ぐ
  float *dz_last = &dZ[(SEQ_LEN - 1) * EMBED_DIM];
  for (int i = 0; i < VOCAB_SIZE; i++) {
    for (int d = 0; d < EMBED_DIM; d++) {
      dz_last[d] += d_logits[i] * model->W_out[i * EMBED_DIM + d];
    }
  }
}

// 相性表の誤差dA
// dA = dZ ・ V^T (Vの転置行列)
// [4 x 4] = [4 x 16] x [16 x 4]
//
// 価値の誤差dV
// dV = A^T (Aの転置行列) ・ dZ
// [4 x 16] = [4 x 4] x [4 x 16]
void backward_attention(
  const float *dZ,
  const float *A_scores,
  const float *V,
  float *dA,
  float *dV
) {
  // dA = dZ ・ V^T の計算
  for (int i = 0; i < SEQ_LEN; i++) {
    for (int j = 0; j < SEQ_LEN; j++) {
      float sum = 0.0f;
      for (int d = 0; d < EMBED_DIM; d++) {
        sum += dZ[i * EMBED_DIM + d] * V[j * EMBED_DIM + d];
      }
      dA[i * SEQ_LEN + j] = sum;
    }
  }

  // dV = A^T ・ dZ の計算
  for (int i = 0; i < SEQ_LEN; i++) {
    for (int d = 0; d < EMBED_DIM; d++) {
      float sum = 0.0f;
      for (int j = 0; j < SEQ_LEN; j++) {
        sum += A_scores[j * SEQ_LEN + i] * dZ[j * EMBED_DIM + d];
      }

      dV[i * EMBED_DIM + d]  = sum;
    }
  }
}