#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "matrix.h"
#include "transformer.h"

// 将来 transformer.h に追加する関数のプロトタイプ宣言（テスト用に先出し）
void backward_attention_qkv(
    const Matrix *dA, 
    const Matrix *A_prob, 
    const Matrix *Q, 
    const Matrix *K, 
    Matrix *dQ, 
    Matrix *dK
);

void test_backward_attention_qkv(void) {
    printf("[テスト] Attention逆伝播の検証中...\n");

    int seq_len = 2;
    int embed_dim = 3;

    // 1. 順伝播の途中の値を再現 (テスト1,2と同じ値)
    Matrix Q = create_matrix(seq_len, embed_dim);
    Matrix K = create_matrix(seq_len, embed_dim);
    Matrix A_prob = create_matrix(seq_len, seq_len);

    Q.data[0] = 0.1f; Q.data[1] = 0.2f; Q.data[2] = 0.3f;
    Q.data[3] = 0.4f; Q.data[4] = 0.5f; Q.data[5] = 0.6f;

    K.data[0] = 0.7f; K.data[1] = 0.8f; K.data[2] = 0.9f;
    K.data[3] = 0.1f; K.data[4] = 0.2f; K.data[5] = 0.3f;

    // 前回のテストで合格した順伝播の出力（A_prob）
    A_prob.data[0] = 0.551775f; A_prob.data[1] = 0.448225f;
    A_prob.data[2] = 0.627058f; A_prob.data[3] = 0.372942f;

    // 2. 上流から流れてきた「相性表への誤差 (dA)」をセット
    Matrix dA = create_matrix(seq_len, seq_len);
    dA.data[0] = 1.0f; dA.data[1] = -1.0f;
    dA.data[2] = 0.5f; dA.data[3] =  0.5f;

    // 3. 自作関数が計算を格納する箱を用意
    Matrix dQ = create_matrix(seq_len, embed_dim);
    Matrix dK = create_matrix(seq_len, embed_dim);

    // 4. PyTorchで backward() をさせて算出した「絶対的な正解の壁」
    float expected_dQ[6] = {
        0.171348f,  0.171348f,  0.171348f,
        0.000000f,  0.000000f,  0.000000f
    };
    float expected_dK[6] = {
        0.028558f,  0.057116f,  0.085674f,
        -0.028558f,  -0.057116f, -0.085674f
    };

    // 5. 逆伝播ロジック（次ステップで実装）を実行
    backward_attention_qkv(&dA, &A_prob, &Q, &K, &dQ, &dK);

    // 6. 検証: dQ のチェック
    for (int i = 0; i < 6; i++) {
        if (fabsf(dQ.data[i] - expected_dQ[i]) > 1e-4f) {
            fprintf(stderr, "【失敗】dQ の計算結果が不一致です。 index %d: Got %f, Expected %f\n", i, dQ.data[i], expected_dQ[i]);
            exit(EXIT_FAILURE);
        }
    }
    // 検証: dK のチェック
    for (int i = 0; i < 6; i++) {
        if (fabsf(dK.data[i] - expected_dK[i]) > 1e-4f) {
            fprintf(stderr, "【失敗】dK の計算結果が不一致です。 index %d: Got %f, Expected %f\n", i, dK.data[i], expected_dK[i]);
            exit(EXIT_FAILURE);
        }
    }

    // 解放
    free_matrix(&Q); free_matrix(&K); free_matrix(&A_prob);
    free_matrix(&dA); free_matrix(&dQ); free_matrix(&dK);

    printf("  => 【合格】Attention逆伝播（Q, Kへの逆流）テスト通過！\n\n");
}