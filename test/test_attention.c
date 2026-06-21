#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "matrix.h"
#include "transformer.h"

void test_attention_layer() {
    printf("[テスト] Attention層（順伝播スコア）の検証中...\n");

    Matrix Q = create_matrix(2, 3);
    Matrix K = create_matrix(2, 3);

    Q.data[0] = 0.1f; Q.data[1] = 0.2f; Q.data[2] = 0.3f;
    Q.data[3] = 0.4f; Q.data[4] = 0.5f; Q.data[5] = 0.6f;

    K.data[0] = 0.7f; K.data[1] = 0.8f; K.data[2] = 0.9f;
    K.data[3] = 0.1f; K.data[4] = 0.2f; K.data[5] = 0.3f;

    // PyTorch正解値: torch.softmax(torch.matmul(Q, K.t()) / math.sqrt(3), dim=-1)
    float expected_A[4] = {
        0.551775, 0.448225,
        0.627058f, 0.372942f
    };

    Matrix A_scores = create_matrix(2, 2);
    mat_mul_b_trans(&Q, &K, &A_scores);

    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            A_scores.data[i * 2 + j] /= sqrtf(3); // EMBED_DIM = 3
        }
        softmax(&A_scores.data[i * 2], 2);
    }

    for (int i = 0; i < 4; i++) {
        if (fabsf(A_scores.data[i] - expected_A[i]) > 1e-5f) {
            fprintf(stderr, "【失敗】Attentionスコアの結果が不一致です。\n");
            fprintf(stderr, "index %d: Got %f, Expected %f\n", i, A_scores.data[i], expected_A[i]);
            free_matrix(&Q); free_matrix(&K); free_matrix(&A_scores);
            exit(EXIT_FAILURE);
        }
    }
    
    free_matrix(&Q); free_matrix(&K); free_matrix(&A_scores);
    printf("  => 【合格】Attention順伝播ロジック通過。\n\n");
}