#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "transformer.h"

void test_softmax_layer() {
    printf("[テスト] Softmax層の単体検証中...\n");

    float x[3] = {1.0f, 2.0f, 3.0f};
    float expected_softmax[3] = {0.090031f, 0.244728f, 0.665241f};
    
    softmax(x, 3);

    for (int i = 0; i < 3; i++) {
        if (fabsf(x[i] - expected_softmax[i]) > 1e-5f) {
            fprintf(stderr, "【失敗】softmax の結果が不一致です。 index %d\n", i);
            exit(EXIT_FAILURE);
        }
    }
    printf("  => 【合格】Softmax単体ロジック通過。\n\n");
}