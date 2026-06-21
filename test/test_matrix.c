#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "matrix.h"

void test_matrix_ops() {
    printf("[テスト] 行列演算関数 (matrix.c) の検証中...\n");

    // テスト用データの準備
    // A: 2行3列, B: 3行2列
    Matrix A = create_matrix(2, 3);
    Matrix B = create_matrix(3, 2);

    A.data[0] = 1.0f; A.data[1] = 2.0f; A.data[2] = 3.0f;
    A.data[3] = 4.0f; A.data[4] = 0.0f; A.data[5] = -1.0f;

    B.data[0] = 2.0f; B.data[1] = 1.0f;
    B.data[2] = -1.0f; B.data[3] = 0.0f;
    B.data[4] = 3.0f; B.data[5] = 4.0f;

    // ----------------------------------------------------------------
    // 検証1: 通常の行列掛け算 (mat_mul) -> C = A * B [2x2]
    // PyTorch正解値: [[9, 13], [5, 0]]
    // ----------------------------------------------------------------
    Matrix C1 = create_matrix(2, 2);
    mat_mul(&A, &B, &C1);

    float expected_c1[4] = {9.0f, 13.0f, 5.0f, 0.0f};
    for (int i = 0; i < 4; i++) {
        if (fabsf(C1.data[i] - expected_c1[i]) > 1e-5f) {
            fprintf(stderr, "【失敗】mat_mul の結果が不一致です。 index %d: Got %f, Expected %f\n", i, C1.data[i], expected_c1[i]);
            exit(EXIT_FAILURE);
        }
    }
    printf("  => mat_mul (通常掛け算): 合格\n");

    // ----------------------------------------------------------------
    // 検証2: 右側転置の掛け算 (mat_mul_b_trans) -> C = A * A^T [2x2]
    // PyTorch正解値: [[14, 1], [1, 17]]
    // ----------------------------------------------------------------
    Matrix C2 = create_matrix(2, 2);
    mat_mul_b_trans(&A, &A, &C2);

    float expected_c2[4] = {14.0f, 1.0f, 1.0f, 17.0f};
    for (int i = 0; i < 4; i++) {
        if (fabsf(C2.data[i] - expected_c2[i]) > 1e-5f) {
            fprintf(stderr, "【失敗】mat_mul_b_trans の結果が不一致です。 index %d: Got %f, Expected %f\n", i, C2.data[i], expected_c2[i]);
            exit(EXIT_FAILURE);
        }
    }
    printf("  => mat_mul_b_trans (右側転置): 合格\n");

    // ----------------------------------------------------------------
    // 検証3: 左側転置の掛け算 (mat_mul_a_trans) -> C = B^T * B^T^T = B^T * B [2x2]
    // PyTorch正解値: [[14, 14], [14, 17]]
    // ----------------------------------------------------------------
    Matrix C3 = create_matrix(2, 2);
    mat_mul_a_trans(&B, &B, &C3);

    float expected_c3[4] = {14.0f, 14.0f, 14.0f, 17.0f};
    for (int i = 0; i < 4; i++) {
        if (fabsf(C3.data[i] - expected_c3[i]) > 1e-5f) {
            fprintf(stderr, "【失敗】mat_mul_a_trans の結果が不一致です。 index %d: Got %f, Expected %f\n", i, C3.data[i], expected_c3[i]);
            exit(EXIT_FAILURE);
        }
    }
    printf("  => mat_mul_a_trans (左側転置): 合格\n");

    // 解放
    free_matrix(&A); free_matrix(&B);
    free_matrix(&C1); free_matrix(&C2); free_matrix(&C3);
    printf("  => 【合格】すべての基本行列演算テストを通過しました。\n\n");
}