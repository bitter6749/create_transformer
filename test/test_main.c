#include <stdio.h>

// 各テストファイルで定義されている関数を外部参照（extern）
extern void test_matrix_ops();
extern void test_softmax_layer();
extern void test_attention_layer();

int main() {
    printf("==============================================\n");
    printf("=== Transformer 分割ユニットテスト 起動中 ===\n");
    printf("==============================================\n\n");

    // 行列演算テスト
    test_matrix_ops();

    // 各層のロジックテスト
    test_softmax_layer();
    test_attention_layer();

    printf("==============================================\n");
    printf("===  すべてのテストに合格しました！ (GREEN)  ===\n");
    printf("==============================================\n");
    return 0;
}