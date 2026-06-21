#include <math.h>
#include "ops.h"

// ソフトマックス関数
void softmax_row(Matrix *m, int row) {
  int cols = m->cols;
  float *row_data = &m->data[row * cols];

  // 最大値の探索 (オーバーフロー対策)
  float max_val = row_data[0];
  for (int i = 1; i < cols; i++) {
    if (row_data[i] > max_val) max_val = row_data[i];
  }

  // 指数関数の計算と総和
  float sum = 0.0f;
  for (int i = 0; i < cols; i++) {
    row_data[i] = expf(row_data[i] - max_val);
    sum += row_data[i];
  }

  // 正規化
  for (int i = 0; i < cols; i++) {
    row_data[i] /= sum;
  }
}