#include "model.h"
#include "transformer.h"
#include "mlp.h"

void init_model(SimpleTransformer *model) {
  // Embedding
  model->token_embedding = create_matrix(VOCAB_SIZE, EMBED_DIM);

  // Attention
  model->W_q = create_matrix(EMBED_DIM, EMBED_DIM);
  model->W_k = create_matrix(EMBED_DIM, EMBED_DIM);
  model->W_v = create_matrix(EMBED_DIM, EMBED_DIM);

  // MLP 
  model->W1 = create_matrix(EMBED_DIM, MLP_HIDDEN_DIM);
  model->b1 = create_matrix(1, MLP_HIDDEN_DIM);
  model->W2 = create_matrix(MLP_HIDDEN_DIM, EMBED_DIM);
  model->b2 = create_matrix(1, MLP_HIDDEN_DIM);

  // 逆伝播用ワークスペース
  // H: 1層目の出力 [SEQ_LEN x MLP_HIDDEN_DIM]
  model->H = create_matrix(SEQ_LEN, MLP_HIDDEN_DIM);

  // Output
  model->W_out = create_matrix(VOCAB_SIZE, EMBED_DIM);
}

void free_model(SimpleTransformer *model) {
  // Embedding
  free_matrix(&model->token_embedding); 

  // Attention
  free_matrix(&model->W_q);
  free_matrix(&model->W_k);
  free_matrix(&model->W_v);

  // MLP
  free_matrix(&model->W1);
  free_matrix(&model->b1);
  free_matrix(&model->W2);
  free_matrix(&model->b2);

  // 逆伝播用ワークスペース
  free_matrix(&model->H);

  // Output
  free_matrix(&model->W_out);
}