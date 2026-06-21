#include "model.h"
#include "transformer.h"

void init_model(SimpleTransformer *model) {
  model->token_embedding = create_matrix(VOCAB_SIZE, EMBED_DIM);
  model->W_q = create_matrix(EMBED_DIM, EMBED_DIM);
  model->W_k = create_matrix(EMBED_DIM, EMBED_DIM);
  model->W_v = create_matrix(EMBED_DIM, EMBED_DIM);
  model->W_out = create_matrix(VOCAB_SIZE, EMBED_DIM);
}

void free_model(SimpleTransformer *model) {
  free_matrix(&model->token_embedding); 
  free_matrix(&model->W_q);
  free_matrix(&model->W_k);
  free_matrix(&model->W_v);
  free_matrix(&model->W_out);
}