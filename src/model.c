#include "model.h"
#include "transformer.h"
#include "mlp.h"

void init_model(SimpleTransformer *model) {
  // Embedding
  model->token_embedding = create_matrix(VOCAB_SIZE, EMBED_DIM);

  for (int l = 0; l < NUM_LAYERS; l++) {
    // LayerNorm1 (Attention手前用) の確保
    model->ln1_gamma[l] = create_matrix(1, EMBED_DIM);
    model->ln1_beta[l]  = create_matrix(1, EMBED_DIM);

    // 初期値設定 (gammaは1.0, betaは0.0)
    for (int j = 0; j < EMBED_DIM; j++) {
      model->ln1_gamma[l].data[j] = 1.0f;
      model->ln1_beta[l].data[j] = 0.0f;
    }

    // LayerNorm2 (MLP手前用) の確保
    model->ln2_gamma[l] = create_matrix(1, EMBED_DIM);
    model->ln2_beta[l]  = create_matrix(1, EMBED_DIM);

    // 初期値設定 (gammaは1.0, betaは0.0)
    for (int j = 0; j < EMBED_DIM; j++) {
      model->ln2_gamma[l].data[j] = 1.0f;
      model->ln2_beta[l].data[j] = 0.0f;
    }

    // Attention層
    model->W_q[l] = create_matrix(EMBED_DIM, EMBED_DIM);
    model->W_k[l] = create_matrix(EMBED_DIM, EMBED_DIM);
    model->W_v[l] = create_matrix(EMBED_DIM, EMBED_DIM);
  
    // MLP 
    model->W1[l] = create_matrix(EMBED_DIM, MLP_HIDDEN_DIM);
    model->b1[l]= create_matrix(1, MLP_HIDDEN_DIM);
    model->W2[l] = create_matrix(MLP_HIDDEN_DIM, EMBED_DIM);
    model->b2[l] = create_matrix(1, EMBED_DIM);

    // 逆伝播用ワークスペース
    // H: 1層目の出力 [SEQ_LEN x MLP_HIDDEN_DIM]
    model->H[l] = create_matrix(SEQ_LEN, MLP_HIDDEN_DIM);
  }

  // Output (W_out): forward_lm_head 内の mat_mul(&last_z, W_out, ...)
  // last_z が [1 x EMBED_DIM] なので、W_out は [EMBED_DIM x VOCAB_SIZE]
  model->W_out = create_matrix(EMBED_DIM, VOCAB_SIZE);
}

void free_model(SimpleTransformer *model) {
  // Embedding
  free_matrix(&model->token_embedding); 

  for (int l = 0; l < NUM_LAYERS; l++) {
    free_matrix(&model->ln1_gamma[l]);
    free_matrix(&model->ln1_beta[l]);
    free_matrix(&model->ln2_gamma[l]);
    free_matrix(&model->ln2_beta[l]);

    free_matrix(&model->W_q[l]);
    free_matrix(&model->W_k[l]);
    free_matrix(&model->W_v[l]);

    free_matrix(&model->W1[l]);
    free_matrix(&model->b1[l]);
    free_matrix(&model->W2[l]);
    free_matrix(&model->b2[l]);

    free_matrix(&model->H[l]);
  }

  // Output
  free_matrix(&model->W_out);
}