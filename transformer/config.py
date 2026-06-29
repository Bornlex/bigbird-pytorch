"""Configuration for the encoder-decoder BigBird (seq2seq) model."""

from dataclasses import dataclass


@dataclass
class TransformerConfig:
  """Hyperparameters for seq2seq BigBird (Pegasus/BART-style encoder-decoder)."""

  # --- model ---
  vocab_size: int = 32000                 # tokenizer vocabulary size
  hidden_size: int = 768                  # embedding / hidden dimension
  num_hidden_layers: int = 12             # encoder layers (decoder: see below)
  num_attention_heads: int = 12           # attention heads (hidden % heads == 0)
  intermediate_size: int = 3072           # feed-forward inner dimension
  hidden_act: str = "gelu"                # "gelu" (tanh approx) or "gelu_exact"
  hidden_dropout_prob: float = 0.1        # dropout on embeddings + FF/attn output
  attention_probs_dropout_prob: float = 0.1   # dropout on attention weights
  max_position_embeddings: int = 4096     # hard cap on sequence length
  use_bias: bool = True                   # bias on q/k/v and attention output
  norm_type: str = "postnorm"             # "postnorm" (BERT) or "prenorm" (Pegasus)
  rescale_embedding: bool = False         # scale embeddings by sqrt(hidden_size)
  use_gradient_checkpointing: bool = False    # trade compute for memory

  # --- encoder attention ---
  attention_type: str = "block_sparse"    # "block_sparse" or "original_full"
  block_size: int = 16                    # tokens per block (sparsity granularity)
  num_rand_blocks: int = 3                # random blocks each query block attends to
  max_encoder_length: int = 1024          # run-time encoder length

  # --- decoder / generation ---
  max_decoder_length: int = 64            # max target length (full causal attn)
  num_decoder_layers: int = None          # None -> same as num_hidden_layers
  couple_encoder_decoder: bool = False    # share encoder weights with the decoder
  beam_size: int = 5                      # beam width for generation
  alpha: float = 0.7                      # beam length penalty (0 short .. 1 long)

  @property
  def head_size(self):
    return self.hidden_size // self.num_attention_heads

  @property
  def decoder_layers(self):
    return self.num_decoder_layers or self.num_hidden_layers
