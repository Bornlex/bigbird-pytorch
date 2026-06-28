"""Configuration for the encoder-only BigBird masked-LM model."""

from dataclasses import dataclass


@dataclass
class MLMConfig:
  # model
  vocab_size: int = 32000
  hidden_size: int = 768
  num_hidden_layers: int = 12
  num_attention_heads: int = 12
  intermediate_size: int = 3072
  hidden_act: str = "gelu"
  hidden_dropout_prob: float = 0.1
  attention_probs_dropout_prob: float = 0.1
  max_position_embeddings: int = 4096
  type_vocab_size: int = 2
  initializer_range: float = 0.02
  use_bias: bool = True
  norm_type: str = "postnorm"           # "postnorm" (BERT) or "prenorm"
  use_gradient_checkpointing: bool = False

  # attention
  attention_type: str = "block_sparse"  # "block_sparse" or "original_full"
  block_size: int = 16
  num_rand_blocks: int = 3
  max_encoder_length: int = 1024

  # masking / pretraining
  masked_lm_prob: float = 0.15
  max_predictions_per_seq: int = 75
  substitute_newline: str = " "

  @property
  def head_size(self):
    return self.hidden_size // self.num_attention_heads
