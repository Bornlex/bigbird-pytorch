"""Configuration for the encoder-decoder BigBird (seq2seq) model."""

from dataclasses import dataclass


@dataclass
class TransformerConfig:
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
  initializer_range: float = 0.02
  use_bias: bool = True
  norm_type: str = "postnorm"            # "postnorm" or "prenorm" (Pegasus)
  rescale_embedding: bool = False
  use_gradient_checkpointing: bool = False

  # encoder attention
  attention_type: str = "block_sparse"   # "block_sparse" or "original_full"
  block_size: int = 16
  num_rand_blocks: int = 3
  max_encoder_length: int = 1024

  # decoder / generation
  max_decoder_length: int = 64
  num_decoder_layers: int = None         # defaults to num_hidden_layers
  couple_encoder_decoder: bool = False
  beam_size: int = 5
  alpha: float = 0.7                      # length penalty

  @property
  def head_size(self):
    return self.hidden_size // self.num_attention_heads

  @property
  def decoder_layers(self):
    return self.num_decoder_layers or self.num_hidden_layers
