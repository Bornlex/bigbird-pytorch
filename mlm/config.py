"""Configuration for the encoder-only BigBird masked-LM model."""

from dataclasses import dataclass


@dataclass
class MLMConfig:
  """Hyperparameters for BigBird MLM. Defaults match BigBird-RoBERTa base."""

  # --- model ---
  vocab_size: int = 32000                 # tokenizer vocabulary size
  hidden_size: int = 768                  # embedding / hidden dimension
  num_hidden_layers: int = 12             # transformer encoder layers
  num_attention_heads: int = 12           # attention heads (hidden % heads == 0)
  intermediate_size: int = 3072           # feed-forward inner dimension
  hidden_act: str = "gelu"                # "gelu" (tanh approx) or "gelu_exact"
  hidden_dropout_prob: float = 0.1        # dropout on embeddings + FF/attn output
  attention_probs_dropout_prob: float = 0.1   # dropout on attention weights
  max_position_embeddings: int = 4096     # hard cap on sequence length
  use_bias: bool = True                   # bias on q/k/v and attention output
  use_gradient_checkpointing: bool = False    # trade compute for memory

  # --- attention ---
  attention_type: str = "block_sparse"    # "block_sparse" or "original_full"
  block_size: int = 16                    # tokens per block (sparsity granularity)
  num_rand_blocks: int = 3                # random blocks each query block attends to
  max_encoder_length: int = 1024          # run-time seq length (<= max_position_embeddings)

  # --- masking / pretraining ---
  masked_lm_prob: float = 0.15            # fraction of words to mask
  max_predictions_per_seq: int = 75       # max masked tokens per example
  substitute_newline: str = " "           # replace "\n" in raw text (None to keep)

  @property
  def head_size(self):
    return self.hidden_size // self.num_attention_heads
