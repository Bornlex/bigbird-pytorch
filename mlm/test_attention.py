"""Correctness test for block-sparse attention.

We check the linear-time kernel against an independent, obviously-correct dense
reference: full attention masked by the exact BigBird pattern (window + global +
random) built straight from the random-block adjacency. If they agree, the
kernel attends to exactly the intended keys and shares one softmax over them --
which also guards any refactor of the kernel.

Run: python -m mlm.test_attention
"""

import math

import torch
from torch import Tensor

from mlm.attention import (block_sparse_attention, build_rand_attn,
                           create_band_mask)
from mlm.config import MLMConfig


def bigbird_mask(
    rand_attn: Tensor,
    n: int,
    bs: int,
    num_heads: int,
) -> Tensor:
  """Dense [num_heads, n, n] boolean BigBird attention pattern."""
  nb = n // bs
  mask = torch.zeros(num_heads, n, n, dtype=torch.bool)
  for h in range(num_heads):
    for i in range(1, nb - 1):                              # non-global rows
      mask[h, i * bs:(i + 1) * bs, (i - 1) * bs:(i + 2) * bs] = True  # window
      for j in rand_attn[h, i - 1].tolist():                          # random
        mask[h, i * bs:(i + 1) * bs, j * bs:(j + 1) * bs] = True
    mask[h, :bs, :] = mask[h, -bs:, :] = True              # global rows
    mask[h, :, :bs] = mask[h, :, -bs:] = True              # global columns
  return mask


def dense_reference(q: Tensor, k: Tensor, v: Tensor, mask: Tensor) -> Tensor:
  d = q.shape[-1]
  scores = torch.einsum("bhqd,bhkd->bhqk", q, k) / math.sqrt(d)
  scores = scores.masked_fill(~mask.unsqueeze(0), float("-inf"))
  return torch.einsum("bhqk,bhkd->bhqd", scores.softmax(-1), v)


def test_matches_dense_reference():
  cfg = MLMConfig(hidden_size=64, num_attention_heads=2, num_hidden_layers=1,
                  block_size=16, num_rand_blocks=3, max_encoder_length=1024,
                  attention_type="block_sparse")
  h, n, d, bs = cfg.num_attention_heads, cfg.max_encoder_length, cfg.head_size, cfg.block_size
  b = 2

  rand_attn = build_rand_attn(cfg, seed=0)            # [h, nb-2, r]
  torch.manual_seed(0)
  q, k, v = (torch.randn(b, h, n, d, dtype=torch.float64) for _ in range(3))

  # All-ones masks -> no padding, so only the sparsity pattern is exercised.
  blocked = torch.ones(b, n // bs, bs, dtype=torch.float64)
  out = block_sparse_attention(
      q, k, v, create_band_mask(blocked),
      torch.ones(b, 1, n, 1, dtype=torch.float64),
      torch.ones(b, 1, 1, n, dtype=torch.float64),
      blocked, rand_attn, h, d, cfg.num_rand_blocks, n, bs)
  out = out.permute(0, 2, 1, 3)                       # [b, h, n, d]

  ref = dense_reference(q, k, v, bigbird_mask(rand_attn, n, bs, h))
  diff = (out - ref).abs().max().item()
  assert torch.allclose(out, ref, atol=1e-8), f"max diff {diff}"
  print(f"[ok] block-sparse == dense BigBird-masked attention (max diff {diff:.2e})")


if __name__ == "__main__":
  test_matches_dense_reference()
  print("\nAttention correctness test passed.")
