"""BigBird attention: block-sparse encoder self-attention and full attention
(decoder self/cross, with an incremental cache)."""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from transformer.config import TransformerConfig


def batched_index_gather(params: Tensor, indices: Tensor, batch_dims: int) -> Tensor:
  """tf.gather(params, indices, batch_dims=k, axis=k) in PyTorch.

  params:  [B_0..B_{k-1}, N, *E]
  indices: [B_0..B_{k-1}, *I]
  returns: [B_0..B_{k-1}, *I, *E]
  """
  batch_shape = list(params.shape[:batch_dims])
  n, e_shape = params.shape[batch_dims], list(params.shape[batch_dims + 1:])
  idx_shape = list(indices.shape[batch_dims:])
  bsz = math.prod(batch_shape) if batch_shape else 1
  isz = math.prod(idx_shape) if idx_shape else 1

  params_flat = params.reshape([bsz, n] + e_shape)
  indices_flat = indices.reshape(bsz, isz).long()
  rows = torch.arange(bsz, device=params.device).unsqueeze(1).expand(bsz, isz)
  gathered = params_flat[rows, indices_flat]
  return gathered.reshape(batch_shape + idx_shape + e_shape)


def build_rand_attn(config: TransformerConfig, seed: int) -> Tensor:
  """Random-block adjacency, shape [num_heads, num_blocks - 2, num_rand_blocks].

  Row `r` is query block `r + 1` (the two global blocks have no random row). Each
  query block picks `num_rand_blocks` key blocks spread across the sequence,
  excluding the global blocks (first/last) and its own sliding window so they are
  never double-counted in the softmax.
  """
  num_blocks = config.max_encoder_length // config.block_size
  r = config.num_rand_blocks
  gen = torch.Generator().manual_seed(seed)
  rand_attn = torch.zeros(config.num_attention_heads, num_blocks - 2, r,
                          dtype=torch.long)
  for head in range(config.num_attention_heads):
    for row in range(num_blocks - 2):
      i = row + 1
      window = {i - 1, i, i + 1}
      candidates = torch.tensor(
          [j for j in range(1, num_blocks - 1) if j not in window])
      pick = torch.randperm(candidates.numel(), generator=gen)[:r]
      rand_attn[head, row] = candidates[pick]
  return rand_attn


def create_band_mask(blocked_mask: Tensor) -> Tensor:
  """Window padding mask for interior blocks (the 3 sliding blocks)."""
  exp = torch.cat([blocked_mask[:, 1:-3], blocked_mask[:, 2:-2],
                   blocked_mask[:, 3:-1]], 2)
  band = torch.einsum("blq,blk->blqk", blocked_mask[:, 2:-2], exp)
  return band.unsqueeze(1)


def _create_rand_mask(
    blocked_mask: Tensor,
    rand_attn: Tensor,
    num_heads: int,
    num_rand_blocks: int,
    block_size: int,
) -> Tensor:
  """Padding mask for the gathered random blocks."""
  num_windows = blocked_mask.shape[1] - 2
  rand_mask = batched_index_gather(blocked_mask, rand_attn, batch_dims=1)
  rand_mask = rand_mask.reshape(-1, num_heads, num_windows,
                                num_rand_blocks * block_size)
  return torch.einsum("blq,bhlk->bhlqk", blocked_mask[:, 1:-1], rand_mask)


def block_sparse_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    band_mask: Tensor,
    from_mask: Tensor,
    to_mask: Tensor,
    blocked_mask: Tensor,
    rand_attn: Tensor,
    num_heads: int,
    head_size: int,
    num_rand_blocks: int,
    seq_length: int,
    block_size: int,
) -> Tensor:
  """BigBird attention = window + global + random, per block.

  q, k, v: [b, h, seq, d]. Returns [b, seq, h, d]. Every query block runs a
  single softmax over the union of the keys it attends to. The first/last query
  blocks are global (attend to all keys); blocks 1 and -2 sit next to the edge
  and get an explicit block set; interior blocks combine the three components.
  """
  b = q.shape[0]
  rand_attn = rand_attn.unsqueeze(0).expand(b, -1, -1, -1)
  rand_mask = _create_rand_mask(blocked_mask, rand_attn, num_heads,
                                num_rand_blocks, block_size)

  h, r, d = num_heads, num_rand_blocks, head_size
  nb, wn = seq_length // block_size, block_size
  scale = d ** -0.5

  bq = q.reshape(b, h, nb, wn, d)
  bk = k.reshape(b, h, nb, wn, d)
  bv = v.reshape(b, h, nb, wn, d)
  # Keys/values of the random blocks each interior query block attends to.
  gk = batched_index_gather(bk, rand_attn, 2).reshape(b, h, nb - 2, r * wn, d)
  gv = batched_index_gather(bv, rand_attn, 2).reshape(b, h, nb - 2, r * wn, d)

  def attend(
      q_block: Tensor,
      key_mat: Tensor,
      value_mat: Tensor,
      add_mask: Tensor,
  ) -> Tensor:
    """softmax(q.kᵀ * scale + mask) . v over one explicit set of key blocks."""
    scores = torch.einsum("bhqd,bhkd->bhqk", q_block, key_mat) * scale + add_mask
    ctx = torch.einsum("bhqk,bhkd->bhqd", scores.softmax(-1), value_mat)
    return ctx.unsqueeze(2)  # [b, h, 1, wn, d]

  # --- global rows: first and last query blocks attend to all keys ---
  first_ctx = attend(bq[:, :, 0], k, v, (1.0 - to_mask) * -10000.0)
  last_ctx = attend(bq[:, :, -1], k, v, (1.0 - to_mask) * -10000.0)

  # --- block 1: window {0,1,2} + global {last} + random ---
  second_k = torch.cat([bk[:, :, 0], bk[:, :, 1], bk[:, :, 2], bk[:, :, -1],
                        gk[:, :, 0]], 2)
  second_v = torch.cat([bv[:, :, 0], bv[:, :, 1], bv[:, :, 2], bv[:, :, -1],
                        gv[:, :, 0]], 2)
  seq_pad = torch.cat([to_mask[:, :, :, :3 * wn], to_mask[:, :, :, -wn:],
                       rand_mask.new_ones(b, 1, 1, r * wn)], 3)
  rand_pad = torch.cat([rand_mask.new_ones(b, h, wn, 4 * wn),
                        rand_mask[:, :, 0]], 3)
  second_ctx = attend(bq[:, :, 1], second_k, second_v,
                      (1.0 - torch.minimum(seq_pad, rand_pad)) * -10000.0)

  # --- block -2: global {0} + window {-3,-2,-1} + random ---
  sl_k = torch.cat([bk[:, :, 0], bk[:, :, -3], bk[:, :, -2], bk[:, :, -1],
                    gk[:, :, -1]], 2)
  sl_v = torch.cat([bv[:, :, 0], bv[:, :, -3], bv[:, :, -2], bv[:, :, -1],
                    gv[:, :, -1]], 2)
  sl_seq_pad = torch.cat([to_mask[:, :, :, :wn], to_mask[:, :, :, -3 * wn:],
                          rand_mask.new_ones(b, 1, 1, r * wn)], 3)
  sl_rand_pad = torch.cat([rand_mask.new_ones(b, h, wn, 4 * wn),
                           rand_mask[:, :, -1]], 3)
  sl_ctx = attend(bq[:, :, -2], sl_k, sl_v,
                  (1.0 - torch.minimum(sl_seq_pad, sl_rand_pad)) * -10000.0)

  # --- interior blocks (2 .. nb-3): window + global + random, one softmax ---
  mid_q = bq[:, :, 2:-2]
  # window: the 3 sliding blocks {i-1, i, i+1}
  window_k = torch.cat([bk[:, :, 1:-3], bk[:, :, 2:-2], bk[:, :, 3:-1]], 3)
  window_v = torch.cat([bv[:, :, 1:-3], bv[:, :, 2:-2], bv[:, :, 3:-1]], 3)
  window = torch.einsum("bhlqd,bhlkd->bhlqk", mid_q, window_k) * scale
  window += (1.0 - band_mask) * -10000.0
  # global: the first and last blocks
  g_first = torch.einsum("bhlqd,bhkd->bhlqk", mid_q, bk[:, :, 0]) * scale
  g_first += (1.0 - to_mask[:, :, :, :wn].unsqueeze(3)) * -10000.0
  g_last = torch.einsum("bhlqd,bhkd->bhlqk", mid_q, bk[:, :, -1]) * scale
  g_last += (1.0 - to_mask[:, :, :, -wn:].unsqueeze(3)) * -10000.0
  # random: the r blocks picked per query block
  rand = torch.einsum("bhlqd,bhlkd->bhlqk", mid_q, gk[:, :, 1:-1]) * scale
  rand += (1.0 - rand_mask[:, :, 1:-1]) * -10000.0
  # one shared softmax over [global-first | window | random | global-last]
  probs = torch.cat([g_first, window, rand, g_last], -1).softmax(-1)
  mid_ctx = torch.einsum("bhlqk,bhlkd->bhlqd", probs[..., wn:4 * wn], window_v)
  mid_ctx += torch.einsum("bhlqk,bhlkd->bhlqd", probs[..., 4 * wn:-wn], gv[:, :, 1:-1])
  mid_ctx += torch.einsum("bhlqk,bhkd->bhlqd", probs[..., :wn], bv[:, :, 0])
  mid_ctx += torch.einsum("bhlqk,bhkd->bhlqd", probs[..., -wn:], bv[:, :, -1])

  ctx = torch.cat([first_ctx, second_ctx, mid_ctx, sl_ctx, last_ctx], 2)
  ctx = ctx.reshape(b, h, seq_length, d) * from_mask
  return ctx.permute(0, 2, 1, 3)


class MultiHeadAttention(nn.Module):
  """Encoder self-attention: full (SDPA) or block-sparse."""

  def __init__(self, config: TransformerConfig, seed: int = 0):
    super().__init__()
    self.config = config
    self.h = config.num_attention_heads
    self.d = config.head_size
    hidden = config.hidden_size
    self.query = nn.Linear(hidden, hidden, bias=config.use_bias)
    self.key = nn.Linear(hidden, hidden, bias=config.use_bias)
    self.value = nn.Linear(hidden, hidden, bias=config.use_bias)
    self.dropout = config.attention_probs_dropout_prob
    if config.attention_type == "block_sparse":
      self.register_buffer("rand_attn", build_rand_attn(config, seed),
                           persistent=False)

  def _split(self, x: Tensor) -> Tensor:
    b, s, _ = x.shape
    return x.view(b, s, self.h, self.d).transpose(1, 2)  # [b, h, s, d]

  def forward(
      self,
      hidden: Tensor,
      band_mask: Optional[Tensor] = None,
      from_mask: Optional[Tensor] = None,
      to_mask: Optional[Tensor] = None,
      blocked_mask: Optional[Tensor] = None,
      attn_mask: Optional[Tensor] = None,
  ) -> Tensor:
    q = self._split(self.query(hidden))
    k = self._split(self.key(hidden))
    v = self._split(self.value(hidden))

    if self.config.attention_type == "block_sparse":
      ctx = block_sparse_attention(
          q, k, v, band_mask, from_mask, to_mask, blocked_mask, self.rand_attn,
          self.h, self.d, self.config.num_rand_blocks,
          self.config.max_encoder_length, self.config.block_size)
    else:
      p = self.dropout if self.training else 0.0
      ctx = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask,
                                           dropout_p=p)
      ctx = ctx.transpose(1, 2)  # [b, s, h, d]

    b, s = ctx.shape[0], ctx.shape[1]
    return ctx.reshape(b, s, self.h * self.d)


class FullAttention(nn.Module):
  """Decoder attention (self and cross), with an incremental key/value cache."""

  def __init__(self, config: TransformerConfig, seed: int = 0):
    super().__init__()
    self.h = config.num_attention_heads
    self.d = config.head_size
    hidden = config.hidden_size
    self.query = nn.Linear(hidden, hidden, bias=config.use_bias)
    self.key = nn.Linear(hidden, hidden, bias=config.use_bias)
    self.value = nn.Linear(hidden, hidden, bias=config.use_bias)
    self.dropout = config.attention_probs_dropout_prob

  def _split(self, x: Tensor) -> Tensor:
    b, s, _ = x.shape
    return x.view(b, s, self.h, self.d).transpose(1, 2)  # [b, h, s, d]

  def forward(
      self,
      from_tensor: Tensor,
      to_tensor: Tensor,
      attn_mask: Optional[Tensor] = None,
      cache: Optional[dict] = None,
      decode_i: Optional[int] = None,
  ) -> Tensor:
    q = self._split(self.query(from_tensor))
    k = self._split(self.key(to_tensor))
    v = self._split(self.value(to_tensor))

    if cache is not None and decode_i is not None:
      max_len = cache["k"].shape[2]
      sel = F.one_hot(torch.tensor(decode_i, device=q.device), max_len)
      sel = sel.to(q.dtype).view(1, 1, max_len, 1)
      k = cache["k"] + k * sel
      v = cache["v"] + v * sel
      cache["k"], cache["v"] = k, v

    p = self.dropout if self.training else 0.0
    ctx = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask,
                                         dropout_p=p)
    ctx = ctx.transpose(1, 2)
    b, s = ctx.shape[0], ctx.shape[1]
    return ctx.reshape(b, s, self.h * self.d)
