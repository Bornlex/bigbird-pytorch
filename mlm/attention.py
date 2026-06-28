"""BigBird attention: full (via SDPA) and linear-time block-sparse."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

MAX_SEQ_LEN = 4096


def batched_index_gather(params, indices, batch_dims):
  """tf.gather(params, indices, batch_dims=k, axis=k) in PyTorch.

  params:  [B_0..B_{k-1}, N, *E]
  indices: [B_0..B_{k-1}, *I]
  returns: [B_0..B_{k-1}, *I, *E]
  """
  batch_shape = list(params.shape[:batch_dims])
  n, e_shape = params.shape[batch_dims], list(params.shape[batch_dims + 1:])
  idx_shape = list(indices.shape[batch_dims:])
  bsz = int(np.prod(batch_shape)) if batch_shape else 1
  isz = int(np.prod(idx_shape)) if idx_shape else 1

  params_flat = params.reshape([bsz, n] + e_shape)
  indices_flat = indices.reshape(bsz, isz).long()
  rows = torch.arange(bsz, device=params.device).unsqueeze(1).expand(bsz, isz)
  gathered = params_flat[rows, indices_flat]
  return gathered.reshape(batch_shape + idx_shape + e_shape)


# --------------------------------------------------------------------------- #
# Random-block adjacency construction (numpy, run once at init).
# --------------------------------------------------------------------------- #

def _single_block_row_attention(block_id, to_start, to_end, num_rand_blocks,
                                window_left=1, window_right=1,
                                global_left=1, global_right=1):
  to_blocks = np.arange(to_start, to_end, dtype=np.int32)
  perm = np.random.permutation(to_blocks)

  illegal = list(range(block_id - window_left, block_id + window_right + 1))
  illegal += list(range(global_left))
  illegal += list(range(to_end - global_right, to_end))
  if block_id == 1:
    illegal.append(to_end - 2)
  if block_id == to_end - 2:
    illegal.append(1)

  selected = []
  for i in range(to_end - to_start):
    if perm[i] not in illegal:
      selected.append(perm[i])
    if len(selected) == num_rand_blocks:
      break
  return np.array(selected, dtype=np.int32)


def _rand_attn_plan(from_seq_length, from_block_size, num_rand_blocks):
  lengths, counts = [], []
  num_blocks = from_seq_length // from_block_size
  if 2 * num_rand_blocks + 5 < num_blocks:
    lengths += [(2 * num_rand_blocks + 5) * from_block_size, from_seq_length]
    counts += [num_rand_blocks, 0]
  elif num_rand_blocks + 5 < num_blocks:
    lengths += [(num_rand_blocks + 5) * from_block_size, from_seq_length]
    counts += [num_rand_blocks // 2, num_rand_blocks - num_rand_blocks // 2]
  else:
    lengths += [from_seq_length]
    counts += [num_rand_blocks]
  return lengths, counts


def _block_rand_mask_with_head(seq_length, block_size, num_heads,
                               plan_from_length, plan_num_rand_blocks,
                               global_top=1, global_bottom=1):
  num_blocks = seq_length // block_size
  plan_block_length = np.array(plan_from_length) // block_size
  max_plan_idx = plan_from_length.index(seq_length)
  rand_attn = [
      np.zeros((num_blocks, np.sum(plan_num_rand_blocks[:max_plan_idx + 1])),
               dtype=np.int32) for _ in range(num_heads)
  ]

  for plan_idx in range(max_plan_idx + 1):
    if plan_idx > 0:
      if plan_num_rand_blocks[plan_idx] > 0:
        a = int(np.sum(plan_num_rand_blocks[:plan_idx]))
        b = int(np.sum(plan_num_rand_blocks[:plan_idx + 1]))
        for blk in range(global_top, plan_block_length[plan_idx - 1]):
          for h in range(num_heads):
            rand_attn[h][blk, a:b] = _single_block_row_attention(
                blk, plan_block_length[plan_idx - 1],
                plan_block_length[plan_idx], plan_num_rand_blocks[plan_idx])
      for pl_id in range(plan_idx):
        if plan_num_rand_blocks[pl_id] == 0:
          continue
        for blk in range(plan_block_length[plan_idx - 1],
                         plan_block_length[plan_idx]):
          a, to_start = 0, 0
          if pl_id > 0:
            a = int(np.sum(plan_num_rand_blocks[:pl_id]))
            to_start = plan_block_length[pl_id - 1]
          b = int(np.sum(plan_num_rand_blocks[:pl_id + 1]))
          for h in range(num_heads):
            rand_attn[h][blk, a:b] = _single_block_row_attention(
                blk, to_start, plan_block_length[pl_id],
                plan_num_rand_blocks[pl_id])

    if plan_num_rand_blocks[plan_idx] == 0:
      continue
    b = int(np.sum(plan_num_rand_blocks[:plan_idx + 1]))
    from_start, to_start, a = global_top, 0, 0
    if plan_idx > 0:
      a = int(np.sum(plan_num_rand_blocks[:plan_idx]))
      from_start = to_start = plan_block_length[plan_idx - 1]
    for blk in range(from_start, plan_block_length[plan_idx]):
      for h in range(num_heads):
        rand_attn[h][blk, a:b] = _single_block_row_attention(
            blk, to_start, plan_block_length[plan_idx],
            plan_num_rand_blocks[plan_idx])

  return [r[global_top:num_blocks - global_bottom, :] for r in rand_attn]


def _block_rand_mask(from_seq_length, to_seq_length, from_block_size,
                     to_block_size, num_rand_blocks, last_idx=-1):
  rand_attn = np.zeros(
      (from_seq_length // from_block_size - 2, num_rand_blocks), dtype=np.int32)
  middle = np.arange(1, to_seq_length // to_block_size - 1, dtype=np.int32)
  last = to_seq_length // to_block_size - 1
  if last_idx > 2 * to_block_size:
    last = (last_idx // to_block_size) - 1

  r = num_rand_blocks
  nblk = from_seq_length // from_block_size
  for i in range(1, nblk - 1):
    start, end = i - 2, i
    if i == 1:
      rand_attn[i - 1] = np.random.permutation(middle[2:last])[:r]
    elif i == 2:
      rand_attn[i - 1] = np.random.permutation(middle[3:last])[:r]
    elif i in (nblk - 3, nblk - 2):
      rand_attn[i - 1] = np.random.permutation(middle[:last])[:r]
    elif start > last:
      rand_attn[i - 1] = np.random.permutation(middle[:last])[:r]
    elif end + 1 == last:
      rand_attn[i - 1] = np.random.permutation(middle[:start])[:r]
    else:
      rand_attn[i - 1] = np.random.permutation(
          np.concatenate((middle[:start], middle[end + 1:last])))[:r]
  return rand_attn


def build_rand_attn(config, seed):
  """Per-head random-block adjacency, shape [num_heads, num_blocks-2, r]."""
  np.random.seed(seed)
  n = config.max_encoder_length
  bs = config.block_size
  if n in (1024, 2048, 3072, 4096):
    rand_attn = [
        _block_rand_mask(MAX_SEQ_LEN, MAX_SEQ_LEN, bs, bs,
                         config.num_rand_blocks, last_idx=1024)[:(n // bs - 2)]
        for _ in range(config.num_attention_heads)
    ]
  else:
    plan_len, plan_cnt = _rand_attn_plan(n, bs, config.num_rand_blocks)
    rand_attn = _block_rand_mask_with_head(
        n, bs, config.num_attention_heads, plan_len, plan_cnt)
  return torch.tensor(np.stack(rand_attn, 0), dtype=torch.long)


# --------------------------------------------------------------------------- #
# Mask helpers (tensor).
# --------------------------------------------------------------------------- #

def create_band_mask(blocked_mask):
  exp = torch.cat([blocked_mask[:, 1:-3], blocked_mask[:, 2:-2],
                   blocked_mask[:, 3:-1]], 2)
  band = torch.einsum("blq,blk->blqk", blocked_mask[:, 2:-2], exp)
  return band.unsqueeze(1)


def _create_rand_mask(blocked_mask, rand_attn, num_heads, r, from_block_size):
  num_windows = blocked_mask.shape[1] - 2
  rand_mask = batched_index_gather(blocked_mask, rand_attn, batch_dims=1)
  rand_mask = rand_mask.reshape(-1, num_heads, num_windows, r * from_block_size)
  return torch.einsum("blq,bhlk->bhlqk", blocked_mask[:, 1:-1], rand_mask)


# --------------------------------------------------------------------------- #
# Block-sparse attention kernel.
# --------------------------------------------------------------------------- #

def block_sparse_attention(q, k, v, band_mask, from_mask, to_mask,
                           blocked_mask, rand_attn, num_heads, head_size,
                           num_rand_blocks, seq_length, block_size):
  """q, k, v: [b, h, seq, d]. Returns [b, seq, h, d]."""
  b = q.shape[0]
  rand_attn = rand_attn.unsqueeze(0).expand(b, -1, -1, -1)
  rand_mask = _create_rand_mask(
      blocked_mask, rand_attn, num_heads, num_rand_blocks, block_size)

  h, r, d = num_heads, num_rand_blocks, head_size
  m = n = seq_length
  wm = wn = block_size
  scale = 1.0 / np.sqrt(d)

  bq = q.reshape(-1, h, m // wm, wm, d)
  bk = k.reshape(-1, h, n // wn, wn, d)
  bv = v.reshape(-1, h, n // wn, wn, d)
  gk = batched_index_gather(bk, rand_attn, 2).reshape(-1, h, m // wm - 2, r * wn, d)
  gv = batched_index_gather(bv, rand_attn, 2).reshape(-1, h, m // wm - 2, r * wn, d)

  # First block attends to everything.
  first = torch.einsum("bhqd,bhkd->bhqk", bq[:, :, 0], k) * scale
  first += (1.0 - to_mask) * -10000.0
  first_ctx = torch.einsum("bhqk,bhkd->bhqd", first.softmax(-1), v).unsqueeze(2)

  # Second block: first 3 + last + random blocks.
  second_k = torch.cat([bk[:, :, 0], bk[:, :, 1], bk[:, :, 2], bk[:, :, -1],
                        gk[:, :, 0]], 2)
  second_v = torch.cat([bv[:, :, 0], bv[:, :, 1], bv[:, :, 2], bv[:, :, -1],
                        gv[:, :, 0]], 2)
  second = torch.einsum("bhqd,bhkd->bhqk", bq[:, :, 1], second_k) * scale
  seq_pad = torch.cat([to_mask[:, :, :, :3 * wn], to_mask[:, :, :, -wn:],
                       torch.ones_like(rand_mask[:, :1, 0, :1])], 3)
  rand_pad = torch.cat([torch.ones_like(second[:, :, :, :4 * wn]),
                        rand_mask[:, :, 0]], 3)
  second += (1.0 - torch.minimum(seq_pad, rand_pad)) * -10000.0
  second_ctx = torch.einsum(
      "bhqk,bhkd->bhqd", second.softmax(-1), second_v).unsqueeze(2)

  # Middle blocks: sliding window (3 blocks) + first + last + random.
  exp_k = torch.cat([bk[:, :, 1:-3], bk[:, :, 2:-2], bk[:, :, 3:-1]], 3)
  exp_v = torch.cat([bv[:, :, 1:-3], bv[:, :, 2:-2], bv[:, :, 3:-1]], 3)
  mid_q = bq[:, :, 2:-2]
  inner = torch.einsum("bhlqd,bhlkd->bhlqk", mid_q, exp_k) * scale
  rand = torch.einsum("bhlqd,bhlkd->bhlqk", mid_q, gk[:, :, 1:-1]) * scale
  first_band = torch.einsum("bhlqd,bhkd->bhlqk", mid_q, bk[:, :, 0]) * scale
  last_band = torch.einsum("bhlqd,bhkd->bhlqk", mid_q, bk[:, :, -1]) * scale
  inner += (1.0 - band_mask) * -10000.0
  first_band += (1.0 - to_mask[:, :, :, :wn].unsqueeze(3)) * -10000.0
  last_band += (1.0 - to_mask[:, :, :, -wn:].unsqueeze(3)) * -10000.0
  rand += (1.0 - rand_mask[:, :, 1:-1]) * -10000.0
  band = torch.cat([first_band, inner, rand, last_band], -1).softmax(-1)
  mid_ctx = torch.einsum("bhlqk,bhlkd->bhlqd", band[:, :, :, :, wn:4 * wn], exp_v)
  mid_ctx += torch.einsum(
      "bhlqk,bhlkd->bhlqd", band[:, :, :, :, 4 * wn:-wn], gv[:, :, 1:-1])
  mid_ctx += torch.einsum(
      "bhlqk,bhkd->bhlqd", band[:, :, :, :, :wn], bv[:, :, 0])
  mid_ctx += torch.einsum(
      "bhlqk,bhkd->bhlqd", band[:, :, :, :, -wn:], bv[:, :, -1])

  # Second-to-last block: first + last 3 + random.
  sl_k = torch.cat([bk[:, :, 0], bk[:, :, -3], bk[:, :, -2], bk[:, :, -1],
                    gk[:, :, -1]], 2)
  sl_v = torch.cat([bv[:, :, 0], bv[:, :, -3], bv[:, :, -2], bv[:, :, -1],
                    gv[:, :, -1]], 2)
  sl = torch.einsum("bhqd,bhkd->bhqk", bq[:, :, -2], sl_k) * scale
  sl_seq_pad = torch.cat([to_mask[:, :, :, :wn], to_mask[:, :, :, -3 * wn:],
                          torch.ones_like(rand_mask[:, :1, 0, :1])], 3)
  sl_rand_pad = torch.cat([torch.ones_like(sl[:, :, :, :4 * wn]),
                           rand_mask[:, :, -1]], 3)
  sl += (1.0 - torch.minimum(sl_seq_pad, sl_rand_pad)) * -10000.0
  sl_ctx = torch.einsum("bhqk,bhkd->bhqd", sl.softmax(-1), sl_v).unsqueeze(2)

  # Last block attends to everything.
  last = torch.einsum("bhqd,bhkd->bhqk", bq[:, :, -1], k) * scale
  last += (1.0 - to_mask) * -10000.0
  last_ctx = torch.einsum("bhqk,bhkd->bhqd", last.softmax(-1), v).unsqueeze(2)

  ctx = torch.cat([first_ctx, second_ctx, mid_ctx, sl_ctx, last_ctx], 2)
  ctx = ctx.reshape(-1, h, m, d) * from_mask
  return ctx.permute(0, 2, 1, 3)


class MultiHeadAttention(nn.Module):
  """Self-attention with full (SDPA) or block-sparse implementation."""

  def __init__(self, config, seed=0):
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

  def _split(self, x):
    b, s, _ = x.shape
    return x.view(b, s, self.h, self.d).transpose(1, 2)  # [b, h, s, d]

  def forward(self, hidden, band_mask=None, from_mask=None, to_mask=None,
              blocked_mask=None, attn_mask=None):
    q, k, v = self._split(self.query(hidden)), self._split(
        self.key(hidden)), self._split(self.value(hidden))

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
