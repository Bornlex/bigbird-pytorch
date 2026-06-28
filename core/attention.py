# Copyright 2021 The BigBird Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""BigBird Attention Layers (PyTorch port)."""

import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from core import utils

MAX_SEQ_LEN = 4096


def batched_index_gather(params, indices, batch_dims):
  """Equivalent of tf.gather(params, indices, batch_dims=k, axis=k).

  Args:
    params: tensor of shape [B_0, ..., B_{k-1}, N, E_0, ...].
    indices: int tensor of shape [B_0, ..., B_{k-1}, I_0, ...].
    batch_dims: number of leading batch dimensions shared by params/indices.

  Returns:
    tensor of shape [B_0, ..., B_{k-1}, I_0, ..., E_0, ...].
  """
  batch_shape = list(params.shape[:batch_dims])
  n = params.shape[batch_dims]
  e_shape = list(params.shape[batch_dims + 1:])
  idx_shape = list(indices.shape[batch_dims:])

  bsz = 1
  for s in batch_shape:
    bsz *= s
  isz = 1
  for s in idx_shape:
    isz *= s

  params_flat = params.reshape([bsz, n] + e_shape)
  indices_flat = indices.reshape(bsz, isz).long()

  batch_arange = torch.arange(
      bsz, device=params.device).unsqueeze(1).expand(bsz, isz)
  gathered = params_flat[batch_arange, indices_flat]  # [bsz, isz, *e_shape]
  return gathered.reshape(batch_shape + idx_shape + e_shape)


######################## RANDOM MASK CONSTRUCTION (numpy) ######################
# The following functions are pure numpy and are identical to the TF version.


def get_single_block_row_attention(block_id,
                                   to_start_block_id,
                                   to_end_block_id,
                                   num_rand_blocks,
                                   window_block_left=1,
                                   window_block_right=1,
                                   global_block_left=1,
                                   global_block_right=1):
  """For a single row block get random row attention."""
  # list of to_blocks from which to choose random attention
  to_block_list = np.arange(to_start_block_id, to_end_block_id, dtype=np.int32)
  # permute the blocks
  perm_block = np.random.permutation(to_block_list)

  # illegal blocks for the current block id, using window
  illegal_blocks = list(
      range(block_id - window_block_left, block_id + window_block_right + 1))

  # Add blocks at the start and at the end
  illegal_blocks.extend(list(range(global_block_left)))
  illegal_blocks.extend(
      list(range(to_end_block_id - global_block_right, to_end_block_id)))

  # The second from_block cannot choose random attention on second last to_block
  if block_id == 1:
    illegal_blocks.append(to_end_block_id - 2)

  # The second last from_block cannot choose random attention on second to_block
  if block_id == to_end_block_id - 2:
    illegal_blocks.append(1)

  selected_random_blokcs = []

  for i in range(to_end_block_id - to_start_block_id):
    if perm_block[i] not in illegal_blocks:
      selected_random_blokcs.append(perm_block[i])
    if len(selected_random_blokcs) == num_rand_blocks:
      break
  return np.array(selected_random_blokcs, dtype=np.int32)


def bigbird_block_rand_mask_with_head(seq_length,
                                      block_size,
                                      num_heads,
                                      plan_from_length,
                                      plan_num_rand_blocks,
                                      window_block_left=1,
                                      window_block_right=1,
                                      global_block_top=1,
                                      global_block_bottom=1,
                                      global_block_left=1,
                                      global_block_right=1):
  """Create adjacency list of random attention (per head)."""
  num_blocks = seq_length // block_size
  plan_block_length = np.array(plan_from_length) // block_size
  max_plan_idx = plan_from_length.index(seq_length)
  rand_attn = [
      np.zeros((num_blocks, np.sum(plan_num_rand_blocks[:max_plan_idx + 1])),
               dtype=np.int32) for i in range(num_heads)
  ]

  for plan_idx in range(max_plan_idx + 1):
    rnd_r_cnt = 0
    if plan_idx > 0:
      if plan_num_rand_blocks[plan_idx] > 0:
        rnd_r_cnt = int(np.sum(plan_num_rand_blocks[:plan_idx]))
        curr_r_cnt = int(np.sum(plan_num_rand_blocks[:plan_idx + 1]))
        for blk_rw_idx in range(global_block_top,
                                plan_block_length[plan_idx - 1]):
          for h in range(num_heads):
            rand_attn[h][blk_rw_idx,
                         rnd_r_cnt:curr_r_cnt] = get_single_block_row_attention(
                             block_id=blk_rw_idx,
                             to_start_block_id=plan_block_length[plan_idx - 1],
                             to_end_block_id=plan_block_length[plan_idx],
                             num_rand_blocks=plan_num_rand_blocks[plan_idx],
                             window_block_left=window_block_left,
                             window_block_right=window_block_right,
                             global_block_left=global_block_left,
                             global_block_right=global_block_right)

      for pl_id in range(plan_idx):
        if plan_num_rand_blocks[pl_id] == 0:
          continue
        for blk_rw_idx in range(plan_block_length[plan_idx - 1],
                                plan_block_length[plan_idx]):
          rnd_r_cnt = 0
          to_start_block_id = 0
          if pl_id > 0:
            rnd_r_cnt = int(np.sum(plan_num_rand_blocks[:pl_id]))
            to_start_block_id = plan_block_length[pl_id - 1]
          curr_r_cnt = int(np.sum(plan_num_rand_blocks[:pl_id + 1]))
          for h in range(num_heads):
            rand_attn[h][blk_rw_idx,
                         rnd_r_cnt:curr_r_cnt] = get_single_block_row_attention(
                             block_id=blk_rw_idx,
                             to_start_block_id=to_start_block_id,
                             to_end_block_id=plan_block_length[pl_id],
                             num_rand_blocks=plan_num_rand_blocks[pl_id],
                             window_block_left=window_block_left,
                             window_block_right=window_block_right,
                             global_block_left=global_block_left,
                             global_block_right=global_block_right)

    if plan_num_rand_blocks[plan_idx] == 0:
      continue
    curr_r_cnt = int(np.sum(plan_num_rand_blocks[:plan_idx + 1]))
    from_start_block_id = global_block_top
    to_start_block_id = 0
    if plan_idx > 0:
      rnd_r_cnt = int(np.sum(plan_num_rand_blocks[:plan_idx]))
      from_start_block_id = plan_block_length[plan_idx - 1]
      to_start_block_id = plan_block_length[plan_idx - 1]

    for blk_rw_idx in range(from_start_block_id, plan_block_length[plan_idx]):
      for h in range(num_heads):
        rand_attn[h][blk_rw_idx,
                     rnd_r_cnt:curr_r_cnt] = get_single_block_row_attention(
                         block_id=blk_rw_idx,
                         to_start_block_id=to_start_block_id,
                         to_end_block_id=plan_block_length[plan_idx],
                         num_rand_blocks=plan_num_rand_blocks[plan_idx],
                         window_block_left=window_block_left,
                         window_block_right=window_block_right,
                         global_block_left=global_block_left,
                         global_block_right=global_block_right)

  for nh in range(num_heads):
    rand_attn[nh] = rand_attn[nh][global_block_top:num_blocks -
                                  global_block_bottom, :]
  return rand_attn


def get_rand_attn_plan(from_seq_length, from_block_size, num_rand_blocks):
  """Gives the plan of where to put random attention."""
  plan_from_length = []
  plan_num_rand_blocks = []
  if (2 * num_rand_blocks + 5) < (from_seq_length // from_block_size):
    plan_from_length.append(int((2 * num_rand_blocks + 5) * from_block_size))
    plan_num_rand_blocks.append(num_rand_blocks)
    plan_from_length.append(from_seq_length)
    plan_num_rand_blocks.append(0)
  elif (num_rand_blocks + 5) < (from_seq_length // from_block_size):
    plan_from_length.append(int((num_rand_blocks + 5) * from_block_size))
    plan_num_rand_blocks.append(num_rand_blocks // 2)
    plan_from_length.append(from_seq_length)
    plan_num_rand_blocks.append(num_rand_blocks - (num_rand_blocks // 2))
  else:
    plan_from_length.append(from_seq_length)
    plan_num_rand_blocks.append(num_rand_blocks)

  return plan_from_length, plan_num_rand_blocks


def bigbird_block_rand_mask(from_seq_length,
                            to_seq_length,
                            from_block_size,
                            to_block_size,
                            num_rand_blocks,
                            last_idx=-1):
  """Create adjacency list of random attention."""
  rand_attn = np.zeros(
      (from_seq_length // from_block_size - 2, num_rand_blocks), dtype=np.int32)
  middle_seq = np.arange(1, to_seq_length // to_block_size - 1, dtype=np.int32)
  last = to_seq_length // to_block_size - 1
  if last_idx > (2 * to_block_size):
    last = (last_idx // to_block_size) - 1

  r = num_rand_blocks  # shorthand
  for i in range(1, from_seq_length // from_block_size - 1):
    start = i - 2
    end = i
    if i == 1:
      rand_attn[i - 1, :] = np.random.permutation(middle_seq[2:last])[:r]
    elif i == 2:
      rand_attn[i - 1, :] = np.random.permutation(middle_seq[3:last])[:r]
    elif i == from_seq_length // from_block_size - 3:
      rand_attn[i - 1, :] = np.random.permutation(middle_seq[:last])[:r]
    elif i == from_seq_length // from_block_size - 2:
      rand_attn[i - 1, :] = np.random.permutation(middle_seq[:last])[:r]
    else:
      if start > last:
        start = last
        rand_attn[i - 1, :] = np.random.permutation(middle_seq[:start])[:r]
      elif (end + 1) == last:
        rand_attn[i - 1, :] = np.random.permutation(middle_seq[:start])[:r]
      else:
        rand_attn[i - 1, :] = np.random.permutation(
            np.concatenate((middle_seq[:start], middle_seq[end + 1:last])))[:r]
  return rand_attn


def full_bigbird_mask(from_seq_length,
                      to_seq_length,
                      from_block_size,
                      to_block_size,
                      rand_attn):
  """Calculate BigBird attention pattern as a full dense matrix."""
  attn_mask = np.zeros((MAX_SEQ_LEN, MAX_SEQ_LEN), dtype=np.int32)
  for i in range(1, (MAX_SEQ_LEN // from_block_size) - 1):
    attn_mask[(i) * from_block_size:(i + 1) * from_block_size,
              (i - 1) * to_block_size:(i + 2) * to_block_size] = 1
    for j in rand_attn[i - 1, :]:
      attn_mask[i * from_block_size:(i + 1) * from_block_size,
                j * to_block_size:(j + 1) * to_block_size] = 1

  attn_mask[:from_block_size, :] = 1
  attn_mask[:, :to_block_size] = 1
  attn_mask[:, -to_block_size:] = 1
  attn_mask[-from_block_size:, :] = 1
  clipped_attn_mask = attn_mask[:from_seq_length, :to_seq_length]
  return np.array(clipped_attn_mask, dtype=bool)


########################## MASK BUILDERS (tensor) #############################


def create_rand_mask_from_inputs(from_blocked_mask,
                                 to_blocked_mask,
                                 rand_attn,
                                 num_attention_heads,
                                 num_rand_blocks,
                                 from_seq_length,
                                 from_block_size):
  """Create 4D attention mask from a 3D tensor mask."""
  num_windows = from_seq_length // from_block_size - 2
  rand_mask = batched_index_gather(to_blocked_mask, rand_attn, batch_dims=1)
  rand_mask = rand_mask.reshape(
      -1, num_attention_heads, num_windows, num_rand_blocks * from_block_size)
  rand_mask = torch.einsum("BLQ,BHLK->BHLQK", from_blocked_mask[:, 1:-1],
                           rand_mask)
  return rand_mask


def create_band_mask_from_inputs(from_blocked_mask, to_blocked_mask):
  """Create 4D attention mask from a 3D blocked tensor mask."""
  exp_blocked_to_pad = torch.cat(
      [to_blocked_mask[:, 1:-3], to_blocked_mask[:, 2:-2],
       to_blocked_mask[:, 3:-1]], 2)
  band_mask = torch.einsum(
      "BLQ,BLK->BLQK", from_blocked_mask[:, 2:-2], exp_blocked_to_pad)
  band_mask = band_mask.unsqueeze(1)
  return band_mask


def create_attention_mask_from_input_mask(from_mask, to_mask):
  """Create attention mask from a 2D tensor mask."""
  mask = torch.einsum("BF,BT->BFT", from_mask, to_mask)
  mask = mask.unsqueeze(1)
  return mask


def bigbird_block_sparse_attention(query_layer,
                                   key_layer,
                                   value_layer,
                                   band_mask,
                                   from_mask,
                                   to_mask,
                                   from_blocked_mask,
                                   to_blocked_mask,
                                   rand_attn,
                                   num_attention_heads,
                                   size_per_head,
                                   num_rand_blocks,
                                   from_seq_length,
                                   to_seq_length,
                                   from_block_size,
                                   to_block_size):
  """BigBird attention sparse calculation using blocks in linear time.

  Assumes from_seq_length//from_block_size == to_seq_length//to_block_size.

  Returns:
    float Tensor of shape [batch_size, from_seq_length, num_attention_heads,
      size_per_head].
  """
  assert from_seq_length // from_block_size == to_seq_length // to_block_size

  # repeat for batch size
  batch_size = utils.get_shape_list(query_layer)[0]
  rand_attn = rand_attn.unsqueeze(0)
  rand_attn = rand_attn.repeat(batch_size, 1, 1, 1)

  rand_mask = create_rand_mask_from_inputs(
      from_blocked_mask, to_blocked_mask, rand_attn,
      num_attention_heads, num_rand_blocks,
      from_seq_length, from_block_size)

  # Define shorthands
  h = num_attention_heads
  r = num_rand_blocks
  d = size_per_head
  m = from_seq_length
  n = to_seq_length
  wm = from_block_size
  wn = to_block_size

  blocked_query_matrix = query_layer.reshape(-1, h, m // wm, wm, d)
  blocked_key_matrix = key_layer.reshape(-1, h, n // wn, wn, d)
  blocked_value_matrix = value_layer.reshape(-1, h, n // wn, wn, d)
  gathered_key = batched_index_gather(
      blocked_key_matrix, rand_attn, batch_dims=2).reshape(
          -1, h, m // wm - 2, r * wn, d)
  gathered_value = batched_index_gather(
      blocked_value_matrix, rand_attn, batch_dims=2).reshape(
          -1, h, m // wm - 2, r * wn, d)

  first_product = torch.einsum(
      "BHQD,BHKD->BHQK", blocked_query_matrix[:, :, 0], key_layer)
  first_product = first_product * (1.0 / np.sqrt(d))
  first_product += (1.0 - to_mask) * -10000.0
  first_attn_weights = F.softmax(first_product, dim=-1)  # [b, h, wm, n]
  first_context_layer = torch.einsum(
      "BHQK,BHKD->BHQD", first_attn_weights, value_layer)
  first_context_layer = first_context_layer.unsqueeze(2)

  second_key_mat = torch.cat([
      blocked_key_matrix[:, :, 0], blocked_key_matrix[:, :, 1],
      blocked_key_matrix[:, :, 2], blocked_key_matrix[:, :, -1],
      gathered_key[:, :, 0]], 2)  # [b, h, (4+r)*wn, -1]
  second_value_mat = torch.cat([
      blocked_value_matrix[:, :, 0], blocked_value_matrix[:, :, 1],
      blocked_value_matrix[:, :, 2], blocked_value_matrix[:, :, -1],
      gathered_value[:, :, 0]], 2)  # [b, h, (4+r)*wn, -1]
  second_product = torch.einsum(
      "BHQD,BHKD->BHQK", blocked_query_matrix[:, :, 1], second_key_mat)
  second_seq_pad = torch.cat([
      to_mask[:, :, :, :3 * wn], to_mask[:, :, :, -wn:],
      torch.ones_like(rand_mask[:, :1, 0, :1])], 3)
  second_rand_pad = torch.cat(
      [torch.ones_like(second_product[:, :, :, :4 * wn]), rand_mask[:, :, 0]], 3)
  second_product = second_product * (1.0 / np.sqrt(d))
  second_product += (1.0 -
                     torch.minimum(second_seq_pad, second_rand_pad)) * -10000.0
  second_attn_weights = F.softmax(second_product, dim=-1)
  second_context_layer = torch.einsum(
      "BHQK,BHKD->BHQD", second_attn_weights, second_value_mat)
  second_context_layer = second_context_layer.unsqueeze(2)

  exp_blocked_key_matrix = torch.cat([
      blocked_key_matrix[:, :, 1:-3], blocked_key_matrix[:, :, 2:-2],
      blocked_key_matrix[:, :, 3:-1]], 3)  # [b, h, m//wm-4, 3*wn, -1]
  exp_blocked_value_matrix = torch.cat([
      blocked_value_matrix[:, :, 1:-3], blocked_value_matrix[:, :, 2:-2],
      blocked_value_matrix[:, :, 3:-1]], 3)  # [b, h, m//wm-4, 3*wn, -1]
  middle_query_matrix = blocked_query_matrix[:, :, 2:-2]
  inner_band_product = torch.einsum(
      "BHLQD,BHLKD->BHLQK", middle_query_matrix, exp_blocked_key_matrix)
  inner_band_product = inner_band_product * (1.0 / np.sqrt(d))
  rand_band_product = torch.einsum(
      "BHLQD,BHLKD->BHLQK", middle_query_matrix, gathered_key[:, :, 1:-1])
  rand_band_product = rand_band_product * (1.0 / np.sqrt(d))
  first_band_product = torch.einsum(
      "BHLQD,BHKD->BHLQK", middle_query_matrix, blocked_key_matrix[:, :, 0])
  first_band_product = first_band_product * (1.0 / np.sqrt(d))
  last_band_product = torch.einsum(
      "BHLQD,BHKD->BHLQK", middle_query_matrix, blocked_key_matrix[:, :, -1])
  last_band_product = last_band_product * (1.0 / np.sqrt(d))
  inner_band_product += (1.0 - band_mask) * -10000.0
  first_band_product += (
      1.0 - to_mask[:, :, :, :wn].unsqueeze(3)) * -10000.0
  last_band_product += (
      1.0 - to_mask[:, :, :, -wn:].unsqueeze(3)) * -10000.0
  rand_band_product += (1.0 - rand_mask[:, :, 1:-1]) * -10000.0
  band_product = torch.cat([
      first_band_product, inner_band_product, rand_band_product,
      last_band_product], -1)  # [b, h, m//wm-4, wm, (5+r)*wn]
  attn_weights = F.softmax(band_product, dim=-1)
  context_layer = torch.einsum(
      "BHLQK,BHLKD->BHLQD", attn_weights[:, :, :, :, wn:4 * wn],
      exp_blocked_value_matrix)
  context_layer += torch.einsum(
      "BHLQK,BHLKD->BHLQD", attn_weights[:, :, :, :, 4 * wn:-wn],
      gathered_value[:, :, 1:-1])
  context_layer += torch.einsum(
      "BHLQK,BHKD->BHLQD", attn_weights[:, :, :, :, :wn],
      blocked_value_matrix[:, :, 0])
  context_layer += torch.einsum(
      "BHLQK,BHKD->BHLQD", attn_weights[:, :, :, :, -wn:],
      blocked_value_matrix[:, :, -1])

  second_last_key_mat = torch.cat([
      blocked_key_matrix[:, :, 0], blocked_key_matrix[:, :, -3],
      blocked_key_matrix[:, :, -2], blocked_key_matrix[:, :, -1],
      gathered_key[:, :, -1]], 2)  # [b, h, (4+r)*wn, -1]
  second_last_value_mat = torch.cat([
      blocked_value_matrix[:, :, 0], blocked_value_matrix[:, :, -3],
      blocked_value_matrix[:, :, -2], blocked_value_matrix[:, :, -1],
      gathered_value[:, :, -1]], 2)  # [b, h, (4+r)*wn, -1]
  second_last_product = torch.einsum(
      "BHQD,BHKD->BHQK", blocked_query_matrix[:, :, -2], second_last_key_mat)
  second_last_seq_pad = torch.cat([
      to_mask[:, :, :, :wn], to_mask[:, :, :, -3 * wn:],
      torch.ones_like(rand_mask[:, :1, 0, :1])], 3)
  second_last_rand_pad = torch.cat(
      [torch.ones_like(second_last_product[:, :, :, :4 * wn]),
       rand_mask[:, :, -1]], 3)
  second_last_product = second_last_product * (1.0 / np.sqrt(d))
  second_last_product += (
      1.0 - torch.minimum(second_last_seq_pad, second_last_rand_pad)) * -10000.0
  second_last_attn_weights = F.softmax(second_last_product, dim=-1)
  second_last_context_layer = torch.einsum(
      "BHQK,BHKD->BHQD", second_last_attn_weights, second_last_value_mat)
  second_last_context_layer = second_last_context_layer.unsqueeze(2)

  last_product = torch.einsum(
      "BHQD,BHKD->BHQK", blocked_query_matrix[:, :, -1], key_layer)
  last_product = last_product * (1.0 / np.sqrt(d))
  last_product += (1.0 - to_mask) * -10000.0
  last_attn_weights = F.softmax(last_product, dim=-1)
  last_context_layer = torch.einsum(
      "BHQK,BHKD->BHQD", last_attn_weights, value_layer)
  last_context_layer = last_context_layer.unsqueeze(2)

  context_layer = torch.cat([
      first_context_layer, second_context_layer, context_layer,
      second_last_context_layer, last_context_layer
  ], 2)
  context_layer = context_layer.reshape(-1, h, m, d) * from_mask
  context_layer = context_layer.permute(0, 2, 1, 3)
  return context_layer


class MultiHeadedAttentionLayer(nn.Module):
  """A multi-headed attention layer.

  It implements following types of multi-headed attention:
  - original_full attention from "Attention is all you Need".
  - simulated_sparse attention from BigBird with full quadratic implemention.
  - block_sparse attention from BigBird with memory efficient linear impl.
  """

  def __init__(self,
               attention_type,
               num_attention_heads=1,
               size_per_head=512,
               num_rand_blocks=3,
               from_seq_length=1024,
               to_seq_length=1024,
               from_block_size=64,
               to_block_size=64,
               attention_probs_dropout_prob=0.0,
               initializer_range=0.02,
               use_bias=True,
               seed=None,
               query_act=None,
               key_act=None,
               value_act=None,
               name=None):
    super().__init__()
    self.attention_type = attention_type
    self.num_attention_heads = num_attention_heads
    self.size_per_head = size_per_head
    self.num_rand_blocks = num_rand_blocks
    self.from_seq_length = from_seq_length
    self.to_seq_length = to_seq_length
    self.from_block_size = from_block_size
    self.to_block_size = to_block_size
    self.seed = seed

    self.query_layer = utils.Dense3dLayer(
        num_attention_heads, size_per_head,
        utils.create_initializer(initializer_range), query_act,
        "query", head_first=True, use_bias=use_bias)
    self.key_layer = utils.Dense3dLayer(
        num_attention_heads, size_per_head,
        utils.create_initializer(initializer_range), key_act,
        "key", head_first=True, use_bias=use_bias)
    self.value_layer = utils.Dense3dLayer(
        num_attention_heads, size_per_head,
        utils.create_initializer(initializer_range), value_act,
        "value", head_first=True, use_bias=use_bias)

    if attention_type == "original_full":
      logging.info("**** Using original full attention ****")
      self.attention_dropout = nn.Dropout(attention_probs_dropout_prob)
      self.attn_impl = self.original_full_attention
    elif attention_type == "simulated_sparse":
      logging.info("**** Using simulated sparse attention ****")
      self.attention_dropout = nn.Identity()
      rand_attn = self.generate_rand_attn_list()
      self.register_buffer("rand_attn", rand_attn)
      self.register_buffer(
          "rand_block_mask", self.convert_attn_list_to_mask(rand_attn))
      self.attn_impl = self.bigbird_simulated_attention
    elif attention_type == "block_sparse":
      logging.info("**** Using block sparse attention ****")
      assert from_seq_length // from_block_size == to_seq_length // to_block_size, (
          "Error the number of blocks needs to be same!")
      self.attention_dropout = None
      self.register_buffer("rand_attn", self.generate_rand_attn_list())
      self.attn_impl = self.bigbird_block_sparse_attention
    else:
      raise NotImplementedError(
          "Attention type {} is not implemented".format(attention_type))

  def generate_rand_attn_list(self):
    if self.seed is not None:
      np.random.seed(self.seed)
    if self.from_seq_length in [1024, 2048, 3072, 4096]:
      rand_attn = [
          bigbird_block_rand_mask(
              MAX_SEQ_LEN, MAX_SEQ_LEN,
              self.from_block_size, self.to_block_size, self.num_rand_blocks,
              last_idx=1024
          )[:(self.from_seq_length // self.from_block_size - 2)]
          for _ in range(self.num_attention_heads)
      ]
    else:
      plan_from_length, plan_num_rand_blocks = get_rand_attn_plan(
          self.from_seq_length, self.from_block_size, self.num_rand_blocks)
      rand_attn = bigbird_block_rand_mask_with_head(
          seq_length=self.from_seq_length,
          block_size=self.from_block_size,
          num_heads=self.num_attention_heads,
          plan_from_length=plan_from_length,
          plan_num_rand_blocks=plan_num_rand_blocks)
    rand_attn = np.stack(rand_attn, axis=0)
    return torch.tensor(rand_attn, dtype=torch.int64)

  def convert_attn_list_to_mask(self, rand_attn):
    temp_mask = [
        full_bigbird_mask(
            self.from_seq_length, self.to_seq_length,
            self.from_block_size, self.to_block_size,
            rand_attn=rand_attn[i].cpu().numpy())
        for i in range(self.num_attention_heads)
    ]
    temp_mask = np.stack(temp_mask, axis=0)
    return torch.tensor(temp_mask, dtype=torch.float32)

  def original_full_attention(self,
                              query_layer,
                              key_layer,
                              value_layer,
                              masks,
                              training=None):
    """Full quadratic attention calculation."""
    attention_mask = masks[0]

    attention_scores = torch.einsum("BNFH,BNTH->BNFT", query_layer, key_layer)
    attention_scores = attention_scores * (
        1.0 / np.sqrt(float(self.size_per_head)))

    if attention_mask is not None:
      adder = (1.0 - attention_mask) * -10000.0
      attention_scores = attention_scores + adder

    attention_probs = F.softmax(attention_scores, dim=-1)
    attention_probs = self.attention_dropout(attention_probs)

    context_layer = torch.einsum(
        "BNFT,BNTH->BFNH", attention_probs, value_layer)
    return context_layer

  def bigbird_simulated_attention(self,
                                  query_layer,
                                  key_layer,
                                  value_layer,
                                  masks,
                                  training=None):
    """BigBird attention calculation using masks in quadratic time."""
    attention_mask = masks[0]
    rand_block_mask = self.rand_block_mask.unsqueeze(0)  # [1, N, F, T]
    if attention_mask is not None:
      attention_mask = torch.minimum(attention_mask, rand_block_mask)
    else:
      attention_mask = rand_block_mask
    return self.original_full_attention(
        query_layer, key_layer, value_layer, [attention_mask],
        training=training)

  def bigbird_block_sparse_attention(self,
                                     query_layer,
                                     key_layer,
                                     value_layer,
                                     masks,
                                     training=None):
    """BigBird attention sparse calculation using blocks in linear time."""
    (_, band_mask, from_mask, to_mask,
     from_blocked_mask, to_blocked_mask) = masks

    return bigbird_block_sparse_attention(
        query_layer, key_layer, value_layer,
        band_mask, from_mask, to_mask, from_blocked_mask, to_blocked_mask,
        self.rand_attn, self.num_attention_heads, self.size_per_head,
        self.num_rand_blocks, self.from_seq_length, self.to_seq_length,
        self.from_block_size, self.to_block_size)

  def forward(self,
              from_tensor,
              to_tensor,
              masks,
              cache=None,
              decode_i=None,
              training=None):
    """Implements a multi-headed attention layer from from_tensor to to_tensor.

    Returns:
      float Tensor of shape [batch_size, from_seq_length, num_attention_heads,
        size_per_head].
    """
    # `query` = [b, h, m, d]
    query = self.query_layer(from_tensor)
    # `key` = [b, h, n, d]
    key = self.key_layer(to_tensor)
    # `value` = [b, h, n, d]
    value = self.value_layer(to_tensor)

    if cache is not None and decode_i is not None:
      max_len = utils.get_shape_list(cache["k"])[2]
      indices_select = F.one_hot(
          torch.as_tensor(decode_i, device=to_tensor.device, dtype=torch.long),
          max_len).to(to_tensor.dtype).reshape(1, 1, max_len, 1)
      key = cache["k"] + key * indices_select
      value = cache["v"] + value * indices_select
      cache["k"] = key
      cache["v"] = value

    contextual_output = self.attn_impl(
        query, key, value, masks, training=training)

    return contextual_output
