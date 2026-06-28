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

"""Beam search branched from Pegasus (PyTorch port).

Notations:
  B: batch_size, M: beam_size, T: max_decode_len, V: vocab_size, U: undefined
"""
# pylint: disable=invalid-name

import torch


def _map_structure(fn, struct):
  """Recursively applies `fn` to every tensor leaf in a nested structure."""
  if isinstance(struct, dict):
    return {k: _map_structure(fn, v) for k, v in struct.items()}
  if isinstance(struct, (list, tuple)):
    return type(struct)(_map_structure(fn, v) for v in struct)
  return fn(struct)


def length_normalization(start, alpha, min_len, max_len, out_of_range_penalty):
  r"""Create length normalization function.

  scores = \sum_j log(P_j) / ((start + lengths)/(1 + start))**alpha
          + out_of_range_penalty * (length > max_len or length < min_len)
  """

  def length_norm_fn(log_probs_BxM, length_int):
    dtype = log_probs_BxM.dtype
    norm_flt = (((start + float(length_int)) / (1. + start)) ** alpha)
    log_probs_BxM = log_probs_BxM / norm_flt
    too_short = length_int < min_len
    too_long = (length_int > max_len) and (max_len > 0)
    out_of_range = too_long or too_short
    if out_of_range:
      log_probs_BxM = log_probs_BxM + out_of_range_penalty
    return log_probs_BxM.to(dtype)

  return length_norm_fn


def beam_search(symbols_to_logits_fn,
                init_seq_BxT,
                initial_cache_BxU,
                vocab_size,
                beam_size,
                length_norm_fn,
                eos_id=1):
  """Beam search.

  Returns:
    Tuple of (beams_BxMxT, scores_BxM). Beam searched sequences and scores.
  """
  B, T = init_seq_BxT.shape
  M, V = beam_size, vocab_size
  dtype = torch.float32
  dtype_min = torch.finfo(dtype).min
  int_dtype = init_seq_BxT.dtype
  device = init_seq_BxT.device

  # initialize.
  init_alive_seq_BxMxT = _expand_to_beam_size(init_seq_BxT, M)
  log_probs_1xM = torch.tensor(
      [[0.] + [dtype_min] * (M - 1)], dtype=dtype, device=device)
  alive_log_probs_BxM = log_probs_1xM.repeat(B, 1)
  alive_seq_BxMxT = init_alive_seq_BxMxT
  alive_cache_BxMxU = _map_structure(
      lambda t: _expand_to_beam_size(t, M), initial_cache_BxU)
  finished_seq_BxMxT = torch.zeros_like(init_alive_seq_BxMxT)
  finished_scores_BxM = torch.zeros(B, M, dtype=dtype, device=device) + dtype_min

  for i in range(T):
    # Decode one step with beam
    logits_BMxV, cache_BMxU = symbols_to_logits_fn(
        _flatten_beam_dim(alive_seq_BxMxT),
        _map_structure(_flatten_beam_dim, alive_cache_BxMxU), i)
    logits_BxMxV = _unflatten_beam_dim(logits_BMxV, M)
    new_cache_BxMxU = _map_structure(
        lambda t: _unflatten_beam_dim(t, M), cache_BMxU)

    # select top 2 * beam_size and fill alive and finished.
    log_probs_BxMxV = logits_BxMxV - torch.logsumexp(
        logits_BxMxV, dim=2, keepdim=True)
    log_probs_BxMxV = log_probs_BxMxV + alive_log_probs_BxM.unsqueeze(2)
    log_probs_BxMV = log_probs_BxMxV.reshape(B, -1)
    new_log_probs_Bx2M, topk_indices_Bx2M = torch.topk(log_probs_BxMV, k=2 * M)
    topk_beam_Bx2M = topk_indices_Bx2M // V
    topk_seq_Bx2MxT, new_cache_Bx2MxU = _gather_nested(
        [alive_seq_BxMxT, new_cache_BxMxU], topk_beam_Bx2M)
    topk_ids_Bx2M = topk_indices_Bx2M % V
    new_seq_Bx2MxT = _update_i(topk_seq_Bx2MxT, topk_ids_Bx2M, i)
    new_finished_flags_Bx2M = torch.any(
        new_seq_Bx2MxT == eos_id, dim=-1).to(dtype)

    # get new alive
    _, topk_alive_indices_BxM = torch.topk(
        new_log_probs_Bx2M + new_finished_flags_Bx2M * dtype_min, k=M)
    (alive_seq_BxMxT, alive_log_probs_BxM, alive_cache_BxMxU) = _gather_nested(
        [new_seq_Bx2MxT, new_log_probs_Bx2M, new_cache_Bx2MxU],
        topk_alive_indices_BxM)

    # get new finished
    new_scores_Bx2M = length_norm_fn(new_log_probs_Bx2M, i + 1)
    new_scores_Bx2M = new_scores_Bx2M + (1 - new_finished_flags_Bx2M) * dtype_min
    finished_seq_Bx3MxT = torch.cat([finished_seq_BxMxT, new_seq_Bx2MxT], dim=1)
    finished_scores_Bx3M = torch.cat(
        [finished_scores_BxM, new_scores_Bx2M], dim=1)
    _, topk_finished_indices_BxM = torch.topk(finished_scores_Bx3M, k=M)
    (finished_seq_BxMxT, finished_scores_BxM) = _gather_nested(
        [finished_seq_Bx3MxT, finished_scores_Bx3M], topk_finished_indices_BxM)

  # process finished.
  final_finished_flag_BxMx1 = torch.any(
      finished_seq_BxMxT == eos_id, dim=-1, keepdim=True)
  final_seq_BxMxT = torch.where(
      final_finished_flag_BxMx1.expand(-1, -1, T), finished_seq_BxMxT,
      alive_seq_BxMxT)
  final_scores_BxM = torch.where(
      final_finished_flag_BxMx1.squeeze(-1), finished_scores_BxM,
      alive_log_probs_BxM)
  return final_seq_BxMxT, final_scores_BxM


def _update_i(tensor_BxNxT, updates_BxN, i):
  B, N, T = tensor_BxNxT.shape
  tensor_BxNxT = tensor_BxNxT.clone()
  tensor_BxNxT[:, :, i] = updates_BxN.to(tensor_BxNxT.dtype)
  return tensor_BxNxT


def _expand_to_beam_size(tensor_BxU, beam_size):
  tensor_Bx1xU = tensor_BxU.unsqueeze(1)
  tile_dims = [1] * tensor_Bx1xU.dim()
  tile_dims[1] = beam_size
  return tensor_Bx1xU.repeat(*tile_dims)


def _flatten_beam_dim(tensor_BxMxU):
  shape = list(tensor_BxMxU.shape)
  return tensor_BxMxU.reshape([shape[0] * shape[1]] + shape[2:])


def _unflatten_beam_dim(tensor_BMxU, M):
  shape = list(tensor_BMxU.shape)
  return tensor_BMxU.reshape([shape[0] // M, M] + shape[1:])


def _gather_nested(nested_BxMxU, indices_BxN):

  def _gather_beam(tensor_BxMxU):
    # tf.gather(tensor, indices, batch_dims=1, axis=1)
    n = indices_BxN.shape[1]
    extra_dims = tensor_BxMxU.dim() - 2
    idx = indices_BxN.long()
    for _ in range(extra_dims):
      idx = idx.unsqueeze(-1)
    idx = idx.expand(-1, n, *tensor_BxMxU.shape[2:])
    return torch.gather(tensor_BxMxU, 1, idx)

  return _map_structure(_gather_beam, nested_BxMxU)
