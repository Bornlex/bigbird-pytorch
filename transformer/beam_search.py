"""Beam search (ported from Pegasus).

Notations: B=batch, M=beam, T=decode_len, V=vocab, U=undefined.
"""
# pylint: disable=invalid-name

import torch


def _map(fn, struct):
  if isinstance(struct, dict):
    return {k: _map(fn, v) for k, v in struct.items()}
  if isinstance(struct, (list, tuple)):
    return type(struct)(_map(fn, v) for v in struct)
  return fn(struct)


def length_normalization(start, alpha, min_len, max_len, out_of_range_penalty):
  def fn(log_probs_BxM, length):
    norm = ((start + float(length)) / (1. + start)) ** alpha
    log_probs_BxM = log_probs_BxM / norm
    if length < min_len or (max_len > 0 and length > max_len):
      log_probs_BxM = log_probs_BxM + out_of_range_penalty
    return log_probs_BxM
  return fn


def beam_search(symbols_to_logits_fn, init_seq_BxT, initial_cache_BxU,
                vocab_size, beam_size, length_norm_fn, eos_id=1):
  B, T = init_seq_BxT.shape
  M, V = beam_size, vocab_size
  dtype = torch.float32
  neg = torch.finfo(dtype).min
  device = init_seq_BxT.device

  alive_seq = _expand_to_beam(init_seq_BxT, M)
  alive_log_probs = torch.tensor(
      [[0.] + [neg] * (M - 1)], dtype=dtype, device=device).repeat(B, 1)
  alive_cache = _map(lambda t: _expand_to_beam(t, M), initial_cache_BxU)
  finished_seq = torch.zeros_like(alive_seq)
  finished_scores = torch.full((B, M), neg, dtype=dtype, device=device)

  for i in range(T):
    logits_BMxV, cache_BMxU = symbols_to_logits_fn(
        _flatten(alive_seq), _map(_flatten, alive_cache), i)
    logits = _unflatten(logits_BMxV, M)
    new_cache = _map(lambda t: _unflatten(t, M), cache_BMxU)

    log_probs = logits - torch.logsumexp(logits, dim=2, keepdim=True)
    log_probs = log_probs + alive_log_probs.unsqueeze(2)
    log_probs_flat = log_probs.reshape(B, -1)
    new_log_probs, idx = torch.topk(log_probs_flat, k=2 * M)
    topk_beam = idx // V
    topk_seq, topk_cache = _gather([alive_seq, new_cache], topk_beam)
    topk_ids = idx % V
    new_seq = _update_i(topk_seq, topk_ids, i)
    finished_flags = torch.any(new_seq == eos_id, dim=-1).to(dtype)

    # alive = top M not-yet-finished
    _, alive_idx = torch.topk(new_log_probs + finished_flags * neg, k=M)
    alive_seq, alive_log_probs, alive_cache = _gather(
        [new_seq, new_log_probs, topk_cache], alive_idx)

    # finished = top M among finished (old + new)
    new_scores = length_norm_fn(new_log_probs, i + 1)
    new_scores = new_scores + (1 - finished_flags) * neg
    cat_seq = torch.cat([finished_seq, new_seq], dim=1)
    cat_scores = torch.cat([finished_scores, new_scores], dim=1)
    _, fin_idx = torch.topk(cat_scores, k=M)
    finished_seq, finished_scores = _gather([cat_seq, cat_scores], fin_idx)

  has_finished = torch.any(finished_seq == eos_id, dim=-1, keepdim=True)
  final_seq = torch.where(has_finished.expand(-1, -1, T),
                          finished_seq, alive_seq)
  final_scores = torch.where(has_finished.squeeze(-1),
                             finished_scores, alive_log_probs)
  return final_seq, final_scores


def _update_i(tensor_BxNxT, updates_BxN, i):
  tensor_BxNxT = tensor_BxNxT.clone()
  tensor_BxNxT[:, :, i] = updates_BxN.to(tensor_BxNxT.dtype)
  return tensor_BxNxT


def _expand_to_beam(tensor_BxU, M):
  t = tensor_BxU.unsqueeze(1)
  reps = [1] * t.dim()
  reps[1] = M
  return t.repeat(*reps)


def _flatten(t):
  shape = list(t.shape)
  return t.reshape([shape[0] * shape[1]] + shape[2:])


def _unflatten(t, M):
  shape = list(t.shape)
  return t.reshape([shape[0] // M, M] + shape[1:])


def _gather(struct, indices_BxN):
  def gather_one(t):
    n = indices_BxN.shape[1]
    idx = indices_BxN.long()
    for _ in range(t.dim() - 2):
      idx = idx.unsqueeze(-1)
    idx = idx.expand(-1, n, *t.shape[2:])
    return torch.gather(t, 1, idx)
  return _map(gather_one, struct)
