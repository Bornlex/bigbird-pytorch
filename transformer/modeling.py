"""Encoder-decoder BigBird (seq2seq) for generative tasks."""

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer import beam_search
from transformer.attention import (FullAttention, MultiHeadAttention,
                                    create_band_mask)

START_TOKEN_ID, EOS_TOKEN_ID = 2, 1


def get_activation(name):
  return {
      "gelu": lambda x: F.gelu(x, approximate="tanh"),
      "relu": F.relu,
      "tanh": torch.tanh,
      "linear": lambda x: x,
  }[name]


class Embeddings(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.word = nn.Embedding(config.vocab_size, config.hidden_size)
    self.position = nn.Embedding(config.max_position_embeddings,
                                 config.hidden_size)
    self.scale = config.hidden_size ** 0.5 if config.rescale_embedding else 1.0
    self.dropout = nn.Dropout(config.hidden_dropout_prob)

  def forward(self, input_ids, start_pos=0):
    s = input_ids.shape[1]
    pos = torch.arange(start_pos, start_pos + s, device=input_ids.device)
    emb = self.word(input_ids) * self.scale + self.position(pos).unsqueeze(0)
    return self.dropout(emb)

  def linear(self, x):
    return F.linear(x, self.word.weight)


# --------------------------------------------------------------------------- #
# Encoder (BigBird self-attention).
# --------------------------------------------------------------------------- #

class EncoderLayer(nn.Module):
  def __init__(self, config, seed=0):
    super().__init__()
    self.prenorm = config.norm_type == "prenorm"
    hidden = config.hidden_size
    self.attn = MultiHeadAttention(config, seed=seed)
    self.attn_out = nn.Linear(hidden, hidden, bias=config.use_bias)
    self.attn_norm = nn.LayerNorm(hidden, eps=1e-12)
    self.intermediate = nn.Linear(hidden, config.intermediate_size)
    self.output = nn.Linear(config.intermediate_size, hidden)
    self.output_norm = nn.LayerNorm(hidden, eps=1e-12)
    self.dropout = nn.Dropout(config.hidden_dropout_prob)
    self.act = get_activation(config.hidden_act)

  def _ff(self, x):
    return self.output(self.act(self.intermediate(x)))

  def forward(self, x, **masks):
    if self.prenorm:
      x = x + self.dropout(self.attn_out(self.attn(self.attn_norm(x), **masks)))
      x = x + self.dropout(self._ff(self.output_norm(x)))
    else:
      x = self.attn_norm(x + self.dropout(self.attn_out(self.attn(x, **masks))))
      x = self.output_norm(x + self.dropout(self._ff(x)))
    return x


class Encoder(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.config = config
    self.prenorm = config.norm_type == "prenorm"
    self.layers = nn.ModuleList(
        [EncoderLayer(config, seed=i) for i in range(config.num_hidden_layers)])
    self.norm = nn.LayerNorm(config.hidden_size, eps=1e-12)

  def _masks(self, input_mask):
    n, bs = self.config.max_encoder_length, self.config.block_size
    if self.config.attention_type == "block_sparse":
      blocked = input_mask.view(-1, n // bs, bs)
      return dict(band_mask=create_band_mask(blocked),
                  from_mask=input_mask.view(-1, 1, n, 1),
                  to_mask=input_mask.view(-1, 1, 1, n),
                  blocked_mask=blocked)
    pair = input_mask[:, None, :, None] * input_mask[:, None, None, :]
    return dict(attn_mask=(1.0 - pair) * -10000.0)

  def forward(self, hidden, input_mask):
    masks = self._masks(input_mask.float())
    if not self.prenorm:
      hidden = self.norm(hidden)
    for layer in self.layers:
      hidden = layer(hidden, **masks)
    if self.prenorm:
      hidden = self.norm(hidden)
    return hidden


# --------------------------------------------------------------------------- #
# Decoder (causal self-attention + cross-attention).
# --------------------------------------------------------------------------- #

class DecoderLayer(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.prenorm = config.norm_type == "prenorm"
    hidden = config.hidden_size
    self.self_attn = FullAttention(config)
    self.self_out = nn.Linear(hidden, hidden, bias=config.use_bias)
    self.self_norm = nn.LayerNorm(hidden, eps=1e-12)
    self.cross_attn = FullAttention(config)
    self.cross_out = nn.Linear(hidden, hidden, bias=config.use_bias)
    self.cross_norm = nn.LayerNorm(hidden, eps=1e-12)
    self.intermediate = nn.Linear(hidden, config.intermediate_size)
    self.output = nn.Linear(config.intermediate_size, hidden)
    self.output_norm = nn.LayerNorm(hidden, eps=1e-12)
    self.dropout = nn.Dropout(config.hidden_dropout_prob)
    self.act = get_activation(config.hidden_act)

  def _ff(self, x):
    return self.output(self.act(self.intermediate(x)))

  def forward(self, x, enc, self_mask, cross_mask, cache=None, decode_i=None):
    if self.prenorm:
      nx = self.self_norm(x)
      x = x + self.dropout(self.self_out(
          self.self_attn(nx, nx, self_mask, cache, decode_i)))
      nc = self.cross_norm(x)
      x = x + self.dropout(self.cross_out(
          self.cross_attn(nc, enc, cross_mask)))
      x = x + self.dropout(self._ff(self.output_norm(x)))
    else:
      s = self.self_attn(x, x, self_mask, cache, decode_i)
      x = self.self_norm(x + self.dropout(self.self_out(s)))
      c = self.cross_attn(x, enc, cross_mask)
      x = self.cross_norm(x + self.dropout(self.cross_out(c)))
      x = self.output_norm(x + self.dropout(self._ff(x)))
    return x


class Decoder(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.prenorm = config.norm_type == "prenorm"
    self.layers = nn.ModuleList(
        [DecoderLayer(config) for _ in range(config.decoder_layers)])
    self.norm = nn.LayerNorm(config.hidden_size, eps=1e-12)

  def forward(self, x, self_mask, enc, enc_mask, cache=None, decode_i=None):
    cross_mask = (1.0 - enc_mask[:, None, None, :].float()) * -10000.0
    if not self.prenorm:
      x = self.norm(x)
    for idx, layer in enumerate(self.layers):
      layer_cache = cache[f"layer_{idx}"] if cache is not None else None
      x = layer(x, enc, self_mask, cross_mask, layer_cache, decode_i)
    if self.prenorm:
      x = self.norm(x)
    return x


# --------------------------------------------------------------------------- #
# Full model.
# --------------------------------------------------------------------------- #

class TransformerModel(nn.Module):
  def __init__(self, config):
    super().__init__()
    config = copy.copy(config)
    if config.max_encoder_length <= 512:
      config.attention_type = "original_full"
    self.pad_to = 0
    if config.attention_type == "block_sparse":
      rem = config.max_encoder_length % config.block_size
      if rem:
        config.max_encoder_length += config.block_size - rem
        self.pad_to = config.max_encoder_length
    self.config = config

    self.embeddings = Embeddings(config)
    self.encoder = Encoder(config)
    self.decoder = Decoder(config)

  # --- encoding ---
  def encode(self, input_ids):
    if self.pad_to:
      input_ids = F.pad(input_ids, (0, self.pad_to - input_ids.shape[1]))
    hidden = self.embeddings(input_ids)
    input_mask = (input_ids > 0)
    return self.encoder(hidden, input_mask), input_mask

  # --- training (teacher forcing) ---
  def _causal_mask(self, length, device):
    m = torch.tril(torch.ones(length, length, device=device))
    return (1.0 - m).view(1, 1, length, length) * -10000.0

  def _shift_targets(self, target_ids):
    length = torch.count_nonzero(target_ids, dim=1)
    start = torch.full((target_ids.shape[0], 1), START_TOKEN_ID,
                       dtype=target_ids.dtype, device=target_ids.device)
    inputs = torch.cat([start, target_ids], 1)
    keep = (torch.arange(self.config.max_decoder_length + 1,
                         device=target_ids.device) < length.unsqueeze(1))
    return (inputs * keep)[:, :-1]

  def decode_train(self, target_ids, enc, enc_mask):
    input_ids = self._shift_targets(target_ids)
    emb = self.embeddings(input_ids)
    self_mask = self._causal_mask(self.config.max_decoder_length, enc.device)
    out = self.decoder(emb, self_mask, enc, enc_mask)
    logits = self.embeddings.linear(out)
    loss = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), target_ids.reshape(-1).long(),
        reduction="none").reshape(target_ids.shape)
    loss = torch.where(target_ids > 0, loss, torch.zeros_like(loss))
    return logits, loss.sum() / (target_ids > 0).sum().clamp(min=1)

  # --- generation (beam search) ---
  def _init_cache(self, batch_size, device):
    cache = {}
    for i in range(self.config.decoder_layers):
      cache[f"layer_{i}"] = {
          "k": torch.zeros(batch_size, self.config.num_attention_heads,
                           self.config.max_decoder_length, self.config.head_size,
                           device=device),
          "v": torch.zeros(batch_size, self.config.num_attention_heads,
                           self.config.max_decoder_length, self.config.head_size,
                           device=device),
      }
    return cache

  @torch.no_grad()
  def generate(self, input_ids):
    enc, enc_mask = self.encode(input_ids)
    batch_size, device = input_ids.shape[0], input_ids.device
    causal = self._causal_mask(self.config.max_decoder_length, device)

    def symbols_to_logits(target_ids, cache, i):
      step_input = target_ids[:, max(0, i - 1):max(0, i - 1) + 1]
      emb = self.embeddings(step_input, start_pos=i)
      out = self.decoder(emb, causal[:, :, i:i + 1, :],
                         cache["encoder_output"], cache["encoder_mask"],
                         cache=cache, decode_i=i)
      return self.embeddings.linear(out).squeeze(1), cache

    cache = self._init_cache(batch_size, device)
    cache["encoder_output"], cache["encoder_mask"] = enc, enc_mask
    init_seq = torch.zeros(batch_size, self.config.max_decoder_length,
                           dtype=torch.long, device=device)
    init_seq[:, 0] = START_TOKEN_ID

    length_norm = beam_search.length_normalization(5, self.config.alpha, 0, -1,
                                                   -1e3)
    beams, _ = beam_search.beam_search(
        symbols_to_logits, init_seq, cache, self.config.vocab_size,
        self.config.beam_size, length_norm, eos_id=EOS_TOKEN_ID)
    return beams[:, 0, :]

  def forward(self, input_ids, target_ids=None, training=None):
    if training:
      enc, enc_mask = self.encode(input_ids)
      logits, loss = self.decode_train(target_ids, enc, enc_mask)
      return {"logits": logits, "loss": loss, "encoder_output": enc}
    return {"pred_ids": self.generate(input_ids)}
