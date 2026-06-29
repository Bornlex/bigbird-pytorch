"""Encoder-only BigBird (postnorm, RoBERTa-style) and the masked-LM head."""

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from mlm.attention import MultiHeadAttention, create_band_mask


def get_activation(name):
  return {
      "gelu": lambda x: F.gelu(x, approximate="tanh"),  # BigBird / gelu_new
      "gelu_exact": F.gelu,                              # RoBERTa / BERT
  }[name]


class Embeddings(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.word = nn.Embedding(config.vocab_size, config.hidden_size)
    self.position = nn.Embedding(config.max_position_embeddings,
                                 config.hidden_size)
    self.dropout = nn.Dropout(config.hidden_dropout_prob)

  def forward(self, input_ids):
    pos_ids = torch.arange(input_ids.shape[1], device=input_ids.device)
    emb = self.word(input_ids) + self.position(pos_ids).unsqueeze(0)
    return self.dropout(emb)


class EncoderLayer(nn.Module):
  """Postnorm transformer block (BERT/RoBERTa style)."""

  def __init__(self, config, seed=0):
    super().__init__()
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
    x = self.attn_norm(x + self.dropout(self.attn_out(self.attn(x, **masks))))
    x = self.output_norm(x + self.dropout(self._ff(x)))
    return x


class Encoder(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.config = config
    self.layers = nn.ModuleList(
        [EncoderLayer(config, seed=i) for i in range(config.num_hidden_layers)])
    self.norm = nn.LayerNorm(config.hidden_size, eps=1e-12)  # embedding norm

  def _masks(self, input_mask):
    n, bs = self.config.max_encoder_length, self.config.block_size
    if self.config.attention_type == "block_sparse":
      blocked = input_mask.view(-1, n // bs, bs)
      return dict(
          band_mask=create_band_mask(blocked),
          from_mask=input_mask.view(-1, 1, n, 1),
          to_mask=input_mask.view(-1, 1, 1, n),
          blocked_mask=blocked)
    pair = input_mask[:, None, :, None] * input_mask[:, None, None, :]
    return dict(attn_mask=(1.0 - pair) * -10000.0)

  def forward(self, hidden, input_mask):
    masks = self._masks(input_mask.float())
    hidden = self.norm(hidden)
    for layer in self.layers:
      if self.config.use_gradient_checkpointing and self.training:
        hidden = torch.utils.checkpoint.checkpoint(
            lambda h, l=layer: l(h, **masks), hidden, use_reentrant=False)
      else:
        hidden = layer(hidden, **masks)
    return hidden


class BigBirdModel(nn.Module):
  """Encoder-only BigBird; returns the sequence output [batch, seq, hidden]."""

  def __init__(self, config):
    super().__init__()
    config = copy.copy(config)
    if config.max_encoder_length <= 512:
      config.attention_type = "original_full"
    # Block-sparse needs the length to be a multiple of block_size.
    self.pad_to = 0
    if config.attention_type == "block_sparse":
      rem = config.max_encoder_length % config.block_size
      if rem:
        config.max_encoder_length += config.block_size - rem
        self.pad_to = config.max_encoder_length
    self.config = config

    self.embeddings = Embeddings(config)
    self.encoder = Encoder(config)

  def forward(self, input_ids):
    if self.pad_to:
      input_ids = F.pad(input_ids, (0, self.pad_to - input_ids.shape[1]))
    hidden = self.embeddings(input_ids)
    return self.encoder(hidden, input_ids > 0)


class BigBirdForMaskedLM(nn.Module):
  """BigBird encoder with a masked-LM head (weights tied to the embeddings)."""

  def __init__(self, config):
    super().__init__()
    self.bert = BigBirdModel(config)
    hidden = config.hidden_size
    self.transform = nn.Linear(hidden, hidden)
    self.transform_act = get_activation(config.hidden_act)
    self.transform_norm = nn.LayerNorm(hidden, eps=1e-12)
    self.bias = nn.Parameter(torch.zeros(config.vocab_size))

  def forward(self, input_ids, masked_lm_positions=None, masked_lm_ids=None,
              masked_lm_weights=None):
    hidden = self.bert(input_ids)

    if masked_lm_positions is not None:
      idx = masked_lm_positions.long().unsqueeze(-1).expand(
          -1, -1, hidden.shape[-1])
      hidden = torch.gather(hidden, 1, idx)

    hidden = self.transform_norm(self.transform_act(self.transform(hidden)))
    logits = F.linear(hidden, self.bert.embeddings.word.weight) + self.bias

    out = {"logits": logits}
    if masked_lm_ids is not None:
      loss = F.cross_entropy(
          logits.reshape(-1, logits.shape[-1]),
          masked_lm_ids.reshape(-1).long(), reduction="none")
      weights = masked_lm_weights.reshape(-1)
      out["loss"] = (loss * weights).sum() / (weights.sum() + 1e-5)
    return out
