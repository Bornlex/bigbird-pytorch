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

"""BigBird Decoder Layers (PyTorch port)."""

import torch
import torch.nn as nn

from core import attention
from core import beam_search
from core import utils


class PrenormDecoderLayer(nn.Module):
  """Decoder layer of a transformer in Pegasus style.

  The layer_norm is taken before self-attention.
  """

  def __init__(self,
               hidden_size=768,
               intermediate_size=3072,
               intermediate_act_fn=utils.gelu,
               attention_probs_dropout_prob=0.0,
               hidden_dropout_prob=0.1,
               initializer_range=0.02,
               num_attention_heads=12,
               use_bias=True,
               name=None):
    super().__init__()

    attention_head_size = hidden_size // num_attention_heads
    # Self-attention
    self.first_layer_norm = utils.NormLayer(hidden_size)
    self.self_attn_layer = attention.MultiHeadedAttentionLayer(
        "original_full", use_bias=use_bias, name="self",
        num_attention_heads=num_attention_heads,
        size_per_head=attention_head_size,
        initializer_range=initializer_range,
        attention_probs_dropout_prob=attention_probs_dropout_prob)
    self.self_proj_layer = utils.Dense3dProjLayer(
        num_attention_heads, attention_head_size,
        utils.create_initializer(initializer_range), None, "dense", use_bias)
    self.self_attn_dropout = nn.Dropout(hidden_dropout_prob)
    # Cross-attention
    self.second_layer_norm = utils.NormLayer(hidden_size)
    self.cross_attn_layer = attention.MultiHeadedAttentionLayer(
        "original_full", use_bias=use_bias, name="encdec",
        num_attention_heads=num_attention_heads,
        size_per_head=attention_head_size,
        initializer_range=initializer_range,
        attention_probs_dropout_prob=attention_probs_dropout_prob)
    self.cross_proj_layer = utils.Dense3dProjLayer(
        num_attention_heads, attention_head_size,
        utils.create_initializer(initializer_range), None, "dense", use_bias)
    self.cross_attn_dropout = nn.Dropout(hidden_dropout_prob)
    # Feedforward
    self.third_layer_norm = utils.NormLayer(hidden_size)
    self.expand_layer = utils.Dense2dLayer(
        hidden_size, intermediate_size,
        utils.create_initializer(initializer_range),
        intermediate_act_fn, "dense")
    self.contract_layer = utils.Dense2dLayer(
        intermediate_size, hidden_size,
        utils.create_initializer(initializer_range), None, "dense")
    self.output_dropout = nn.Dropout(hidden_dropout_prob)

  def forward(self,
              layer_input,
              encoder_outputs,
              self_attention_mask,
              attention_mask,
              cache=None,
              decode_i=None,
              training=None):
    # self-attention
    normalized_layer_input = self.first_layer_norm(layer_input)
    self_attention_output = self.self_attn_layer(
        normalized_layer_input, normalized_layer_input, [self_attention_mask],
        cache=cache, decode_i=decode_i, training=training)

    self_attention_output = self.self_proj_layer(self_attention_output)
    self_attention_output = self.self_attn_dropout(self_attention_output)
    self_attention_output = self_attention_output + layer_input

    # cross-attention
    normalized_self_attention_output = self.second_layer_norm(
        self_attention_output)
    attention_output = self.cross_attn_layer(
        normalized_self_attention_output, encoder_outputs, [attention_mask],
        training=training)

    attention_output = self.cross_proj_layer(attention_output)
    attention_output = self.cross_attn_dropout(attention_output)
    attention_output = attention_output + self_attention_output

    normalized_attention_output = self.third_layer_norm(attention_output)
    intermediate_output = self.expand_layer(normalized_attention_output)

    layer_output = self.contract_layer(intermediate_output)
    layer_output = self.output_dropout(layer_output)
    layer_output = layer_output + attention_output
    return layer_output


class PostnormDecoderLayer(nn.Module):
  """Decoder layer of a transformer in BERT style.

  The layer_norm is taken after self-attention.
  """

  def __init__(self,
               hidden_size=768,
               intermediate_size=3072,
               intermediate_act_fn=utils.gelu,
               attention_probs_dropout_prob=0.0,
               hidden_dropout_prob=0.1,
               initializer_range=0.02,
               num_attention_heads=12,
               use_bias=True,
               name=None):
    super().__init__()

    attention_head_size = hidden_size // num_attention_heads
    # Self-attention
    self.self_attn_layer = attention.MultiHeadedAttentionLayer(
        "original_full", use_bias=use_bias, name="self",
        num_attention_heads=num_attention_heads,
        size_per_head=attention_head_size,
        initializer_range=initializer_range,
        attention_probs_dropout_prob=attention_probs_dropout_prob)
    self.self_proj_layer = utils.Dense3dProjLayer(
        num_attention_heads, attention_head_size,
        utils.create_initializer(initializer_range), None, "dense", use_bias)
    self.first_layer_norm = utils.NormLayer(hidden_size)
    self.self_attn_dropout = nn.Dropout(hidden_dropout_prob)
    # Cross-attention
    self.cross_attn_layer = attention.MultiHeadedAttentionLayer(
        "original_full", use_bias=use_bias, name="encdec",
        num_attention_heads=num_attention_heads,
        size_per_head=attention_head_size,
        initializer_range=initializer_range,
        attention_probs_dropout_prob=attention_probs_dropout_prob)
    self.cross_proj_layer = utils.Dense3dProjLayer(
        num_attention_heads, attention_head_size,
        utils.create_initializer(initializer_range), None, "dense", use_bias)
    self.second_layer_norm = utils.NormLayer(hidden_size)
    self.cross_attn_dropout = nn.Dropout(hidden_dropout_prob)
    # Feedforward
    self.expand_layer = utils.Dense2dLayer(
        hidden_size, intermediate_size,
        utils.create_initializer(initializer_range),
        intermediate_act_fn, "dense")
    self.contract_layer = utils.Dense2dLayer(
        intermediate_size, hidden_size,
        utils.create_initializer(initializer_range), None, "dense")
    self.third_layer_norm = utils.NormLayer(hidden_size)
    self.output_dropout = nn.Dropout(hidden_dropout_prob)

  def forward(self,
              layer_input,
              encoder_outputs,
              self_attention_mask,
              attention_mask,
              cache=None,
              decode_i=None,
              training=None):
    # self-attention
    self_attention_output = self.self_attn_layer(
        layer_input, layer_input, [self_attention_mask],
        cache=cache, decode_i=decode_i, training=training)

    self_attention_output = self.self_proj_layer(self_attention_output)
    self_attention_output = self.self_attn_dropout(self_attention_output)
    self_attention_output = self.first_layer_norm(
        self_attention_output + layer_input)

    # cross-attention
    attention_output = self.cross_attn_layer(
        self_attention_output, encoder_outputs, [attention_mask],
        training=training)

    attention_output = self.cross_proj_layer(attention_output)
    attention_output = self.cross_attn_dropout(attention_output)
    attention_output = self.second_layer_norm(
        attention_output + self_attention_output)

    intermediate_output = self.expand_layer(attention_output)

    layer_output = self.contract_layer(intermediate_output)
    layer_output = self.output_dropout(layer_output)
    layer_output = self.third_layer_norm(layer_output + attention_output)
    return layer_output


class DecoderStack(nn.Module):
  """Transformer decoder stack."""

  def __init__(self, params):
    super().__init__()
    self.params = params

    if params["norm_type"] == "prenorm":
      decoder_class = PrenormDecoderLayer
    elif params["norm_type"] == "postnorm":
      decoder_class = PostnormDecoderLayer
    else:
      raise NotImplementedError(
          "Norm type {} is not implemented".format(params["norm_type"]))

    if self.params.get("num_decoder_layers", None) is not None:
      num_hidden_layers = self.params["num_decoder_layers"]
    else:
      num_hidden_layers = self.params["num_hidden_layers"]

    self.layer_names = ["layer_%d" % i for i in range(num_hidden_layers)]
    self.decoder_layers = nn.ModuleList([
        decoder_class(
            self.params["hidden_size"],
            self.params["intermediate_size"],
            utils.get_activation(self.params["hidden_act"]),
            self.params["attention_probs_dropout_prob"],
            self.params["hidden_dropout_prob"],
            self.params["initializer_range"],
            self.params["num_attention_heads"],
            self.params["use_bias"],
            name="layer_%d" % layer_idx)
        for layer_idx in range(num_hidden_layers)
    ])

    self.layer_norm = utils.NormLayer(self.params["hidden_size"])

  def forward(self,
              decoder_inputs,
              self_attention_mask,
              encoder_outputs,
              encoder_mask,
              cache=None,
              decode_i=None,
              training=None):
    """Return the output of the decoder layer stacks."""
    # Expand encoder mask to broadcast over num heads and from_seq axis
    attention_mask = encoder_mask.unsqueeze(1).unsqueeze(1).float()

    if self.params["norm_type"] == "postnorm":
      decoder_inputs = self.layer_norm(decoder_inputs)

    layer_output = decoder_inputs
    for name, layer in zip(self.layer_names, self.decoder_layers):
      layer_cache = cache[name] if cache is not None else None
      layer_output = layer(
          layer_output, encoder_outputs, self_attention_mask, attention_mask,
          layer_cache, decode_i, training=training)

    if self.params["norm_type"] == "prenorm":
      layer_output = self.layer_norm(layer_output)

    return layer_output


def create_self_attention_mask(length, device=None):
  valid_locs = torch.tril(torch.ones(length, length, device=device))
  valid_locs = valid_locs.reshape(1, 1, length, length)
  return valid_locs


def inplace_update_i(inp_tensor, updates, i):
  """Inplace update column `i` of a [B, L] tensor with `updates` of shape [B]."""
  inp_tensor = inp_tensor.clone()
  inp_tensor[:, i] = updates
  return inp_tensor


# pylint: disable=invalid-name
def left2right_decode(symbols_to_logits_fn,
                      start_symbols,
                      context_BxU_dict,
                      batch_size,
                      max_decode_len,
                      vocab_size,
                      beam_size=1,
                      beam_start=5,
                      beam_alpha=0.6,
                      beam_min=0,
                      beam_max=-1,
                      eos_id=1,
                      device=None):
  """left to right decode.

  Notations:
    B: batch_size, V: vocab_size, T: decode_len, U: undefined dimensions

  Returns:
    decodes: Tensor[batch, decode_len]
  """
  dtype = torch.int32
  start_symbols = start_symbols.unsqueeze(1)
  if device is None:
    device = start_symbols.device

  if beam_size == 1:
    init_dec_BxT = torch.cat([
        start_symbols.to(dtype),
        torch.zeros(batch_size, max_decode_len - 1, dtype=dtype, device=device)
    ], dim=1)
    decodes_BxT = init_dec_BxT
    i = 0
    while i < max_decode_len:
      logits_BxV = symbols_to_logits_fn(decodes_BxT, context_BxU_dict, i)
      decodes_BxT = inplace_update_i(
          decodes_BxT, torch.argmax(logits_BxV, dim=-1).to(dtype), i)
      finished_B = torch.any(decodes_BxT == eos_id, dim=1)
      if bool(torch.all(finished_B)):
        break
      i += 1
    return decodes_BxT

  else:
    def symbols_to_logits_fn_with_sampling(decodes_BxT, states_BxU_dict, i):
      logits_BxV = symbols_to_logits_fn(decodes_BxT, states_BxU_dict, i)
      return logits_BxV, states_BxU_dict

    length_norm_fn = beam_search.length_normalization(
        beam_start, beam_alpha, beam_min, beam_max, -1e3)

    init_dec_BxT = torch.cat([
        start_symbols.to(torch.int32),
        torch.zeros(batch_size, max_decode_len - 1, dtype=torch.int32,
                    device=device)
    ], dim=1)

    beams, _ = beam_search.beam_search(
        symbols_to_logits_fn_with_sampling,
        init_dec_BxT,
        context_BxU_dict, vocab_size, beam_size, length_norm_fn, eos_id)
    return beams[:, 0, :]
