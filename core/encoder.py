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

"""BigBird Encoder Layers (PyTorch port)."""

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint

from core import attention
from core import utils


class PrenormEncoderLayer(nn.Module):
  """Encoder layer of a transformer in Pegasus style.

  The layer_norm is taken before self-attention.
  """

  def __init__(self,
               attention_type,
               hidden_size=768,
               intermediate_size=3072,
               intermediate_act_fn=utils.gelu,
               attention_probs_dropout_prob=0.0,
               hidden_dropout_prob=0.1,
               initializer_range=0.02,
               num_attention_heads=12,
               num_rand_blocks=3,
               seq_length=1024,
               block_size=64,
               use_bias=True,
               seed=None,
               name=None):
    super().__init__()

    attention_head_size = hidden_size // num_attention_heads
    # Pre-Normalization layer
    self.first_layer_norm = utils.NormLayer(hidden_size)
    # Self-Attention layer
    self.attn_layer = attention.MultiHeadedAttentionLayer(
        attention_type, num_attention_heads, attention_head_size,
        num_rand_blocks, seq_length, seq_length, block_size, block_size,
        attention_probs_dropout_prob, initializer_range, use_bias,
        seed, name="self")
    # Feedforward layer
    self.projection_layer = utils.Dense3dProjLayer(
        num_attention_heads, attention_head_size,
        utils.create_initializer(initializer_range), None, "dense", use_bias)
    self.attention_dropout = nn.Dropout(hidden_dropout_prob)

    # Normalization layer
    self.second_layer_norm = utils.NormLayer(hidden_size)
    # Feedforward layers
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
              attention_mask=None,
              band_mask=None,
              from_mask=None,
              to_mask=None,
              input_blocked_mask=None,
              training=None):
    # self-attention
    normalized_layer_input = self.first_layer_norm(layer_input)
    attention_output = self.attn_layer(
        normalized_layer_input, normalized_layer_input, [
            attention_mask, band_mask, from_mask, to_mask, input_blocked_mask,
            input_blocked_mask
        ], training=training)

    attention_output = self.projection_layer(attention_output)
    attention_output = self.attention_dropout(attention_output)
    attention_output = attention_output + layer_input

    normalized_attention_output = self.second_layer_norm(attention_output)
    intermediate_output = self.expand_layer(normalized_attention_output)

    layer_output = self.contract_layer(intermediate_output)
    layer_output = self.output_dropout(layer_output)
    layer_output = layer_output + attention_output
    return layer_output


class PostnormEncoderLayer(nn.Module):
  """Encoder layer of a transformer in BERT style.

  The layer_norm is taken after self-attention.
  """

  def __init__(self,
               attention_type,
               hidden_size=768,
               intermediate_size=3072,
               intermediate_act_fn=utils.gelu,
               attention_probs_dropout_prob=0.0,
               hidden_dropout_prob=0.1,
               initializer_range=0.02,
               num_attention_heads=12,
               num_rand_blocks=3,
               seq_length=1024,
               block_size=64,
               use_bias=True,
               seed=None,
               name=None):
    super().__init__()

    attention_head_size = hidden_size // num_attention_heads
    # Self-Attention layer
    self.attn_layer = attention.MultiHeadedAttentionLayer(
        attention_type, num_attention_heads, attention_head_size,
        num_rand_blocks, seq_length, seq_length, block_size, block_size,
        attention_probs_dropout_prob, initializer_range, use_bias,
        seed, name="self")
    self.projection_layer = utils.Dense3dProjLayer(
        num_attention_heads, attention_head_size,
        utils.create_initializer(initializer_range), None, "dense", use_bias)
    self.first_layer_norm = utils.NormLayer(hidden_size)
    self.attention_dropout = nn.Dropout(hidden_dropout_prob)

    self.expand_layer = utils.Dense2dLayer(
        hidden_size, intermediate_size,
        utils.create_initializer(initializer_range),
        intermediate_act_fn, "dense")
    self.contract_layer = utils.Dense2dLayer(
        intermediate_size, hidden_size,
        utils.create_initializer(initializer_range), None, "dense")
    self.second_layer_norm = utils.NormLayer(hidden_size)
    self.output_dropout = nn.Dropout(hidden_dropout_prob)

  def forward(self,
              layer_input,
              attention_mask=None,
              band_mask=None,
              from_mask=None,
              to_mask=None,
              input_blocked_mask=None,
              training=None):
    # self-attention
    attention_output = self.attn_layer(
        layer_input, layer_input, [
            attention_mask, band_mask, from_mask, to_mask, input_blocked_mask,
            input_blocked_mask
        ], training=training)

    attention_output = self.projection_layer(attention_output)
    attention_output = self.attention_dropout(attention_output)
    attention_output = self.first_layer_norm(attention_output + layer_input)

    intermediate_output = self.expand_layer(attention_output)

    layer_output = self.contract_layer(intermediate_output)
    layer_output = self.output_dropout(layer_output)
    layer_output = self.second_layer_norm(layer_output + attention_output)
    return layer_output


class EncoderStack(nn.Module):
  """Transformer encoder stack."""

  def __init__(self, params):
    super().__init__()
    self.params = params

    if params["norm_type"] == "prenorm":
      encoder_class = PrenormEncoderLayer
    elif params["norm_type"] == "postnorm":
      encoder_class = PostnormEncoderLayer
    else:
      raise NotImplementedError(
          "Norm type {} is not implemented".format(params["norm_type"]))

    self.encoder_layers = nn.ModuleList([
        encoder_class(
            self.params["attention_type"],
            self.params["hidden_size"],
            self.params["intermediate_size"],
            utils.get_activation(self.params["hidden_act"]),
            self.params["attention_probs_dropout_prob"],
            self.params["hidden_dropout_prob"],
            self.params["initializer_range"],
            self.params["num_attention_heads"],
            self.params["num_rand_blocks"],
            self.params["max_encoder_length"],
            self.params["block_size"],
            self.params["use_bias"],
            seed=layer_idx,
            name="layer_%d" % layer_idx)
        for layer_idx in range(self.params["num_hidden_layers"])
    ])

    self.layer_norm = utils.NormLayer(self.params["hidden_size"])

  def forward(self,
              encoder_inputs,
              encoder_inputs_mask,
              training=None):
    """Return the output of the encoder layer stacks.

    Args:
      encoder_inputs: tensor with shape [batch_size, input_length, hidden_size]
      encoder_inputs_mask: Mask for encoder input. [batch_size, input_length]
      training: Boolean indicating whether the call is training or inference.

    Returns:
      Final layer encoder output. float tensor with shape
        [batch_size, input_length, hidden_size]
    """
    if self.params["attention_type"] == "block_sparse":
      encoder_length = self.params["max_encoder_length"]
      encoder_block_size = self.params["block_size"]
      encoder_inputs_mask = encoder_inputs_mask.float()
      blocked_encoder_mask = encoder_inputs_mask.reshape(
          -1, encoder_length // encoder_block_size, encoder_block_size)
      encoder_from_mask = encoder_inputs_mask.reshape(-1, 1, encoder_length, 1)
      encoder_to_mask = encoder_inputs_mask.reshape(-1, 1, 1, encoder_length)

      band_mask = attention.create_band_mask_from_inputs(
          blocked_encoder_mask, blocked_encoder_mask)

      attention_mask = None
    else:
      blocked_encoder_mask = None
      encoder_to_mask = None
      encoder_from_mask = None
      band_mask = None

      encoder_inputs_mask = encoder_inputs_mask.float()
      attention_mask = attention.create_attention_mask_from_input_mask(
          encoder_inputs_mask, encoder_inputs_mask)

    if self.params["norm_type"] == "postnorm":
      encoder_inputs = self.layer_norm(encoder_inputs)

    layer_output = encoder_inputs
    for layer in self.encoder_layers:
      if self.params["use_gradient_checkpointing"] and self.training:
        layer_output = checkpoint.checkpoint(
            layer, layer_output, attention_mask, band_mask,
            encoder_from_mask, encoder_to_mask, blocked_encoder_mask,
            use_reentrant=False)
      else:
        layer_output = layer(
            layer_output, attention_mask, band_mask,
            encoder_from_mask, encoder_to_mask, blocked_encoder_mask,
            training=training)

    if self.params["norm_type"] == "prenorm":
      layer_output = self.layer_norm(layer_output)

    return layer_output
