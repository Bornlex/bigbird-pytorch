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

"""Helper and utility functions (PyTorch port)."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


############################### SHAPE UTILS ####################################


def get_shape_list(tensor):
  """Returns the shape of `tensor` as a python list of ints."""
  return list(tensor.shape)


############################### INITIALIZERS ###################################


def create_initializer(initializer_range=0.02):
  """Returns a callable that initializes a tensor with a truncated normal."""

  def _init(tensor):
    return nn.init.trunc_normal_(
        tensor, mean=0.0, std=initializer_range,
        a=-2 * initializer_range, b=2 * initializer_range)

  return _init


############################### DENSE LAYERS ###################################


class Dense3dLayer(nn.Module):
  """A dense layer with 3D kernel."""

  def __init__(self,
               num_attention_heads,
               size_per_head,
               initializer,
               activation,
               name=None,
               head_first=False,
               use_bias=True):
    super().__init__()
    self.num_attention_heads = num_attention_heads
    self.size_per_head = size_per_head
    self.activation = activation
    self.head_first = head_first
    self.use_bias = use_bias

    hidden_size = num_attention_heads * size_per_head
    self.w = nn.Parameter(torch.empty(hidden_size, hidden_size))
    initializer(self.w)
    if use_bias:
      self.b = nn.Parameter(torch.zeros(hidden_size))
    else:
      self.register_parameter("b", None)

  def forward(self, input_tensor):
    # input_tensor: [batch, seq_length, hidden_size]
    hidden_size = self.num_attention_heads * self.size_per_head
    reshape_w = self.w.reshape(
        hidden_size, self.num_attention_heads, self.size_per_head)
    if self.head_first:
      ret = torch.einsum("abc,cde->adbe", input_tensor, reshape_w)
    else:
      ret = torch.einsum("abc,cde->abde", input_tensor, reshape_w)

    if self.use_bias:
      if self.head_first:
        reshape_b = self.b.reshape(
            1, self.num_attention_heads, 1, self.size_per_head)
      else:
        reshape_b = self.b.reshape(self.num_attention_heads, self.size_per_head)
      ret = ret + reshape_b

    if self.activation is not None:
      return self.activation(ret)
    return ret


class Dense3dProjLayer(nn.Module):
  """A dense layer with 3D kernel for projection."""

  def __init__(self,
               num_attention_heads,
               size_per_head,
               initializer,
               activation,
               name=None,
               use_bias=True):
    super().__init__()
    self.num_attention_heads = num_attention_heads
    self.size_per_head = size_per_head
    self.activation = activation
    self.use_bias = use_bias

    hidden_size = num_attention_heads * size_per_head
    self.w = nn.Parameter(torch.empty(hidden_size, hidden_size))
    initializer(self.w)
    if use_bias:
      self.b = nn.Parameter(torch.zeros(hidden_size))
    else:
      self.register_parameter("b", None)

  def forward(self, input_tensor):
    # input_tensor: [batch, from_seq_length, num_attention_heads, size_per_head]
    hidden_size = self.num_attention_heads * self.size_per_head
    reshape_w = self.w.reshape(
        self.num_attention_heads, self.size_per_head, hidden_size)
    ret = torch.einsum("BFNH,NHD->BFD", input_tensor, reshape_w)
    if self.use_bias:
      ret = ret + self.b
    if self.activation is not None:
      return self.activation(ret)
    return ret


class Dense2dLayer(nn.Module):
  """A dense layer with 2D kernel."""

  def __init__(self,
               input_size,
               output_size,
               initializer,
               activation,
               name=None,
               use_bias=True):
    super().__init__()
    self.activation = activation
    self.use_bias = use_bias

    self.w = nn.Parameter(torch.empty(input_size, output_size))
    initializer(self.w)
    if use_bias:
      self.b = nn.Parameter(torch.zeros(output_size))
    else:
      self.register_parameter("b", None)

  def forward(self, input_tensor):
    ret = torch.einsum("abc,cd->abd", input_tensor, self.w)
    if self.use_bias:
      ret = ret + self.b
    if self.activation is not None:
      return self.activation(ret)
    return ret


class SimpleDenseLayer(nn.Module):
  """A simple dense layer with 2D kernel (rank-2 input)."""

  def __init__(self,
               input_size,
               output_size,
               initializer,
               activation,
               name=None,
               use_bias=True):
    super().__init__()
    self.activation = activation
    self.use_bias = use_bias

    self.w = nn.Parameter(torch.empty(input_size, output_size))
    initializer(self.w)
    if use_bias:
      self.b = nn.Parameter(torch.zeros(output_size))
    else:
      self.register_parameter("b", None)

  def forward(self, input_tensor):
    ret = torch.einsum("ab,bc->ac", input_tensor, self.w)
    if self.use_bias:
      ret = ret + self.b
    if self.activation is not None:
      return self.activation(ret)
    return ret


def gelu(x):
  """Gaussian Error Linear Unit (tanh approximation, matches original)."""
  cdf = 0.5 * (1.0 + torch.tanh(
      math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
  return x * cdf


def get_activation(activation_string):
  """Map a name to an activation function ("gelu", "relu", "tanh", "linear")."""
  if not isinstance(activation_string, str):
    return activation_string
  if not activation_string:
    return None

  act = activation_string.lower()
  if act == "linear":
    return None
  elif act == "relu":
    return F.relu
  elif act == "gelu":
    return gelu
  elif act == "tanh":
    return torch.tanh
  else:
    raise ValueError("Unsupported activation: %s" % act)


############################## NORM LAYERS #####################################


class NormLayer(nn.Module):
  """Layer normalization over the last axis (matches TF NormLayer)."""

  def __init__(self, hdim, dtype=torch.float32, name="LayerNorm"):
    super().__init__()
    self._dtype = dtype
    self.gamma = nn.Parameter(torch.ones(hdim))
    self.beta = nn.Parameter(torch.zeros(hdim))

  def forward(self, inputs):
    variance_epsilon = 1e-12 if self._dtype != torch.float16 else 1e-3
    mean = inputs.mean(dim=-1, keepdim=True)
    # Biased variance, matching tf.nn.moments.
    variance = inputs.var(dim=-1, keepdim=True, unbiased=False)
    outputs = (inputs - mean) / torch.sqrt(variance + variance_epsilon)
    return outputs * self.gamma + self.beta


############################# EMBEDDING LAYER ##################################


class EmbeddingLayer(nn.Module):
  """An embedding layer."""

  def __init__(self,
               vocab_size,
               emb_dim,
               initializer,
               scale_emb=False,
               use_token_type=False,
               num_token_types=16,
               use_position_embeddings=True,
               max_position_embeddings=4096,
               dropout_prob=0.0,
               name="embeddings"):
    super().__init__()
    self.vocab_size = vocab_size
    self.emb_dim = emb_dim
    self.scale_emb = scale_emb
    self.num_token_types = num_token_types
    self.max_position_embeddings = max_position_embeddings
    self.dropout_prob = dropout_prob

    self.word_embeddings = nn.Parameter(torch.empty(vocab_size, emb_dim))
    initializer(self.word_embeddings)

    if use_token_type:
      self.token_type_table = nn.Parameter(
          torch.empty(num_token_types, emb_dim))
      initializer(self.token_type_table)
    else:
      self.register_parameter("token_type_table", None)

    if use_position_embeddings:
      self.position_embeddings = nn.Parameter(
          torch.empty(max_position_embeddings, emb_dim))
      initializer(self.position_embeddings)
    else:
      self.register_parameter("position_embeddings", None)

    self.dropout = nn.Dropout(dropout_prob)

  def forward(self,
              input_ids,
              seq_length,
              start_pos=0,
              token_type_ids=None,
              training=None):
    if input_ids is None:
      return None

    # subtoken embedding
    output = F.embedding(input_ids, self.word_embeddings)

    if self.scale_emb:
      output = output * self.emb_dim ** 0.5

    if self.token_type_table is not None:
      one_hot_ids = F.one_hot(
          token_type_ids.long(), num_classes=self.num_token_types).to(
              output.dtype)
      token_type_embeddings = torch.tensordot(
          one_hot_ids, self.token_type_table, dims=1)
      output = output + token_type_embeddings

    if self.position_embeddings is not None:
      position_embeddings = self.position_embeddings[
          start_pos:start_pos + seq_length, :]
      output = output + position_embeddings.unsqueeze(0)

    if training is None:
      training = self.training
    if training and self.dropout_prob > 0:
      output = self.dropout(output)
    return output

  def linear(self, x):
    # Project [..., hidden_size] to [..., vocab_size] with tied weights.
    return torch.tensordot(x, self.word_embeddings, dims=([-1], [1]))


########################## DEFAULT CONFIG UTILS ################################


def get_default_config():
  """Default values for BigBird."""

  default_config = {
      # transformer basic configs
      "attention_probs_dropout_prob": 0.1,
      "hidden_act": "gelu",
      "hidden_dropout_prob": 0.1,
      "hidden_size": 768,
      "initializer_range": 0.02,
      "intermediate_size": 3072,
      "max_position_embeddings": 4096,
      "num_attention_heads": 12,
      "num_hidden_layers": 12,
      "type_vocab_size": 2,
      "use_bias": True,
      "rescale_embedding": False,
      "scope": "bert",
      "use_gradient_checkpointing": False,
      # sparse mask configs
      "attention_type": "block_sparse",
      "norm_type": "postnorm",
      "block_size": 16,
      "num_rand_blocks": 3,
      # common bert configs
      "max_encoder_length": 1024,
      "max_decoder_length": 64,
      "couple_encoder_decoder": False,
      "beam_size": 5,
      "alpha": 0.7,
      "label_smoothing": 0.1,
      "weight_decay_rate": 0.01,
      "optimizer_beta1": 0.9,
      "optimizer_beta2": 0.999,
      "optimizer_epsilon": 1e-6,
      "vocab_size": 32000,
  }

  return default_config
