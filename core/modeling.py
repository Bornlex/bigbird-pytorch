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

"""The main BigBird model and related functions (PyTorch port)."""

import copy
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from core import decoder
from core import encoder
from core import utils


def _sequence_mask(lengths, maxlen, dtype=torch.int32, device=None):
  """Equivalent of tf.sequence_mask."""
  row = torch.arange(maxlen, device=device).unsqueeze(0)
  mask = row < lengths.unsqueeze(1)
  return mask.to(dtype)


class BertModel(nn.Module):
  """BERT model ("Bidirectional Encoder Representations from Transformers").

  Example usage:

  ```python
  input_ids = torch.tensor([[31, 51, 99], [15, 5, 0]])
  token_type_ids = torch.tensor([[0, 0, 1], [0, 2, 0]])

  params = utils.get_default_config()
  params['vocab_size'] = 32000
  ...
  model = modeling.BertModel(params)
  sequence_output, pooled_output = model(
      input_ids=input_ids, token_type_ids=token_type_ids)
  ```
  """

  def __init__(self, params):
    super().__init__()
    self.params = copy.deepcopy(params)
    self.scope = params["scope"]

    # validate params
    self._pad_size = 0
    if params["max_encoder_length"] <= 512:
      logging.info("Switching to full attention for short sequences")
      self.params["attention_type"] = "original_full"
    if self.params["attention_type"] in ("simulated_sparse", "block_sparse"):
      if params["max_encoder_length"] % params["block_size"]:
        logging.info("Expand max_encoder_length to next multiple of block_size")
        self.params["max_encoder_length"] = (
            params["max_encoder_length"] // params["block_size"] +
            1) * params["block_size"]
        self._pad_size = (
            self.params["max_encoder_length"] - params["max_encoder_length"])

    self.embeder = utils.EmbeddingLayer(
        vocab_size=self.params["vocab_size"],
        emb_dim=self.params["hidden_size"],
        initializer=utils.create_initializer(self.params["initializer_range"]),
        scale_emb=self.params["rescale_embedding"],
        use_token_type=True,
        num_token_types=self.params["type_vocab_size"],
        use_position_embeddings=True,
        max_position_embeddings=self.params["max_position_embeddings"],
        dropout_prob=self.params["hidden_dropout_prob"])
    self.encoder = encoder.EncoderStack(self.params)
    self.pooler = utils.SimpleDenseLayer(
        input_size=self.params["hidden_size"],
        output_size=self.params["hidden_size"],
        initializer=utils.create_initializer(self.params["initializer_range"]),
        activation=torch.tanh,
        name="pooler/dense")

  def pad(self, x):
    if self._pad_size:
      return F.pad(x, (0, self._pad_size))
    return x

  def forward(self, input_ids, token_type_ids=None, training=None):
    # Returns (sequence_output [B, S, H], pooled_output [B, H]).
    input_ids = self.pad(input_ids)

    if token_type_ids is None:
      token_type_ids = torch.zeros_like(input_ids, dtype=torch.int32)
    else:
      token_type_ids = self.pad(token_type_ids)

    embedding_output = self.embeder(
        input_ids, self.params["max_encoder_length"],
        token_type_ids=token_type_ids, training=training)

    input_mask = (input_ids > 0).int()

    sequence_output = self.encoder(embedding_output, input_mask, training)

    first_token_tensor = sequence_output[:, 0, :]
    pooled_output = self.pooler(first_token_tensor)

    return sequence_output, pooled_output


class TransformerModel(nn.Module):
  """Encoder-Decoder transformer model."""

  def __init__(self, params):
    super().__init__()
    self.params = copy.deepcopy(params)
    self.scope = params["scope"]

    self._pad_size = 0
    if params["max_encoder_length"] <= 512:
      logging.info("Switching to full attention for short sequences")
      self.params["attention_type"] = "original_full"
    if self.params["attention_type"] in ("simulated_sparse", "block_sparse"):
      if params["max_encoder_length"] % params["block_size"]:
        logging.info("Expand max_encoder_length to next multiple of block_size")
        self.params["max_encoder_length"] = (
            params["max_encoder_length"] // params["block_size"] +
            1) * params["block_size"]
        self._pad_size = (
            self.params["max_encoder_length"] - params["max_encoder_length"])

    self.embeder = utils.EmbeddingLayer(
        vocab_size=self.params["vocab_size"],
        emb_dim=self.params["hidden_size"],
        initializer=utils.create_initializer(self.params["initializer_range"]),
        scale_emb=self.params["rescale_embedding"],
        use_token_type=False,
        num_token_types=None,
        use_position_embeddings=True,
        max_position_embeddings=self.params["max_position_embeddings"],
        dropout_prob=self.params["hidden_dropout_prob"])
    self.encoder = encoder.EncoderStack(self.params)
    self.decoder = decoder.DecoderStack(self.params)

  def pad(self, x):
    if self._pad_size:
      return F.pad(x, (0, self._pad_size))
    return x

  def _encode(self, input_ids, training=None):
    """Generate continuous representation for ids."""
    input_ids = self.pad(input_ids)
    input_embs = self.embeder(
        input_ids, self.params["max_encoder_length"], training=training)
    input_mask = (input_ids > 0).int()
    encoder_output = self.encoder(input_embs, input_mask, training=training)
    return encoder_output, input_mask

  def _get_start_token_ids(self, tensor_for_shape):
    start_token_id = 2
    batch_size = utils.get_shape_list(tensor_for_shape)[0]
    return torch.ones(
        batch_size, dtype=torch.int32,
        device=tensor_for_shape.device) * start_token_id

  def get_inputs_from_targets(self, targets, start_token_ids):
    """Converts target ids to input ids, i.e. adds <s> and removes last."""
    length = torch.count_nonzero(targets, dim=1).int()
    inputs = torch.cat([start_token_ids.unsqueeze(1), targets], 1)
    mask = _sequence_mask(
        length, self.params["max_decoder_length"] + 1,
        dtype=inputs.dtype, device=inputs.device)
    inputs = (mask * inputs)[:, :-1]
    return inputs

  def _decode(self, target_ids, target_mask, start_token_ids,
              encoder_output, encoder_mask, training=None):
    """Compute likelihood of target tokens under the model."""
    input_ids = self.get_inputs_from_targets(target_ids, start_token_ids)

    input_embs = self.embeder(
        input_ids, self.params["max_decoder_length"], training=training)

    outputs = self.decoder(
        input_embs, target_mask, encoder_output, encoder_mask,
        training=training)

    logits = self.embeder.linear(outputs)
    output_ids = torch.argmax(logits, dim=-1).int()

    log_probs = -F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        target_ids.reshape(-1).long(),
        reduction="none").reshape(target_ids.shape)
    log_probs = torch.where(
        target_ids > 0, log_probs, torch.zeros_like(log_probs))

    return (log_probs, logits, output_ids)

  def _init_cache(self, batch_size, device):
    """Initialize cache for decoding."""
    max_decode_len = self.params["max_decoder_length"]
    num_heads = self.params["num_attention_heads"]
    head_size = int(self.params["hidden_size"] / num_heads)

    cache = {}
    for layer in range(self.params["num_hidden_layers"]):
      cache["layer_%d" % layer] = {
          "k": torch.zeros(
              batch_size, num_heads, max_decode_len, head_size, device=device),
          "v": torch.zeros(
              batch_size, num_heads, max_decode_len, head_size, device=device),
      }
    return cache

  def _get_symbols_to_logits_fn(self, decoder_self_attention_mask):
    """Returns a decoding function that calculates logits of the next tokens."""

    def _symbols_to_logits_fn(target_ids, cache, i):
      decoder_input = target_ids[:, max(0, i - 1):max(0, i - 1) + 1]
      self_attention_mask = decoder_self_attention_mask[:, :, i:i + 1, :]

      decoder_input = self.embeder(
          decoder_input, 1, start_pos=i, training=False)

      decoder_output = self.decoder(
          decoder_input, self_attention_mask,
          cache.get("encoder_output"), cache.get("encoder_mask"),
          cache=cache, decode_i=i, training=False)

      logits = self.embeder.linear(decoder_output)
      logits = logits.squeeze(1)
      return logits

    return _symbols_to_logits_fn

  def _predict(self, target_ids, target_mask, start_token_ids,
               encoder_output, encoder_mask):
    """Beam decode output tokens and probabilities."""
    batch_size = utils.get_shape_list(start_token_ids)[0]
    device = start_token_ids.device
    end_token_id = 1

    symbols_to_logits_fn = self._get_symbols_to_logits_fn(target_mask)

    cache = self._init_cache(batch_size, device)
    if encoder_output is not None:
      cache["encoder_output"] = encoder_output
      cache["encoder_mask"] = encoder_mask

    decoded_ids = decoder.left2right_decode(
        symbols_to_logits_fn,
        start_token_ids,
        cache,
        batch_size,
        self.params["max_decoder_length"],
        vocab_size=self.params["vocab_size"],
        beam_size=self.params["beam_size"],
        beam_start=5,
        beam_alpha=self.params["alpha"],
        beam_min=0,
        beam_max=-1,
        eos_id=end_token_id,
        device=device)

    output_ids = decoded_ids.int()

    calc_ids = output_ids if target_ids is None else target_ids
    output_log_probs, output_logits, _ = self._decode(
        calc_ids, target_mask, start_token_ids,
        encoder_output, encoder_mask, training=False)

    return (output_log_probs, output_logits, output_ids)

  def _decode_and_predict(self, target_ids, encoder_output, encoder_mask,
                          training=None):
    """Decodes a sequence given the input and the encoder."""
    start_token_ids = self._get_start_token_ids(encoder_output)

    target_mask = decoder.create_self_attention_mask(
        self.params["max_decoder_length"], device=encoder_output.device)

    if training:
      predictions = self._decode(
          target_ids, target_mask, start_token_ids,
          encoder_output, encoder_mask, training=True)
    else:
      predictions = self._predict(
          target_ids, target_mask, start_token_ids,
          encoder_output, encoder_mask)

    return predictions

  def forward(self, input_ids, target_ids=None, training=None):
    encoder_output, encoder_mask = self._encode(input_ids, training=training)
    predictions = self._decode_and_predict(
        target_ids, encoder_output, encoder_mask, training=training)
    return predictions, encoder_output
