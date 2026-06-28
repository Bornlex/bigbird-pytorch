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

"""Masked LM / next sentence pre-training for BigBird (PyTorch)."""

import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from core import flags
from core import modeling
from core import optimization
from core import utils
from core.attention import batched_index_gather

# Special token ids in the BigBird SentencePiece vocab.
CLS_ID, SEP_ID, MASK_ID, FIRST_RANDOM_ID = 65, 66, 67, 101


class MaskedLMLayer(nn.Module):
  """Loss and log-probs for the masked LM objective."""

  def __init__(self, hidden_size, vocab_size, embeder,
               initializer=None, activation_fn=None):
    super().__init__()
    self.vocab_size = vocab_size
    self.embeder = embeder
    self.extra_layer = utils.Dense2dLayer(
        hidden_size, hidden_size, initializer, activation_fn, "transform")
    self.norm_layer = utils.NormLayer(hidden_size)
    self.output_bias = nn.Parameter(torch.zeros(vocab_size))

  def forward(self, input_tensor, label_ids=None, label_weights=None,
              masked_lm_positions=None):
    if masked_lm_positions is not None:
      input_tensor = batched_index_gather(
          input_tensor, masked_lm_positions, batch_dims=1)

    input_tensor = self.extra_layer(input_tensor)
    input_tensor = self.norm_layer(input_tensor)

    logits = self.embeder.linear(input_tensor) + self.output_bias
    log_probs = F.log_softmax(logits, dim=-1)

    if label_ids is None:
      return logits.new_zeros(()), log_probs

    per_example_loss = -log_probs.gather(
        -1, label_ids.long().unsqueeze(-1)).squeeze(-1)
    numerator = (label_weights * per_example_loss).sum()
    denominator = label_weights.sum() + 1e-5
    return numerator / denominator, log_probs


class NSPLayer(nn.Module):
  """Loss and log-probs for the next-sentence-prediction objective."""

  def __init__(self, hidden_size, initializer=None):
    super().__init__()
    self.output_weights = nn.Parameter(torch.empty(2, hidden_size))
    (initializer or utils.create_initializer())(self.output_weights)
    self.output_bias = nn.Parameter(torch.zeros(2))

  def forward(self, input_tensor, next_sentence_labels=None):
    logits = input_tensor @ self.output_weights.t() + self.output_bias
    log_probs = F.log_softmax(logits, dim=-1)

    if next_sentence_labels is None:
      return logits.new_zeros(()), log_probs

    loss = F.nll_loss(log_probs, next_sentence_labels.reshape(-1).long())
    return loss, log_probs


class BigBirdForPreTraining(nn.Module):
  """BigBird encoder with masked-LM and next-sentence-prediction heads."""

  def __init__(self, config):
    super().__init__()
    self.use_nsp = config.get("use_nsp", False)
    self.bert = modeling.BertModel(config)
    initializer = utils.create_initializer(config["initializer_range"])
    self.masked_lm = MaskedLMLayer(
        config["hidden_size"], config["vocab_size"], self.bert.embeder,
        initializer=initializer,
        activation_fn=utils.get_activation(config["hidden_act"]))
    # NSP is the legacy BERT objective; RoBERTa/BigBird drop it. Only build the
    # head (and use the pooled output) when explicitly enabled.
    self.next_sentence = (
        NSPLayer(config["hidden_size"], initializer=initializer)
        if self.use_nsp else None)

  def forward(self, features, training=None):
    sequence_output, pooled_output = self.bert(
        features["input_ids"], token_type_ids=features.get("segment_ids"),
        training=training)

    mlm_loss, mlm_log_probs = self.masked_lm(
        sequence_output,
        label_ids=features.get("masked_lm_ids"),
        label_weights=features.get("masked_lm_weights"),
        masked_lm_positions=features.get("masked_lm_positions"))

    out = {
        "loss": mlm_loss,
        "masked_lm_loss": mlm_loss,
        "masked_lm_log_probs": mlm_log_probs,
    }
    if self.use_nsp:
      nsp_loss, nsp_log_probs = self.next_sentence(
          pooled_output, features.get("next_sentence_labels"))
      out["loss"] = mlm_loss + nsp_loss
      out["next_sentence_loss"] = nsp_loss
      out["next_sentence_log_probs"] = nsp_log_probs
    return out


def whole_word_mask(subtokens, word_start_subtoken, vocab_size,
                    max_encoder_length, max_predictions_per_seq,
                    masked_lm_prob):
  """Whole-word masking of a token sequence (port of the TF numpy_masking).

  Args:
    subtokens: 1-D int array of SentencePiece ids for a document.
    word_start_subtoken: bool array of size vocab_size, True for ids that begin
      a new word (piece starting with the "▁" marker).
    vocab_size: tokenizer vocabulary size.
    max_encoder_length / max_predictions_per_seq / masked_lm_prob: as in config.

  Returns:
    dict of numpy arrays: input_ids, segment_ids, masked_lm_positions,
    masked_lm_ids, masked_lm_weights, next_sentence_labels.
  """
  # Pick a random window of the document that fits the encoder.
  end_pos = max_encoder_length - 2 + np.random.randint(
      max(1, len(subtokens) - max_encoder_length - 2))
  start_pos = max(0, end_pos - max_encoder_length + 2)
  subtokens = subtokens[start_pos:end_pos]

  # Shift the window so it starts at a word boundary.
  word_begin_mark = word_start_subtoken[subtokens]
  word_begins_pos = np.flatnonzero(word_begin_mark).astype(np.int32)
  if word_begins_pos.size == 0:
    word_begins_pos = np.arange(len(subtokens), dtype=np.int32)
    word_begin_mark = np.logical_not(word_begin_mark)
  correct_start_pos = word_begins_pos[0]
  subtokens = subtokens[correct_start_pos:]
  word_begin_mark = word_begin_mark[correct_start_pos:]
  word_begins_pos = word_begins_pos - correct_start_pos
  num_tokens = len(subtokens)

  # Group subtoken indices into whole words.
  words = np.split(np.arange(num_tokens, dtype=np.int32), word_begins_pos)[1:]

  num_to_predict = min(
      max_predictions_per_seq,
      max(1, int(round(len(word_begins_pos) * masked_lm_prob))))
  chosen = np.random.choice(len(words), num_to_predict, replace=False)
  masked_lm_positions = np.concatenate([words[i] for i in chosen])

  # A chosen word may push us over the prediction budget; drop the last
  # (possibly partial) word so we never cross a word boundary.
  if len(masked_lm_positions) > max_predictions_per_seq:
    masked_lm_positions = masked_lm_positions[:max_predictions_per_seq + 1]
    truncate_at = np.flatnonzero(word_begin_mark[masked_lm_positions])[-1]
    masked_lm_positions = masked_lm_positions[:truncate_at]

  masked_lm_positions = np.sort(masked_lm_positions)
  masked_lm_ids = subtokens[masked_lm_positions]

  # 80% [MASK], 10% random token, 10% unchanged.
  randomness = np.random.rand(len(masked_lm_positions))
  subtokens[masked_lm_positions[randomness < 0.8]] = MASK_ID
  random_index = masked_lm_positions[randomness > 0.9]
  subtokens[random_index] = np.random.randint(
      FIRST_RANDOM_ID, vocab_size, len(random_index), dtype=np.int32)

  subtokens = np.concatenate(
      [[CLS_ID], subtokens, [SEP_ID]]).astype(np.int32)

  # Pad to fixed shapes. Positions shift by 1 because of the prepended [CLS].
  subtokens = np.pad(subtokens, [0, max_encoder_length - num_tokens - 2])
  pad_out = max_predictions_per_seq - len(masked_lm_positions)
  masked_lm_weights = np.pad(
      np.ones_like(masked_lm_positions, dtype=np.float32), [0, pad_out])
  masked_lm_positions = np.pad(masked_lm_positions + 1, [0, pad_out])
  masked_lm_ids = np.pad(masked_lm_ids, [0, pad_out])

  return {
      "input_ids": subtokens,
      "segment_ids": np.zeros_like(subtokens),
      "masked_lm_positions": masked_lm_positions.astype(np.int32),
      "masked_lm_ids": masked_lm_ids.astype(np.int32),
      "masked_lm_weights": masked_lm_weights,
      "next_sentence_labels": np.zeros(1, dtype=np.int32),
  }


class TextMaskedLMDataset(Dataset):
  """Tokenizes and whole-word-masks raw text documents on the fly."""

  def __init__(self, documents, sp_model, config):
    self.documents = documents
    self.sp = sp_model
    self.config = config
    vocab_size = sp_model.GetPieceSize()
    self.word_start = np.array(
        [sp_model.IdToPiece(i)[0] == "▁" for i in range(vocab_size)])
    self.vocab_size = vocab_size

  def __len__(self):
    return len(self.documents)

  def __getitem__(self, idx):
    text = self.documents[idx]
    sub = self.config.get("substitute_newline")
    if sub is not None:
      text = text.replace("\n", sub)
    subtokens = np.array(self.sp.EncodeAsIds(text), dtype=np.int32)
    features = whole_word_mask(
        subtokens, self.word_start, self.vocab_size,
        self.config["max_encoder_length"],
        self.config["max_predictions_per_seq"],
        self.config["masked_lm_prob"])
    return {k: torch.from_numpy(v) for k, v in features.items()}


def train(model, dataset, config, device):
  model.to(device).train()
  loader = DataLoader(
      dataset, batch_size=config["train_batch_size"], shuffle=True,
      drop_last=True, num_workers=config.get("num_workers", 0))

  optimizer = optimization.get_optimizer(config, model, config["learning_rate"])
  lr_fn = optimization.get_linear_warmup_linear_decay_lr(
      config["learning_rate"], config["num_train_steps"],
      config["num_warmup_steps"])

  step = 0
  while step < config["num_train_steps"]:
    for batch in loader:
      lr = lr_fn(step)
      for group in optimizer.param_groups:
        group["lr"] = lr

      batch = {k: v.to(device) for k, v in batch.items()}
      optimizer.zero_grad()
      out = model(batch, training=True)
      out["loss"].backward()
      torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
      optimizer.step()

      if step % 100 == 0:
        print(f"step {step}  loss {out['loss'].item():.4f}  lr {lr:.2e}")
      if config["save_checkpoints_steps"] and step and (
          step % config["save_checkpoints_steps"] == 0):
        save_checkpoint(model, config["output_dir"], step)

      step += 1
      if step >= config["num_train_steps"]:
        break

  save_checkpoint(model, config["output_dir"], step)


def save_checkpoint(model, output_dir, step):
  os.makedirs(output_dir, exist_ok=True)
  path = os.path.join(output_dir, f"checkpoint-{step}.pt")
  torch.save(model.state_dict(), path)
  print(f"saved {path}")


def read_documents(input_file):
  with open(input_file, encoding="utf-8") as f:
    return [line.strip() for line in f if line.strip()]


def parse_args():
  p = argparse.ArgumentParser(description="BigBird pre-training (PyTorch)")
  p.add_argument("--input_file", required=True,
                 help="UTF-8 text file, one document per line.")
  p.add_argument("--vocab_model_file", required=True,
                 help="Path to the SentencePiece .model file.")
  p.add_argument("--output_dir", default="/tmp/bigb")
  p.add_argument("--init_checkpoint", default=None)
  p.add_argument("--max_encoder_length", type=int, default=512)
  p.add_argument("--max_predictions_per_seq", type=int, default=75)
  p.add_argument("--masked_lm_prob", type=float, default=0.15)
  p.add_argument("--substitute_newline", default=" ")
  p.add_argument("--train_batch_size", type=int, default=4)
  p.add_argument("--optimizer", default="AdamWeightDecay")
  p.add_argument("--learning_rate", type=float, default=1e-4)
  p.add_argument("--num_train_steps", type=int, default=100000)
  p.add_argument("--num_warmup_steps", type=int, default=10000)
  p.add_argument("--save_checkpoints_steps", type=int, default=1000)
  p.add_argument("--use_nsp", action="store_true")
  p.add_argument("--attention_type", default="block_sparse")
  return p.parse_args()


def main():
  import sentencepiece as spm  # local import keeps the package optional

  args = parse_args()
  config = flags.get_default_config()
  config.update(vars(args))

  if args.max_encoder_length > config["max_position_embeddings"]:
    raise ValueError(
        f"max_encoder_length {args.max_encoder_length} exceeds "
        f"max_position_embeddings {config['max_position_embeddings']}")

  sp_model = spm.SentencePieceProcessor()
  sp_model.Load(args.vocab_model_file)
  config["vocab_size"] = sp_model.GetPieceSize()

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model = BigBirdForPreTraining(config)
  if args.init_checkpoint:
    model.load_state_dict(torch.load(args.init_checkpoint, map_location="cpu"))

  dataset = TextMaskedLMDataset(read_documents(args.input_file), sp_model, config)
  train(model, dataset, config, device)


if __name__ == "__main__":
  main()
