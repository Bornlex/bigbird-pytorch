"""Whole-word masking and a Dataset that masks raw text on the fly."""

import numpy as np
import torch
from torch.utils.data import Dataset

# Special token ids in the BigBird SentencePiece vocab.
CLS_ID, SEP_ID, MASK_ID, FIRST_RANDOM_ID = 65, 66, 67, 101


def whole_word_mask(subtokens, word_start, vocab_size, max_encoder_length,
                    max_predictions_per_seq, masked_lm_prob):
  """Mask whole words in a token sequence (80% mask, 10% random, 10% keep)."""
  # Pick a random window of the document that fits the encoder.
  end = max_encoder_length - 2 + np.random.randint(
      max(1, len(subtokens) - max_encoder_length - 2))
  start = max(0, end - max_encoder_length + 2)
  subtokens = subtokens[start:end]

  # Shift the window so it begins at a word boundary.
  begin_mark = word_start[subtokens]
  begins = np.flatnonzero(begin_mark).astype(np.int32)
  if begins.size == 0:
    begins = np.arange(len(subtokens), dtype=np.int32)
    begin_mark = np.logical_not(begin_mark)
  subtokens = subtokens[begins[0]:]
  begin_mark = begin_mark[begins[0]:]
  begins = begins - begins[0]
  num_tokens = len(subtokens)

  words = np.split(np.arange(num_tokens, dtype=np.int32), begins)[1:]
  num_to_predict = min(max_predictions_per_seq,
                       max(1, round(len(begins) * masked_lm_prob)))
  chosen = np.random.choice(len(words), num_to_predict, replace=False)
  positions = np.concatenate([words[i] for i in chosen])

  # A chosen word may overflow the budget; drop the last partial word.
  if len(positions) > max_predictions_per_seq:
    positions = positions[:max_predictions_per_seq + 1]
    positions = positions[:np.flatnonzero(begin_mark[positions])[-1]]

  positions = np.sort(positions)
  label_ids = subtokens[positions]
  rnd = np.random.rand(len(positions))
  subtokens[positions[rnd < 0.8]] = MASK_ID
  rand_pos = positions[rnd > 0.9]
  subtokens[rand_pos] = np.random.randint(
      FIRST_RANDOM_ID, vocab_size, len(rand_pos), dtype=np.int32)

  subtokens = np.concatenate([[CLS_ID], subtokens, [SEP_ID]]).astype(np.int32)
  subtokens = np.pad(subtokens, [0, max_encoder_length - num_tokens - 2])

  pad = max_predictions_per_seq - len(positions)
  weights = np.pad(np.ones(len(positions), np.float32), [0, pad])
  positions = np.pad(positions + 1, [0, pad]).astype(np.int32)  # +1 for [CLS]
  label_ids = np.pad(label_ids, [0, pad]).astype(np.int32)

  return {
      "input_ids": subtokens,
      "masked_lm_positions": positions,
      "masked_lm_ids": label_ids,
      "masked_lm_weights": weights,
  }


class TextMaskedLMDataset(Dataset):
  """Tokenizes and whole-word-masks UTF-8 documents (one per item)."""

  def __init__(self, documents, sp_model, config):
    self.documents = documents
    self.sp = sp_model
    self.config = config
    self.vocab_size = sp_model.GetPieceSize()
    self.word_start = np.array(
        [sp_model.IdToPiece(i)[0] == "▁" for i in range(self.vocab_size)])

  def __len__(self):
    return len(self.documents)

  def __getitem__(self, idx):
    text = self.documents[idx]
    if self.config.substitute_newline is not None:
      text = text.replace("\n", self.config.substitute_newline)
    subtokens = np.array(self.sp.EncodeAsIds(text), dtype=np.int32)
    features = whole_word_mask(
        subtokens, self.word_start, self.vocab_size,
        self.config.max_encoder_length, self.config.max_predictions_per_seq,
        self.config.masked_lm_prob)
    return {k: torch.from_numpy(v) for k, v in features.items()}
