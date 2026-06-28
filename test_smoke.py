"""Smoke tests for the PyTorch BigBird port.

Run from the repo root:

    python test_smoke.py
"""

import torch

from core import modeling
from core import utils


def _small_config():
  cfg = utils.get_default_config()
  cfg.update({
      "vocab_size": 256,
      "hidden_size": 32,
      "num_hidden_layers": 2,
      "num_attention_heads": 4,
      "intermediate_size": 64,
      "max_position_embeddings": 1024,
      "block_size": 16,
      "num_rand_blocks": 2,
  })
  return cfg


def test_bert_full_attention():
  cfg = _small_config()
  cfg["max_encoder_length"] = 128  # <= 512 -> original_full
  cfg["attention_type"] = "block_sparse"  # will be switched to full
  model = modeling.BertModel(cfg).eval()
  ids = torch.randint(1, cfg["vocab_size"], (2, 128))
  seq, pooled = model(ids)
  assert seq.shape == (2, 128, 32), seq.shape
  assert pooled.shape == (2, 32), pooled.shape
  print("[ok] BertModel original_full:", seq.shape, pooled.shape)


def test_bert_block_sparse():
  cfg = _small_config()
  cfg["max_encoder_length"] = 1024  # > 512 -> block_sparse stays
  cfg["attention_type"] = "block_sparse"
  model = modeling.BertModel(cfg).eval()
  ids = torch.randint(1, cfg["vocab_size"], (2, 1024))
  seq, pooled = model(ids)
  assert seq.shape == (2, 1024, 32), seq.shape
  assert pooled.shape == (2, 32), pooled.shape
  print("[ok] BertModel block_sparse:", seq.shape, pooled.shape)


def test_bert_backward():
  cfg = _small_config()
  cfg["max_encoder_length"] = 1024
  cfg["attention_type"] = "block_sparse"
  model = modeling.BertModel(cfg).train()
  ids = torch.randint(1, cfg["vocab_size"], (2, 1024))
  seq, pooled = model(ids)
  loss = seq.float().pow(2).mean() + pooled.float().pow(2).mean()
  loss.backward()
  grads = [p.grad is not None for p in model.parameters() if p.requires_grad]
  assert all(grads), "some params have no grad"
  print("[ok] BertModel backward, loss=", float(loss))


def test_transformer_train_decode():
  cfg = _small_config()
  cfg["max_encoder_length"] = 256
  cfg["max_decoder_length"] = 16
  cfg["attention_type"] = "block_sparse"
  cfg["norm_type"] = "postnorm"
  cfg["beam_size"] = 2
  model = modeling.TransformerModel(cfg).eval()

  input_ids = torch.randint(1, cfg["vocab_size"], (2, 256))
  target_ids = torch.randint(1, cfg["vocab_size"], (2, 16))

  # teacher-forced decode (training path)
  (log_probs, logits, pred_ids), enc = model(
      input_ids, target_ids, training=True)
  assert logits.shape == (2, 16, cfg["vocab_size"]), logits.shape
  assert log_probs.shape == (2, 16), log_probs.shape
  print("[ok] TransformerModel train:", logits.shape, "enc", enc.shape)

  # beam-search prediction path
  with torch.no_grad():
    (log_probs, logits, pred_ids), enc = model(
        input_ids, target_ids, training=False)
  assert pred_ids.shape == (2, 16), pred_ids.shape
  print("[ok] TransformerModel predict:", pred_ids.shape)


if __name__ == "__main__":
  torch.manual_seed(0)
  test_bert_full_attention()
  test_bert_block_sparse()
  test_bert_backward()
  test_transformer_train_decode()
  print("\nAll smoke tests passed.")
