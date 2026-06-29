"""Smoke tests for the MLM model. Run: python -m mlm.test_smoke"""

import numpy as np
import torch

from mlm.config import MLMConfig
from mlm.modeling import BigBirdForMaskedLM, BigBirdModel
from mlm.data import TextMaskedLMDataset


def small_config(**kw):
  base = dict(vocab_size=256, hidden_size=32, num_hidden_layers=2,
              num_attention_heads=4, intermediate_size=64,
              max_position_embeddings=1024, block_size=16, num_rand_blocks=2)
  base.update(kw)
  return MLMConfig(**base)


def test_full_attention():
  cfg = small_config(max_encoder_length=128, attention_type="block_sparse")
  seq = BigBirdModel(cfg).eval()(torch.randint(1, 256, (2, 128)))
  assert seq.shape == (2, 128, 32)
  print("[ok] full attention (short seq):", tuple(seq.shape))


def test_block_sparse():
  cfg = small_config(max_encoder_length=1024, attention_type="block_sparse")
  seq = BigBirdModel(cfg).eval()(torch.randint(1, 256, (2, 1024)))
  assert seq.shape == (2, 1024, 32)
  print("[ok] block sparse:", tuple(seq.shape))


def test_gradient_checkpointing():
  cfg = small_config(max_encoder_length=1024, use_gradient_checkpointing=True)
  model = BigBirdModel(cfg).train()
  seq = model(torch.randint(1, 256, (2, 1024)))
  seq.pow(2).mean().backward()
  print("[ok] gradient checkpointing backward")


def test_mlm_forward_backward():
  cfg = small_config(max_encoder_length=256)
  model = BigBirdForMaskedLM(cfg).train()
  B, P = 2, 10
  out = model(
      input_ids=torch.randint(1, 256, (B, 256)),
      masked_lm_positions=torch.randint(0, 256, (B, P)),
      masked_lm_ids=torch.randint(1, 256, (B, P)),
      masked_lm_weights=torch.ones(B, P))
  assert out["logits"].shape == (B, P, 256)
  out["loss"].backward()
  assert any(p.grad is not None for p in model.parameters())
  print("[ok] MLM forward/backward, loss =", round(float(out["loss"]), 3))


class _StubSP:
  def GetPieceSize(self): return 256
  def IdToPiece(self, i): return ("▁t" if i % 3 == 0 else "t") + str(i)
  def EncodeAsIds(self, text): return list(np.random.randint(110, 256, 300))


def test_masking_dataset():
  cfg = small_config(max_encoder_length=256, max_predictions_per_seq=20)
  ds = TextMaskedLMDataset(["doc"] * 4, _StubSP(), cfg)
  item = ds[0]
  assert item["input_ids"].shape == (256,)
  assert item["masked_lm_ids"].shape == (20,)
  print("[ok] masking dataset:", {k: tuple(v.shape) for k, v in item.items()})


if __name__ == "__main__":
  np.random.seed(0)
  torch.manual_seed(0)
  test_full_attention()
  test_block_sparse()
  test_gradient_checkpointing()
  test_mlm_forward_backward()
  test_masking_dataset()
  print("\nAll MLM smoke tests passed.")
