"""Smoke tests for the seq2seq model. Run: python -m transformer.test_smoke"""

import torch

from transformer.config import TransformerConfig
from transformer.modeling import TransformerModel


def small_config(**kw):
  base = dict(vocab_size=256, hidden_size=32, num_hidden_layers=2,
              num_attention_heads=4, intermediate_size=64,
              max_position_embeddings=1024, block_size=16, num_rand_blocks=2,
              max_encoder_length=256, max_decoder_length=16, beam_size=2)
  base.update(kw)
  return TransformerConfig(**base)


def test_train():
  cfg = small_config()
  model = TransformerModel(cfg).train()
  input_ids = torch.randint(1, 256, (2, 256))
  target_ids = torch.randint(1, 256, (2, 16))
  out = model(input_ids, target_ids, training=True)
  assert out["logits"].shape == (2, 16, 256)
  out["loss"].backward()
  assert any(p.grad is not None for p in model.parameters())
  print("[ok] train forward/backward, loss =", round(float(out["loss"]), 3))


def test_generate():
  cfg = small_config()
  model = TransformerModel(cfg).eval()
  input_ids = torch.randint(1, 256, (2, 256))
  out = model(input_ids, training=False)
  assert out["pred_ids"].shape == (2, 16)
  print("[ok] beam-search generate:", tuple(out["pred_ids"].shape))


def test_prenorm():
  cfg = small_config(norm_type="prenorm", rescale_embedding=True)
  model = TransformerModel(cfg).train()
  input_ids = torch.randint(1, 256, (2, 256))
  target_ids = torch.randint(1, 256, (2, 16))
  out = model(input_ids, target_ids, training=True)
  out["loss"].backward()
  print("[ok] prenorm train backward, loss =", round(float(out["loss"]), 3))


if __name__ == "__main__":
  torch.manual_seed(0)
  test_train()
  test_generate()
  test_prenorm()
  print("\nAll transformer smoke tests passed.")
