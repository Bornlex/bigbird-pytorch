"""Offline test of the checkpoint converter (no HF download).

Fabricates HuggingFace-style state dicts and checks they map cleanly onto our
model. Run: python -m mlm.test_convert
"""

import torch

from mlm.convert_checkpoint import SourceSpec, _LAYER_MAP, convert_state_dict
from mlm.modeling import BigBirdForMaskedLM

H, I, V, L, HEADS = 32, 64, 256, 2, 4


def _weight_shape(hf_name):
  if hf_name.endswith("LayerNorm"):
    return (H,)
  if hf_name == "intermediate.dense":
    return (I, H)
  if hf_name == "output.dense":
    return (H, I)
  return (H, H)  # q/k/v and attention.output.dense


def fake_hf_state_dict(spec, pos_rows):
  p = spec.prefix
  r = torch.randn
  sd = {
      f"{p}.embeddings.word_embeddings.weight": r(V, H),
      f"{p}.embeddings.position_embeddings.weight": r(pos_rows, H),
      f"{p}.embeddings.token_type_embeddings.weight": r(spec.type_vocab_size, H),
      f"{p}.embeddings.LayerNorm.weight": r(H),
      f"{p}.embeddings.LayerNorm.bias": r(H),
  }
  for i in range(spec.num_hidden_layers):
    for _, hf in _LAYER_MAP:
      w = _weight_shape(hf)
      sd[f"{p}.encoder.layer.{i}.{hf}.weight"] = r(*w)
      sd[f"{p}.encoder.layer.{i}.{hf}.bias"] = r(w[0])
  if spec.head == "roberta":
    sd["lm_head.dense.weight"] = r(H, H)
    sd["lm_head.dense.bias"] = r(H)
    sd["lm_head.layer_norm.weight"] = r(H)
    sd["lm_head.layer_norm.bias"] = r(H)
    sd["lm_head.bias"] = r(V)
  else:
    t = "cls.predictions.transform"
    sd[f"{t}.dense.weight"] = r(H, H)
    sd[f"{t}.dense.bias"] = r(H)
    sd[f"{t}.LayerNorm.weight"] = r(H)
    sd[f"{t}.LayerNorm.bias"] = r(H)
    sd["cls.predictions.bias"] = r(V)
  return sd


def _spec(prefix, head, offset, type_vocab):
  return SourceSpec(prefix=prefix, head=head, position_offset=offset,
                    hidden_act="gelu_exact" if head == "roberta" else "gelu",
                    vocab_size=V, hidden_size=H, num_hidden_layers=L,
                    num_attention_heads=HEADS, intermediate_size=I,
                    type_vocab_size=type_vocab, block_size=16, num_rand_blocks=2)


def _check(spec, pos_rows, target_pos):
  hf_sd = fake_hf_state_dict(spec, pos_rows)
  config, sd = convert_state_dict(
      hf_sd, spec, max_position_embeddings=target_pos,
      max_encoder_length=target_pos, attention_type="original_full")

  model = BigBirdForMaskedLM(config)
  res = model.load_state_dict(sd, strict=False)
  missing = [k for k in res.missing_keys
             if "rand_attn" not in k and "pooler" not in k]
  assert not missing, f"unexpectedly missing: {missing}"
  assert not res.unexpected_keys, f"unexpected: {res.unexpected_keys}"

  # Position table tiled/truncated to the target length.
  assert model.bert.embeddings.position.weight.shape == (target_pos, H)
  # Word embeddings copied verbatim.
  assert torch.equal(model.bert.embeddings.word.weight,
                     hf_sd[f"{spec.prefix}.embeddings.word_embeddings.weight"])

  out = model(input_ids=torch.randint(1, V, (2, target_pos)))
  assert out["logits"].shape == (2, target_pos, V)


def test_roberta_source():
  # 512-style: fewer position rows than target -> tiled; offset 2.
  _check(_spec("roberta", "roberta", offset=2, type_vocab=1),
         pos_rows=2 + 64, target_pos=128)
  print("[ok] roberta-base mapping (position tiling, offset 2)")


def test_bigbird_source():
  # Already-long checkpoint: position rows match target; offset 0.
  _check(_spec("bert", "bert", offset=0, type_vocab=2),
         pos_rows=128, target_pos=128)
  print("[ok] bigbird-roberta mapping (direct position copy)")


if __name__ == "__main__":
  torch.manual_seed(0)
  test_roberta_source()
  test_bigbird_source()
  print("\nAll converter tests passed.")
