"""Convert a HuggingFace RoBERTa / BigBird checkpoint to this MLM model.

Warm-starts our encoder-only BigBird from pretrained weights. Two sources:

  * google/bigbird-roberta-base  -- already 4096 ctx + sparse-trained; weights
    drop in directly (best starting point).
  * roberta-base                 -- 512 ctx full-attention RoBERTa; its 512
    position embeddings are tiled up to fill the longer table, then you adapt
    with sparse attention (the original BigBird recipe).

The encoder/FFN/LayerNorm weights are identical in shape and meaning to ours,
and because both sides use nn.Linear no transpose is needed.

Usage:
  python -m mlm.convert_checkpoint --source google/bigbird-roberta-base \
      --output bigbird_init.pt --max_position_embeddings 4096
"""

import argparse
from dataclasses import dataclass

import torch

from mlm.config import MLMConfig
from mlm.modeling import BigBirdForMaskedLM

# Per-layer module name map: ours -> HuggingFace (BERT/RoBERTa share this).
_LAYER_MAP = [
    ("attn.query", "attention.self.query"),
    ("attn.key", "attention.self.key"),
    ("attn.value", "attention.self.value"),
    ("attn_out", "attention.output.dense"),
    ("attn_norm", "attention.output.LayerNorm"),
    ("intermediate", "intermediate.dense"),
    ("output", "output.dense"),
    ("output_norm", "output.LayerNorm"),
]

_ACT_MAP = {"gelu": "gelu_exact", "gelu_new": "gelu", "relu": "relu"}


@dataclass
class SourceSpec:
  prefix: str            # top-level key prefix: "roberta" or "bert"
  head: str              # MLM head style: "roberta" or "bert"
  position_offset: int   # RoBERTa stores positions starting at index 2
  hidden_act: str        # our activation name
  vocab_size: int
  hidden_size: int
  num_hidden_layers: int
  num_attention_heads: int
  intermediate_size: int
  block_size: int
  num_rand_blocks: int


def convert_state_dict(hf_sd, spec, max_position_embeddings,
                       max_encoder_length, attention_type):
  """Map a HuggingFace state_dict to our (config, state_dict).

  We deliberately drop the source's token-type embeddings and pooler: this
  model does single-segment MLM, so they are unused here.
  """
  config = MLMConfig(
      vocab_size=spec.vocab_size,
      hidden_size=spec.hidden_size,
      num_hidden_layers=spec.num_hidden_layers,
      num_attention_heads=spec.num_attention_heads,
      intermediate_size=spec.intermediate_size,
      hidden_act=spec.hidden_act,
      max_position_embeddings=max_position_embeddings,
      max_encoder_length=max_encoder_length,
      attention_type=attention_type,
      block_size=spec.block_size,
      num_rand_blocks=spec.num_rand_blocks)

  p = spec.prefix
  sd = {}

  # Embeddings.
  sd["bert.embeddings.word.weight"] = hf_sd[f"{p}.embeddings.word_embeddings.weight"]
  # Embedding LayerNorm == our initial (postnorm) encoder norm.
  sd["bert.encoder.norm.weight"] = hf_sd[f"{p}.embeddings.LayerNorm.weight"]
  sd["bert.encoder.norm.bias"] = hf_sd[f"{p}.embeddings.LayerNorm.bias"]

  # Position embeddings: drop RoBERTa's offset, then tile/truncate to length.
  pos = hf_sd[f"{p}.embeddings.position_embeddings.weight"][spec.position_offset:]
  n = max_position_embeddings
  if pos.shape[0] < n:
    reps = (n + pos.shape[0] - 1) // pos.shape[0]
    pos = pos.repeat(reps, 1)
  sd["bert.embeddings.position.weight"] = pos[:n].clone()

  # Encoder layers.
  for i in range(spec.num_hidden_layers):
    for ours, hf in _LAYER_MAP:
      for suffix in ("weight", "bias"):
        sd[f"bert.encoder.layers.{i}.{ours}.{suffix}"] = hf_sd[
            f"{p}.encoder.layer.{i}.{hf}.{suffix}"]

  # MLM head.
  if spec.head == "roberta":
    sd["transform.weight"] = hf_sd["lm_head.dense.weight"]
    sd["transform.bias"] = hf_sd["lm_head.dense.bias"]
    sd["transform_norm.weight"] = hf_sd["lm_head.layer_norm.weight"]
    sd["transform_norm.bias"] = hf_sd["lm_head.layer_norm.bias"]
    sd["bias"] = hf_sd["lm_head.bias"]
  else:
    t = "cls.predictions.transform"
    sd["transform.weight"] = hf_sd[f"{t}.dense.weight"]
    sd["transform.bias"] = hf_sd[f"{t}.dense.bias"]
    sd["transform_norm.weight"] = hf_sd[f"{t}.LayerNorm.weight"]
    sd["transform_norm.bias"] = hf_sd[f"{t}.LayerNorm.bias"]
    sd["bias"] = hf_sd["cls.predictions.bias"]

  return config, sd


def _spec_from_hf(hf_model, source):
  c = hf_model.config
  sd = hf_model.state_dict()
  if any(k.startswith("roberta.") for k in sd):
    prefix = "roberta"
  elif any(k.startswith("bert.") for k in sd):
    prefix = "bert"
  else:
    raise ValueError(f"Unrecognized checkpoint layout for {source}")
  head = "roberta" if any(k.startswith("lm_head.") for k in sd) else "bert"
  return SourceSpec(
      prefix=prefix,
      head=head,
      position_offset=getattr(c, "pad_token_id", 0) + 1 if prefix == "roberta"
      else 0,
      hidden_act=_ACT_MAP.get(c.hidden_act, "gelu"),
      vocab_size=c.vocab_size,
      hidden_size=c.hidden_size,
      num_hidden_layers=c.num_hidden_layers,
      num_attention_heads=c.num_attention_heads,
      intermediate_size=c.intermediate_size,
      block_size=getattr(c, "block_size", 64),
      num_rand_blocks=getattr(c, "num_random_blocks", 3))


def from_pretrained(source, max_position_embeddings=4096,
                    max_encoder_length=1024, attention_type="block_sparse"):
  """Download a HF checkpoint and return (MLMConfig, our state_dict)."""
  from transformers import AutoModelForMaskedLM  # optional dependency

  hf_model = AutoModelForMaskedLM.from_pretrained(source)
  spec = _spec_from_hf(hf_model, source)
  return convert_state_dict(
      hf_model.state_dict(), spec, max_position_embeddings,
      max_encoder_length, attention_type)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--source", default="google/bigbird-roberta-base",
                      help="HF model id (e.g. roberta-base).")
  parser.add_argument("--output", required=True, help="Output .pt path.")
  parser.add_argument("--max_position_embeddings", type=int, default=4096)
  parser.add_argument("--max_encoder_length", type=int, default=1024)
  parser.add_argument("--attention_type", default="block_sparse")
  args = parser.parse_args()

  config, sd = from_pretrained(
      args.source, args.max_position_embeddings, args.max_encoder_length,
      args.attention_type)

  model = BigBirdForMaskedLM(config)
  result = model.load_state_dict(sd, strict=False)
  missing = [k for k in result.missing_keys if "rand_attn" not in k]
  print(f"loaded from {args.source}")
  print(f"  missing (left initialized): {missing or 'none'}")
  print(f"  unexpected: {result.unexpected_keys or 'none'}")

  torch.save({"config": config, "state_dict": model.state_dict()}, args.output)
  print(f"saved {args.output}")


if __name__ == "__main__":
  main()
