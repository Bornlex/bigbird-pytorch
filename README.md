# BigBird (PyTorch port)

A faithful PyTorch translation of Google's official TensorFlow
[BigBird](https://arxiv.org/abs/2007.14062) implementation. The module layout,
layer names and computation mirror the original so the two can be cross-checked
and TF checkpoints can be converted.

## Layout

```
core/
  utils.py         # dense/embedding/norm layers, activations, default config
  attention.py     # MultiHeadedAttentionLayer + block-sparse attention kernel
  encoder.py       # Pre/Postnorm encoder layers + EncoderStack
  decoder.py       # Pre/Postnorm decoder layers + DecoderStack + greedy decode
  beam_search.py   # beam search (ported from Pegasus)
  optimization.py  # AdamWeightDecay optimizer + LR schedules
  flags.py         # config helpers (replaces absl flags)
pretrain/
  run_pretraining.py  # MLM + NSP heads, whole-word masking, training loop
test_smoke.py      # shape/grad/decode smoke tests
```

## What was translated

| TF concept | PyTorch equivalent |
|---|---|
| `tf.keras.layers.Layer` | `torch.nn.Module` |
| `tf.compat.v1.get_variable` | `nn.Parameter` (+ truncated-normal init) |
| `tf.einsum` | `torch.einsum` (same subscripts) |
| `tf.gather(..., batch_dims=k)` | `attention.batched_index_gather` |
| `tf.nn.batch_normalization` (layer norm) | `utils.NormLayer` |
| `RecomputingDropout` | `nn.Dropout` |
| `recompute_grad` / gradient checkpointing | `torch.utils.checkpoint.checkpoint` |
| `tf.while_loop` beam/greedy decode | plain Python loops over tensors |
| `AdamWeightDecayOptimizer` | `optimization.AdamWeightDecayOptimizer` |

## Usage

```python
import torch
from core import modeling, utils

params = utils.get_default_config()
params.update(dict(
    vocab_size=32000,
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12,
    intermediate_size=3072,
    max_encoder_length=1024,
    attention_type="block_sparse",   # or "original_full" / "simulated_sparse"
    block_size=16,
    num_rand_blocks=3,
))

model = modeling.BertModel(params).eval()
input_ids = torch.randint(1, params["vocab_size"], (2, 1024))
sequence_output, pooled_output = model(input_ids)
```

Encoder-decoder (summarization-style) model:

```python
model = modeling.TransformerModel(params)
# training (teacher forcing):
(log_probs, logits, pred_ids), enc = model(input_ids, target_ids, training=True)
# inference (beam search):
(log_probs, logits, pred_ids), enc = model(input_ids, target_ids, training=False)
```

## Pre-training

`pretrain/run_pretraining.py` ports the masked-LM (+ optional NSP) objective.
The TF data pipeline (TFRecords / tfds / TPUEstimator) is replaced by a plain
`torch.utils.data.Dataset` that tokenizes and whole-word-masks raw text on the
fly, plus a standard training loop.

```bash
python -m pretrain.run_pretraining \
    --input_file docs.txt \
    --vocab_model_file vocab/pegasus.model \
    --output_dir /tmp/bigb \
    --max_encoder_length 512 \
    --train_batch_size 4 \
    --num_train_steps 100000
```

`docs.txt` is UTF-8 text, one document per line. The heads
(`MaskedLMLayer`, `NSPLayer`, `BigBirdForPreTraining`) are importable on their
own if you want to plug them into a custom loop.

## Notes & fidelity caveats

- **Attention types.** `original_full` and `block_sparse` are fully exercised by
  the smoke tests at arbitrary lengths. `simulated_sparse` reproduces the
  original exactly, including the original's constraint that the dense mask is
  built over the full `MAX_SEQ_LEN` (4096) grid — so it is only valid when
  `max_encoder_length == 4096` (same limitation as the TF code).
- **GELU** uses the tanh approximation, matching the original.
- **LayerNorm** uses biased variance and `eps=1e-12` (`1e-3` for fp16) to match
  `tf.nn.moments` + `tf.nn.batch_normalization`.
- **Dropout** follows PyTorch semantics (`model.train()` / `model.eval()`); the
  `training=` kwargs are kept for signature parity.
- **Checkpoint conversion.** Parameter *structure* matches the TF model, but TF
  variable names (e.g. `bert/encoder/layer_0/attention/self/query/kernel`)
  differ from PyTorch's dotted `state_dict` keys. A name-mapping script is
  required to load TF checkpoints; the 1:1 layer correspondence makes this
  mechanical.

## Tests

```bash
python test_smoke.py
```
