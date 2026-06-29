# BigBird (PyTorch)

A clean PyTorch implementation of [BigBird](https://arxiv.org/abs/2007.14062),
the sparse-attention transformer that scales to long sequences (up to 4096
tokens). It builds on PyTorch built-ins (`nn.Linear`, `nn.LayerNorm`,
`nn.Embedding`, SDPA, `AdamW`) and keeps the only genuinely custom piece — the
linear-time block-sparse attention kernel.

The code is split into two **self-contained** packages so you only read what
the task needs:

```
mlm/           # encoder-only BigBird for masked language modeling
  config.py            # MLMConfig
  attention.py         # block-sparse kernel + full attention (SDPA)
  modeling.py          # embeddings, encoder, BigBirdModel, BigBirdForMaskedLM
  data.py              # whole-word masking + Dataset
  run_pretraining.py   # training loop (AdamW + linear warmup/decay)
  test_smoke.py

transformer/   # encoder-decoder BigBird for generative (seq2seq) tasks
  config.py            # TransformerConfig
  attention.py         # block-sparse + full attention (self/cross, with cache)
  modeling.py          # embeddings, encoder, decoder, TransformerModel
  beam_search.py
  test_smoke.py
```

The two `attention.py` files share the block-sparse kernel by design — the
duplication keeps each package readable on its own.

## Masked LM (encoder-only)

```python
import torch
from mlm.config import MLMConfig
from mlm.modeling import BigBirdForMaskedLM

config = MLMConfig(vocab_size=32000, max_encoder_length=1024,
                   attention_type="block_sparse")   # or "original_full"
model = BigBirdForMaskedLM(config)

out = model(
    input_ids=torch.randint(1, 32000, (2, 1024)),
    masked_lm_positions=torch.randint(0, 1024, (2, 75)),
    masked_lm_ids=torch.randint(1, 32000, (2, 75)),
    masked_lm_weights=torch.ones(2, 75))
out["loss"].backward()
```

Pre-train from raw text (one document per line, SentencePiece vocab):

```bash
python -m mlm.run_pretraining \
    --input_file docs.txt \
    --vocab_model_file vocab.model \
    --output_dir /tmp/bigb \
    --max_encoder_length 1024 \
    --train_batch_size 4 \
    --num_train_steps 100000
```

## Loading pretrained weights

`mlm/convert_checkpoint.py` warm-starts the encoder-only model from a
HuggingFace checkpoint (no TensorFlow needed). The encoder/FFN/LayerNorm weights
are shape-identical to ours and, since both sides use `nn.Linear`, copy across
with no transpose.

```bash
pip install transformers          # optional, only for this script

# Best starting point: already 4096-ctx and sparse-trained.
python -m mlm.convert_checkpoint --source google/bigbird-roberta-base \
    --output bigbird_init.pt

# Or warm-start from vanilla RoBERTa (512 positions are tiled up to 4096),
# then adapt with sparse attention -- the original BigBird recipe.
python -m mlm.convert_checkpoint --source roberta-base \
    --output roberta_init.pt --max_position_embeddings 4096
```

```python
import torch
from mlm.modeling import BigBirdForMaskedLM
ckpt = torch.load("bigbird_init.pt")
model = BigBirdForMaskedLM(ckpt["config"])
model.load_state_dict(ckpt["state_dict"])
```

Gotchas the script handles: position-embedding offset/tiling (RoBERTa starts at
index 2 and has only 512 rows), the embedding-LayerNorm ↔ encoder-norm
correspondence, and the gelu variant (exact for RoBERTa, tanh for BigBird). The
**tokenizer must match the word embeddings** — use RoBERTa's BPE with
`roberta-base`, and BigBird's SentencePiece with `google/bigbird-roberta-base`.
The source's token-type embeddings and pooler are dropped (this model does
single-segment MLM, so neither is used).

## Seq2seq (encoder-decoder)

```python
import torch
from transformer.config import TransformerConfig
from transformer.modeling import TransformerModel

config = TransformerConfig(vocab_size=32000, max_encoder_length=1024,
                           max_decoder_length=64, beam_size=5)
model = TransformerModel(config)

input_ids = torch.randint(1, 32000, (2, 1024))
target_ids = torch.randint(1, 32000, (2, 64))

out = model(input_ids, target_ids, training=True)   # teacher forcing
out["loss"].backward()

pred_ids = model(input_ids, training=False)["pred_ids"]   # beam search
```

## Notes

- **Attention.** For `max_encoder_length <= 512` the model falls back to full
  attention automatically (it's the standard BERT/RoBERTa setup at that point).
  Block-sparse pads the sequence up to a multiple of `block_size`.
- **GELU** uses the tanh approximation by default (`gelu_exact` for RoBERTa);
  **LayerNorm** uses `eps=1e-12`.
- **Norm style.** `mlm/` is postnorm only (BERT/RoBERTa), since that's what MLM
  and the RoBERTa warm-start use. `transformer/` keeps both postnorm and the
  prenorm (Pegasus) style via `norm_type`.
- **Single segment.** `mlm/` has no token-type embeddings or pooler — they are
  unused for single-segment masked-LM pretraining.

## Tests

```bash
python -m mlm.test_smoke
python -m transformer.test_smoke
```
