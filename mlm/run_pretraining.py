"""Masked-LM pre-training for the encoder-only BigBird model."""

import argparse
import os

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from mlm.config import MLMConfig
from mlm.data import TextMaskedLMDataset
from mlm.modeling import BigBirdForMaskedLM


def build_optimizer(model, lr, num_train_steps, num_warmup_steps,
                    weight_decay=0.01):
  # Decoupled weight decay (AdamW), excluding biases and LayerNorm.
  decay, no_decay = [], []
  for name, p in model.named_parameters():
    if p.ndim == 1 or name.endswith(".bias"):
      no_decay.append(p)
    else:
      decay.append(p)
  optimizer = AdamW([
      {"params": decay, "weight_decay": weight_decay},
      {"params": no_decay, "weight_decay": 0.0},
  ], lr=lr, betas=(0.9, 0.999), eps=1e-6)

  def lr_lambda(step):
    if step < num_warmup_steps:
      return step / max(1, num_warmup_steps)
    return max(0.0, (num_train_steps - step) /
               max(1, num_train_steps - num_warmup_steps))

  return optimizer, LambdaLR(optimizer, lr_lambda)


def train(model, dataset, config, lr, num_train_steps, num_warmup_steps,
          batch_size, output_dir, save_every, device):
  model.to(device).train()
  loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                      drop_last=True)
  optimizer, scheduler = build_optimizer(
      model, lr, num_train_steps, num_warmup_steps)

  step = 0
  while step < num_train_steps:
    for batch in loader:
      batch = {k: v.to(device) for k, v in batch.items()}
      optimizer.zero_grad()
      out = model(**batch)
      out["loss"].backward()
      torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
      optimizer.step()
      scheduler.step()

      if step % 100 == 0:
        print(f"step {step}  loss {out['loss'].item():.4f}"
              f"  lr {scheduler.get_last_lr()[0]:.2e}")
      if save_every and step and step % save_every == 0:
        save(model, output_dir, step)
      step += 1
      if step >= num_train_steps:
        break
  save(model, output_dir, step)


def save(model, output_dir, step):
  os.makedirs(output_dir, exist_ok=True)
  path = os.path.join(output_dir, f"checkpoint-{step}.pt")
  torch.save(model.state_dict(), path)
  print(f"saved {path}")


def parse_args():
  p = argparse.ArgumentParser(description="BigBird MLM pre-training")
  p.add_argument("--input_file", required=True, help="UTF-8 text, one doc/line")
  p.add_argument("--vocab_model_file", required=True, help="SentencePiece .model")
  p.add_argument("--output_dir", default="/tmp/bigb")
  p.add_argument("--init_checkpoint", default=None)
  p.add_argument("--max_encoder_length", type=int, default=512)
  p.add_argument("--attention_type", default="block_sparse")
  p.add_argument("--learning_rate", type=float, default=1e-4)
  p.add_argument("--num_train_steps", type=int, default=100000)
  p.add_argument("--num_warmup_steps", type=int, default=10000)
  p.add_argument("--train_batch_size", type=int, default=4)
  p.add_argument("--save_checkpoints_steps", type=int, default=1000)
  return p.parse_args()


def main():
  import sentencepiece as spm

  args = parse_args()
  sp = spm.SentencePieceProcessor()
  sp.Load(args.vocab_model_file)

  config = MLMConfig(
      vocab_size=sp.GetPieceSize(),
      max_encoder_length=args.max_encoder_length,
      attention_type=args.attention_type)
  if args.max_encoder_length > config.max_position_embeddings:
    raise ValueError("max_encoder_length exceeds max_position_embeddings")

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model = BigBirdForMaskedLM(config)
  if args.init_checkpoint:
    model.load_state_dict(torch.load(args.init_checkpoint, map_location="cpu"))

  with open(args.input_file, encoding="utf-8") as f:
    documents = [line.strip() for line in f if line.strip()]
  dataset = TextMaskedLMDataset(documents, sp, config)

  train(model, dataset, config, args.learning_rate, args.num_train_steps,
        args.num_warmup_steps, args.train_batch_size, args.output_dir,
        args.save_checkpoints_steps, device)


if __name__ == "__main__":
  main()
