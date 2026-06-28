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

"""Functions and classes related to optimization (weight updates), PyTorch."""

import math
import re

import torch
from torch.optim import Optimizer


def get_linear_warmup_linear_decay_lr(init_lr, num_train_steps,
                                      num_warmup_steps):
  """Returns a (step -> lr) schedule with linear warmup and linear decay."""

  def lr_at(step):
    learning_rate = init_lr
    # Linear decay.
    decayed = max(0.0, 1.0 - step / float(num_train_steps))
    learning_rate = init_lr * decayed
    # Linear warmup.
    if num_warmup_steps:
      if step < num_warmup_steps:
        learning_rate = init_lr * (step / float(num_warmup_steps))
    return learning_rate

  return lr_at


def get_linear_warmup_rsqrt_decay_lr(init_lr, hidden_size, num_warmup_steps):
  """Returns a (step -> lr) schedule with linear warmup and rsqrt decay."""
  num_warmup_steps = float(num_warmup_steps)

  def lr_at(step):
    step = float(step)
    learning_rate = init_lr / math.sqrt(hidden_size)
    learning_rate *= min(1.0, step / num_warmup_steps)
    learning_rate /= math.sqrt(max(step, num_warmup_steps))
    return learning_rate

  return lr_at


class AdamWeightDecayOptimizer(Optimizer):
  """A basic Adam optimizer that includes "correct" L2 weight decay.

  Matches the BERT/BigBird optimizer: no bias correction, decoupled weight
  decay applied to the update before the learning-rate scaling.
  """

  def __init__(self,
               params,
               lr,
               weight_decay_rate=0.0,
               beta_1=0.9,
               beta_2=0.999,
               epsilon=1e-6,
               exclude_from_weight_decay=("LayerNorm", "layer_norm", "bias")):
    defaults = dict(
        lr=lr, weight_decay_rate=weight_decay_rate, beta_1=beta_1,
        beta_2=beta_2, epsilon=epsilon,
        exclude_from_weight_decay=exclude_from_weight_decay)
    super().__init__(params, defaults)

  def _do_use_weight_decay(self, name, exclude, weight_decay_rate):
    if not weight_decay_rate:
      return False
    if name is not None and exclude:
      for r in exclude:
        if re.search(r, name) is not None:
          return False
    return True

  @torch.no_grad()
  def step(self, closure=None):
    loss = None
    if closure is not None:
      with torch.enable_grad():
        loss = closure()

    for group in self.param_groups:
      beta_1 = group["beta_1"]
      beta_2 = group["beta_2"]
      epsilon = group["epsilon"]
      lr = group["lr"]
      weight_decay_rate = group["weight_decay_rate"]
      exclude = group["exclude_from_weight_decay"]
      names = group.get("names", [None] * len(group["params"]))

      for p, name in zip(group["params"], names):
        if p.grad is None:
          continue
        grad = p.grad
        state = self.state[p]
        if not state:
          state["m"] = torch.zeros_like(p)
          state["v"] = torch.zeros_like(p)
        m, v = state["m"], state["v"]

        next_m = beta_1 * m + (1.0 - beta_1) * grad
        next_v = beta_2 * v + (1.0 - beta_2) * grad * grad

        update = next_m / (next_v.sqrt() + epsilon)

        if self._do_use_weight_decay(name, exclude, weight_decay_rate):
          update = update + weight_decay_rate * p

        p.add_(update, alpha=-lr)
        state["m"] = next_m
        state["v"] = next_v

    return loss


def get_optimizer(params, model, learning_rate):
  """Build an optimizer for `model` given the config `params`.

  Args:
    params: config dictionary with an "optimizer" key.
    model: an nn.Module whose parameters are optimized.
    learning_rate: float learning rate.

  Returns:
    A torch.optim.Optimizer.
  """
  optimizer_name = params["optimizer"]

  if optimizer_name == "Adam":
    return torch.optim.Adam(
        model.parameters(), lr=learning_rate,
        betas=(params["optimizer_beta1"], params["optimizer_beta2"]),
        eps=params["optimizer_epsilon"])

  if optimizer_name in ("AdamWeightDecay", "Adafactor"):
    # Carry the parameter names so weight decay can be excluded by name.
    named = list(model.named_parameters())
    param_group = {
        "params": [p for _, p in named],
        "names": [n for n, _ in named],
    }
    return AdamWeightDecayOptimizer(
        [param_group],
        lr=learning_rate,
        weight_decay_rate=params["weight_decay_rate"],
        beta_1=params["optimizer_beta1"],
        beta_2=params["optimizer_beta2"],
        epsilon=params["optimizer_epsilon"],
        exclude_from_weight_decay=("LayerNorm", "layer_norm", "bias"))

  if optimizer_name == "SGD":
    return torch.optim.SGD(model.parameters(), lr=learning_rate)

  raise ValueError("Unknown optimizer: {}.".format(optimizer_name))
