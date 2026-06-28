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

"""Configuration helpers (PyTorch port).

The original TF code defines its config through absl flags. In this port we
expose the same defaults as a plain dictionary (see `utils.get_default_config`)
and add the extra training-related keys here so that a single dict fully
describes a model + training run.
"""

from core import utils


def get_default_config():
  """Return the default BigBird config extended with training keys."""
  config = utils.get_default_config()
  config.update({
      # data
      "vocab_size": 32000,
      # training
      "optimizer": "AdamWeightDecay",
      "learning_rate": 1e-4,
      "num_train_steps": 100000,
      "num_warmup_steps": 10000,
      "train_batch_size": 4,
      "eval_batch_size": 4,
      # model selection
      "substitute_newline": None,
  })
  return config


def load_sentencepiece_tokenizer(model_path):
  """Load a SentencePiece tokenizer (requires the `sentencepiece` package)."""
  import sentencepiece as spm  # pylint: disable=g-import-not-at-top
  sp = spm.SentencePieceProcessor()
  sp.Load(model_path)
  return sp
