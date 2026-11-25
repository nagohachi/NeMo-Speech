# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

"""The OOMptimizer batches of the AED model train with the prompt loss mask and with the auxiliary CTC loss."""

import importlib
import sys
from pathlib import Path

import pytest
import torch
from omegaconf import open_dict

from nemo.collections.asr.models.aed_multitask_models import EncDecMultiTaskModel

# the AED model fixture of the padded tests
from tests.collections.asr.test_asr_multitask_model_bpe import asr_model  # noqa: F401

SCRIPTS = Path(__file__).resolve().parents[3] / "scripts" / "speech_recognition"


@pytest.fixture(scope="module", name="oomptimizer")
def load_oomptimizer():
    sys.path.insert(0, str(SCRIPTS))
    try:
        yield importlib.import_module("oomptimizer")
    finally:
        sys.path.remove(str(SCRIPTS))


@pytest.mark.unit
@pytest.mark.parametrize("use_loss_mask_for_prompt", [False, True])
def test_aed_oomptimizer_batch_trains(asr_model, oomptimizer, use_loss_mask_for_prompt):
    with open_dict(asr_model.cfg):
        asr_model.cfg.use_loss_mask_for_prompt = use_loss_mask_for_prompt
    asr_model.metric.eval = lambda **kwargs: {}
    generator = oomptimizer.ProfilingBatchGenerator(asr_model.oomptimizer_schema, start_batch_size=3, device="cpu")

    batch = generator(input_seq_length=16000, output_seq_length=12)

    assert batch.prompt_lens.tolist() == [1, 1, 1]
    loss = asr_model.training_step(batch, batch_nb=0)["loss"]
    assert torch.isfinite(loss)


@pytest.mark.unit
def test_aed_oomptimizer_batch_trains_with_aux_ctc(asr_model, oomptimizer):
    """Check that the schema generates the transcripts as labels, which the auxiliary CTC loss reads."""
    cfg = asr_model.cfg.copy()
    with open_dict(cfg):
        cfg.aux_ctc = {"ctc_loss_weight": 0.3}
    model = EncDecMultiTaskModel(cfg=cfg)
    model.configure_optimizers()
    model.metric.eval = lambda **kwargs: {}
    generator = oomptimizer.ProfilingBatchGenerator(model.oomptimizer_schema, start_batch_size=3, device="cpu")

    batch = generator(input_seq_length=16000, output_seq_length=12)

    assert batch.transcript.shape == (3, 12)
    assert batch.transcript_lens.tolist() == [12, 12, 12]
    # without labels for the transcripts, the CTC targets of the batch are empty and the step raises IndexError
    loss = model.training_step(batch, batch_nb=0)["loss"]
    assert torch.isfinite(loss)
