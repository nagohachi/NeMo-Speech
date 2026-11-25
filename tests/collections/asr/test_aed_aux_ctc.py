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

"""The auxiliary CTC loss of the AED model (``model.aux_ctc``) in its padded training step."""

import math

import pytest
import torch
from lhotse import CutSet, SupervisionSegment
from lhotse.testing.dummies import dummy_cut
from omegaconf import open_dict

from nemo.collections.asr.data.audio_to_text_lhotse_prompted import PromptedAudioToTextLhotseDataset
from nemo.collections.asr.models.aed_multitask_models import EncDecMultiTaskModel
from nemo.collections.common.prompts.canary import CanaryPromptFormatter

# the AED model fixture of the padded tests
from tests.collections.asr.test_asr_multitask_model_bpe import asr_model  # noqa: F401


@pytest.fixture(name="cuts")
def make_cuts():
    texts = ["short", "a very long transcript", "mid length text"]
    durations = [1.0, 5.0, 2.3]
    cuts = []
    for idx, (text, duration) in enumerate(zip(texts, durations)):
        cut = dummy_cut(
            idx,
            duration=duration,
            recording_duration=duration,
            with_data=True,
            supervisions=[
                SupervisionSegment(
                    id=f"cut-{idx}", recording_id=f"cut-{idx}", start=0, duration=duration, text=text, language="en"
                )
            ],
        )
        cut.source_lang = "en"
        cut.target_lang = "en"
        cut.task = "asr"
        cut.pnc = "no"
        cuts.append(cut)
    return CutSet(cuts)


@pytest.fixture(name="asr_model_ctc")
def make_asr_model_ctc(asr_model):
    """Build the AED model of the padded tests with an auxiliary CTC loss."""
    cfg = asr_model.cfg.copy()
    with open_dict(cfg):
        cfg.aux_ctc = {"ctc_loss_weight": 0.3}
    model = EncDecMultiTaskModel(cfg=cfg)
    model.configure_optimizers()
    model.metric.eval = lambda **kwargs: {}
    return model


@pytest.mark.unit
def test_ctc_target_lengths_exclude_end_of_text(asr_model_ctc, cuts):
    batch = _make_padded_dataset(asr_model_ctc)[cuts]
    eos = asr_model_ctc.tokenizer.eos_id
    lens = asr_model_ctc._calc_ctc_target_lengths(batch.transcript, batch.transcript_lens)
    # the answers of the prompt format end without the end-of-text token: the targets are the whole answers
    assert torch.equal(lens, batch.transcript_lens)
    # an answer that ends with it loses it
    with_eos = torch.nn.functional.pad(batch.transcript, (0, 1), value=asr_model_ctc.tokenizer.pad_id)
    with_eos[torch.arange(len(lens)), batch.transcript_lens] = eos
    torch.testing.assert_close(asr_model_ctc._calc_ctc_target_lengths(with_eos, batch.transcript_lens + 1), lens)


@pytest.mark.unit
def test_padded_training_step_adds_the_ctc_loss(asr_model_ctc, cuts):
    """Check that the padded step trains on (1 - w) * attention loss + w * CTC loss: the CTC head gets gradients,
    and without the weight the loss is the attention loss alone."""
    asr_model_ctc.eval()
    batch = _make_padded_dataset(asr_model_ctc)[cuts]
    loss = asr_model_ctc.training_step(batch, batch_nb=0)["loss"]
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in asr_model_ctc.ctc_decoder.parameters())
    with torch.no_grad():
        asr_model_ctc.ctc_loss_weight = 0.0
        attention_only = asr_model_ctc.training_step(batch, batch_nb=0)["loss"]
    assert not torch.allclose(loss.detach(), attention_only)


@pytest.mark.unit
def test_change_vocabulary_rebuilds_the_ctc_head(asr_model_ctc):
    old_classes = asr_model_ctc.ctc_decoder.num_classes_with_blank
    tokenizer_cfg = asr_model_ctc.cfg.tokenizer.copy()
    with open_dict(tokenizer_cfg):
        del tokenizer_cfg.langs["de"]

    asr_model_ctc.change_vocabulary(new_tokenizer_dir=tokenizer_cfg, new_tokenizer_type="agg")

    vocab_size = 8 * math.ceil(asr_model_ctc.tokenizer.vocab_size / 8)
    assert asr_model_ctc.ctc_decoder.num_classes_with_blank == vocab_size + 1 != old_classes
    assert asr_model_ctc.ctc_loss.blank == vocab_size
    assert asr_model_ctc.ctc_loss.config_reduction == "mean_volume"


@pytest.mark.unit
def test_aux_ctc_reduction_defaults_to_per_token(asr_model_ctc):
    assert asr_model_ctc.ctc_loss.config_reduction == "mean_volume"


@pytest.mark.unit
def test_aux_ctc_refuses_keys_it_does_not_read(asr_model):
    cfg = asr_model.cfg.copy()
    with open_dict(cfg):
        cfg.aux_ctc = {"ctc_loss_weight": 0.3, "decoder": {"feat_in": 128}}
    with pytest.raises(ValueError, match="decoder"):
        EncDecMultiTaskModel(cfg=cfg)


def _make_padded_dataset(model):
    return PromptedAudioToTextLhotseDataset(tokenizer=model.tokenizer, prompt=CanaryPromptFormatter(model.tokenizer))
