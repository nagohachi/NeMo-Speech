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
def test_validation_loss_is_the_attention_loss_without_ctc(asr_model_ctc, cuts, monkeypatch):
    """Check that val_loss of a model with aux_ctc is the attention loss alone: the training loss of the same batch
    with the CTC weight set to 0, not the mixed training loss."""
    asr_model_ctc.eval()
    batch = _make_padded_dataset(asr_model_ctc)[cuts]
    with torch.no_grad():
        val_loss = asr_model_ctc.validation_pass(batch, batch_idx=0)["val_loss"]
        mixed_loss = asr_model_ctc.training_step(batch, batch_nb=0)["loss"]
        monkeypatch.setattr(asr_model_ctc, "ctc_loss_weight", 0.0)
        attention_loss = asr_model_ctc.training_step(batch, batch_nb=0)["loss"]
    torch.testing.assert_close(val_loss, attention_loss)
    assert not torch.allclose(val_loss, mixed_loss)


@pytest.mark.unit
def test_padded_training_step_mixes_attention_and_ctc_losses(asr_model_ctc, cuts):
    """Check that the padded step trains on 0.7 * attention loss + 0.3 * CTC loss, the CTC loss computed here
    from the encoder output, and that the CTC head gets gradients."""
    asr_model_ctc.eval()
    batch = _make_padded_dataset(asr_model_ctc)[cuts]
    loss = asr_model_ctc.training_step(batch, batch_nb=0)["loss"]
    loss.backward()
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in asr_model_ctc.ctc_decoder.parameters())
    with torch.no_grad():
        input_ids, _ = batch.get_decoder_inputs_outputs()
        _, encoded_len, _, _, encoded = asr_model_ctc._forward(
            input_signal=batch.audio,
            input_signal_length=batch.audio_lens,
            transcript=input_ids,
            transcript_length=batch.prompted_transcript_lens - 1,
        )
        ctc_loss = asr_model_ctc.ctc_loss(
            log_probs=asr_model_ctc.ctc_decoder(encoder_output=encoded),
            targets=batch.transcript,
            input_lengths=encoded_len,
            target_lengths=batch.transcript_lens,  # the answers end without the end-of-text token
        )
        asr_model_ctc.ctc_loss_weight = 0.0
        attention_loss = asr_model_ctc.training_step(batch, batch_nb=0)["loss"]
    assert torch.isfinite(ctc_loss) and ctc_loss > 0
    torch.testing.assert_close(loss.detach(), 0.7 * attention_loss + 0.3 * ctc_loss)


@pytest.mark.unit
def test_ctc_loss_of_a_transcript_too_long_for_its_frames_is_infinite_unless_opted_out(asr_model):
    """Check that zero_infinity is off by default, so such an utterance is not dropped from the CTC term silently,
    and that zero_infinity: true zeroes its loss."""
    targets = torch.tensor([[1, 2, 3]])  # three tokens
    for aux_ctc_config, expected_finite in (
        ({"ctc_loss_weight": 0.3}, False),
        ({"ctc_loss_weight": 0.3, "zero_infinity": True}, True),
    ):
        cfg = asr_model.cfg.copy()
        with open_dict(cfg):
            cfg.aux_ctc = aux_ctc_config
        model = EncDecMultiTaskModel(cfg=cfg)
        log_probs = torch.randn(1, 2, model.ctc_decoder.num_classes_with_blank).log_softmax(-1)  # two frames
        loss = model.ctc_loss(
            log_probs=log_probs, targets=targets, input_lengths=torch.tensor([2]), target_lengths=torch.tensor([3])
        )
        assert bool(torch.isfinite(loss)) == expected_finite


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
def test_aux_ctc_refuses_keys_it_does_not_read(asr_model):
    cfg = asr_model.cfg.copy()
    with open_dict(cfg):
        cfg.aux_ctc = {"ctc_loss_weight": 0.3, "decoder": {"feat_in": 128}}
    with pytest.raises(ValueError, match="decoder"):
        EncDecMultiTaskModel(cfg=cfg)


@pytest.mark.unit
@pytest.mark.parametrize(
    "aux_ctc_config,message",
    [
        ({"ctc_loss_weight": -0.1}, "between 0 and 1"),
        ({"ctc_loss_weight": 1.5}, "between 0 and 1"),
        ({"ctc_loss_weight": None}, "between 0 and 1"),
        ({"ctc_reduction": "mean_batch"}, "apply only with ctc_loss_weight > 0"),
        ({"ctc_loss_weight": 0.0, "zero_infinity": True}, "apply only with ctc_loss_weight > 0"),
        ({"ctc_loss_weight": 0.3, "ctc_reduction": "none"}, "ctc_reduction must be one of"),
        ({"ctc_loss_weight": 0.3, "zero_infinity": "yes"}, "zero_infinity must be a bool"),
    ],
)
def test_aux_ctc_rejects_values_it_cannot_apply(asr_model, aux_ctc_config, message):
    cfg = asr_model.cfg.copy()
    with open_dict(cfg):
        cfg.aux_ctc = aux_ctc_config
    with pytest.raises(ValueError, match=message):
        EncDecMultiTaskModel(cfg=cfg)


def _make_padded_dataset(model):
    return PromptedAudioToTextLhotseDataset(tokenizer=model.tokenizer, prompt=CanaryPromptFormatter(model.tokenizer))
