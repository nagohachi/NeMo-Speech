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

"""Sequence-packed teacher forcing of the AED model against its padded training step."""

import traceback
from collections import namedtuple

import pytest
import torch
from lhotse import AudioSource, CutSet, Recording
from lhotse.testing.random import deterministic_rng  # noqa: F401
from lhotse.utils import fastcopy
from omegaconf import OmegaConf, open_dict
from torch.utils._python_dispatch import TorchDispatchMode, _disable_current_modes
from torch.utils._pytree import tree_flatten
from torch.utils.data._utils.pin_memory import pin_memory
from torch.utils.weak import WeakIdKeyDictionary

from nemo.collections.asr.data.audio_to_text_lhotse_prompted import (
    PackedPromptedAudioToTextMiniBatch,
    PromptedAudioToTextLhotseDataset,
)
from nemo.collections.asr.models.aed_multitask_models import EncDecMultiTaskModel
from nemo.collections.asr.modules.conv_asr import ConvASRDecoder
from nemo.collections.asr.parts.packed_sequence import pack_encoder_output
from nemo.collections.common.data import apply_prompt_format_fn
from nemo.collections.common.data.lhotse import dataloader as lhotse_dataloader
from nemo.collections.common.data.lhotse.audio_loading import INT16_AUDIO_SCALE, read_unpadded_audio_rows
from nemo.collections.common.parts.adapter_modules import LinearAdapterConfig
from nemo.collections.common.prompts.canary import CanaryPromptFormatter

# the AED model fixture of the padded tests
from tests.collections.asr.test_aed_aux_ctc import make_asr_model_ctc, make_cuts  # noqa: F401
from tests.collections.asr.test_asr_multitask_model_bpe import asr_model  # noqa: F401


@pytest.mark.unit
def test_packed_batch_matches_padded_batch(asr_model, cuts):
    padded_ds, packed_ds = _make_datasets(asr_model)
    padded = padded_ds[cuts]
    packed = packed_ds[cuts]

    assert isinstance(packed, PackedPromptedAudioToTextMiniBatch)
    assert torch.equal(packed.audio_lens, padded.audio_lens)
    assert packed.audio_cu_seqlens.tolist() == [0] + padded.audio_lens.cumsum(0).tolist()
    expected_audio = torch.cat([padded.audio[i, :n] for i, n in enumerate(padded.audio_lens.tolist())])
    assert torch.equal(packed.audio, expected_audio)

    assert torch.equal(packed.prompted_transcript_lens, padded.prompted_transcript_lens)
    assert torch.equal(packed.prompt_lens, padded.prompt_lens)
    inputs, targets = padded.get_decoder_inputs_outputs()
    lens = (padded.prompted_transcript_lens - 1).tolist()
    assert torch.equal(packed.decoder_input_ids, torch.cat([inputs[i, :n] for i, n in enumerate(lens)]))
    assert torch.equal(packed.decoder_targets, torch.cat([targets[i, :n] for i, n in enumerate(lens)]))
    assert torch.equal(packed.transcript_lens, padded.transcript_lens)
    assert torch.equal(packed.transcript, padded.transcript)
    assert packed.max_decoder_len == int((padded.prompted_transcript_lens - 1).max())
    assert packed.host_audio_lens == tuple(padded.audio_lens.tolist())


@pytest.mark.unit
def test_int16_audio_packed_batch_scales_to_the_float_batch(asr_model, cuts):
    """Check that with int16_audio the packed batch carries the 16-bit samples, which divided by INT16_AUDIO_SCALE
    are the float32 samples."""
    _, packed_ds = _make_datasets(asr_model)
    float_batch, int16_batch = packed_ds[cuts], _make_int16_dataset(asr_model)[cuts]

    assert int16_batch.audio.dtype == torch.int16
    assert torch.equal(int16_batch.audio.to(torch.float32) / INT16_AUDIO_SCALE, float_batch.audio)
    for name, value in vars(float_batch).items():
        if torch.is_tensor(value) and name != "audio":
            assert torch.equal(getattr(int16_batch, name), value), name
    assert int16_batch.host_audio_lens == float_batch.host_audio_lens


@pytest.mark.unit
def test_int16_audio_gives_the_float_features(asr_model, cuts):
    """Check that the packed preprocessor scales int16 audio itself: the features of the float audio, bit for bit."""
    asr_model.eval()
    _, packed_ds = _make_datasets(asr_model)
    float_features, int16_features = (
        _calc_packed_features(asr_model, batch) for batch in (packed_ds[cuts], _make_int16_dataset(asr_model)[cuts])
    )
    assert torch.equal(int16_features.data, float_features.data)
    assert torch.equal(int16_features.lengths, float_features.lengths)


@pytest.mark.unit
@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_int16_audio_gives_the_float_features_on_cuda(asr_model, cuts):
    """Check that the int16 audio is scaled on the GPU: the CUDA features of the float audio, bit for bit."""
    asr_model = asr_model.cuda().eval()
    _, packed_ds = _make_datasets(asr_model)
    float_features, int16_features = (
        _calc_packed_features(asr_model, _move_to_cuda(batch))
        for batch in (packed_ds[cuts], _make_int16_dataset(asr_model)[cuts])
    )
    assert torch.equal(int16_features.data, float_features.data)


@pytest.mark.unit
def test_int16_audio_needs_pack_sequences(asr_model):
    """Check that int16_audio without pack_sequences raises: only the packed batches carry int16 audio."""
    with pytest.raises(ValueError, match="int16_audio needs pack_sequences"):
        PromptedAudioToTextLhotseDataset(
            tokenizer=asr_model.tokenizer, prompt=CanaryPromptFormatter(asr_model.tokenizer), int16_audio=True
        )


@pytest.mark.unit
def test_int16_audio_refuses_cuts_it_cannot_read_as_lhotse_does(asr_model, cuts):
    """Check that the reader raises for a cut whose float samples come out of a transform (here resampling), which
    has no exact int16 form, even when it drops cuts with unreadable audio."""
    dataset = _make_int16_dataset(asr_model)
    for fault_tolerant in (False, True):
        with pytest.raises(ValueError, match="audio transforms"):
            dataset.with_fault_tolerant_audio_loading(fault_tolerant)[cuts.resample(8000)]
    with pytest.raises(ValueError, match="float32 or int16"):
        read_unpadded_audio_rows(cuts, _make_int16_dataset(asr_model).load_audio, dtype=torch.float16)


@pytest.mark.unit
def test_int16_audio_drops_unreadable_cuts_only_when_fault_tolerant(asr_model, cuts):
    """Check that a cut whose audio fails to decode raises, or is dropped with fault-tolerant audio loading."""
    cuts = cuts.to_eager()
    cuts[1].recording.sources[0].source = b"not audio"
    dataset = _make_int16_dataset(asr_model)
    with pytest.raises(Exception, match="Format not recognised"):
        dataset.with_fault_tolerant_audio_loading(False)[cuts]
    batch = dataset.with_fault_tolerant_audio_loading(True)[cuts]
    assert [cut.id for cut in batch.cuts] == [cuts[0].id, cuts[2].id]
    assert batch.audio.dtype == torch.int16 and len(batch.host_audio_lens) == 2


@pytest.mark.unit
def test_pack_sequences_refuses_chunking(asr_model):
    with pytest.raises(ValueError, match="enable_chunking"):
        PromptedAudioToTextLhotseDataset(
            tokenizer=asr_model.tokenizer,
            prompt=CanaryPromptFormatter(asr_model.tokenizer),
            enable_chunking=True,
            pack_sequences=True,
        )


@pytest.mark.unit
@pytest.mark.parametrize("use_loss_mask_for_prompt", [False, True])
def test_packed_training_loss_matches_padded(deterministic_rng, asr_model, cuts, use_loss_mask_for_prompt):
    """Check that in eval mode (no dropout, no SpecAugment, running BatchNorm statistics) both forms give one loss."""
    with open_dict(asr_model.cfg):
        asr_model.cfg.use_loss_mask_for_prompt = use_loss_mask_for_prompt
    asr_model.eval()
    padded_ds, packed_ds = _make_datasets(asr_model)
    padded = padded_ds[cuts]
    packed = packed_ds[cuts]
    # the training batch metric decodes greedily; it is not part of the comparison
    asr_model.metric.eval = lambda **kwargs: {}

    with torch.no_grad():
        padded_loss = asr_model.training_step(padded, batch_nb=0)["loss"]
        packed_out = asr_model.training_step(packed, batch_nb=0)

    torch.testing.assert_close(packed_out["loss"], padded_loss, rtol=1e-5, atol=1e-5)


@pytest.mark.unit
@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_packed_training_loss_matches_padded_cuda_bf16(deterministic_rng, asr_model, cuts):
    """Check the loss on CUDA, where the packed form runs variable-length FlashAttention and the Triton
    relative-position kernel."""
    asr_model = asr_model.cuda().eval()
    padded_ds, packed_ds = _make_datasets(asr_model)
    padded = _move_to_cuda(padded_ds[cuts])
    packed = _move_to_cuda(packed_ds[cuts])
    asr_model.metric.eval = lambda **kwargs: {}

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        padded_loss = asr_model.training_step(padded, batch_nb=0)["loss"]
        packed_loss = asr_model.training_step(packed, batch_nb=0)["loss"]

    torch.testing.assert_close(packed_loss.float(), padded_loss.float(), rtol=2e-2, atol=2e-2)


@pytest.mark.unit
def test_packed_training_step_backward(deterministic_rng, asr_model, cuts):
    asr_model.train()
    _, packed_ds = _make_datasets(asr_model)
    loss = asr_model.training_step(packed_ds[cuts], batch_nb=0)["loss"]
    assert torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in asr_model.encoder.parameters() if p.requires_grad]
    assert any(g is not None and g.abs().sum() > 0 for g in grads)
    assert asr_model.transf_decoder.embedding.token_embedding.weight.grad is not None


@pytest.mark.unit
def test_packed_batch_cuts_drop_token_ids(asr_model, cuts):
    """Check that the packed batch drops the token tensors of pretokenized cuts (it collates them),
    the sampler keeps them, and the padded batch leaves its cuts as they were."""
    prompt = CanaryPromptFormatter(asr_model.tokenizer)
    for cut in cuts:
        formatted = apply_prompt_format_fn(cut, prompt)
        cut.input_ids = formatted["input_ids"]
        cut.context_ids = formatted["context_ids"]
        cut.answer_ids = formatted["answer_ids"]
    packed_batch = PromptedAudioToTextLhotseDataset(tokenizer=asr_model.tokenizer, prompt=prompt, pack_sequences=True)[
        cuts
    ]
    for cut in packed_batch.cuts:
        assert not any(hasattr(cut, key) for key in ("input_ids", "context_ids", "answer_ids"))
        assert cut.target_lang == "en"
    assert all(hasattr(cut, "input_ids") for cut in cuts)
    padded_batch = PromptedAudioToTextLhotseDataset(tokenizer=asr_model.tokenizer, prompt=prompt)[cuts]
    assert all(hasattr(cut, "input_ids") for cut in padded_batch.cuts)


@pytest.mark.unit
def test_packed_batch_drops_an_unreadable_cut_with_its_tokens(tmp_path, asr_model, cuts):
    """Check that with fault-tolerant audio loading a cut whose audio is missing leaves the packed batch together
    with its tokens, and the other cuts keep their audio and tokens in order."""
    _, packed_ds = _make_datasets(asr_model)
    first, second, third = cuts
    unreadable = _make_unreadable_cut(second, tmp_path, text="words that only the unreadable cut has")
    batch = packed_ds.with_fault_tolerant_audio_loading(True)[CutSet([first, unreadable, third])]
    _assert_same_batch(batch, packed_ds[CutSet([first, third])])
    assert [cut.id for cut in batch.cuts] == [first.id, third.id]


@pytest.mark.unit
def test_packed_batch_raises_when_no_audio_loads(tmp_path, asr_model, cuts):
    """Check that a batch whose cuts all fail to load raises instead of an empty packed batch."""
    _, packed_ds = _make_datasets(asr_model)
    unreadable = CutSet([_make_unreadable_cut(cut, tmp_path, text=cut.supervisions[0].text) for cut in cuts])
    with pytest.raises(ValueError, match="none of the 3 cuts"):
        packed_ds.with_fault_tolerant_audio_loading(True)[unreadable]


@pytest.mark.unit
def test_aux_ctc_packed_matches_padded(deterministic_rng, asr_model_ctc, cuts):
    asr_model_ctc.eval()
    padded_ds, packed_ds = _make_datasets(asr_model_ctc)
    with torch.no_grad():
        padded_loss = asr_model_ctc.training_step(padded_ds[cuts], batch_nb=0)["loss"]
        packed_loss = asr_model_ctc.training_step(packed_ds[cuts], batch_nb=0)["loss"]
        asr_model_ctc.ctc_loss_weight = 0.0
        attention_only = asr_model_ctc.training_step(padded_ds[cuts], batch_nb=0)["loss"]
    torch.testing.assert_close(packed_loss, padded_loss, rtol=1e-5, atol=1e-5)
    assert not torch.allclose(padded_loss, attention_only)


@pytest.mark.unit
@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_aux_ctc_packed_matches_padded_cuda_bf16(deterministic_rng, asr_model_ctc, cuts):
    """Check the packed CTC loss on CUDA, where it runs the Triton kernel over the token-flat log probabilities."""
    asr_model_ctc = asr_model_ctc.cuda().eval()
    padded_ds, packed_ds = _make_datasets(asr_model_ctc)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        padded_loss = asr_model_ctc.training_step(_move_to_cuda(padded_ds[cuts]), batch_nb=0)["loss"]
        packed_loss = asr_model_ctc.training_step(_move_to_cuda(packed_ds[cuts]), batch_nb=0)["loss"]
    torch.testing.assert_close(packed_loss.float(), padded_loss.float(), rtol=2e-2, atol=2e-2)


@pytest.mark.unit
@pytest.mark.parametrize("temperature", [1.0, 2.0])
def test_conv_asr_decoder_forward_packed_matches_forward(deterministic_rng, temperature):
    """Check that the packed log probabilities are those of forward at the valid frames, also with a temperature."""
    decoder = ConvASRDecoder(feat_in=8, num_classes=5)
    decoder.temperature = temperature
    lengths = torch.tensor([4, 2, 3])
    padded = torch.randn(3, 8, 4)
    expected = decoder(encoder_output=padded)  # (B, T, C)
    packed = decoder.forward_packed(pack_encoder_output(padded.transpose(1, 2), lengths))
    torch.testing.assert_close(packed.data, torch.cat([expected[i, :n] for i, n in enumerate(lengths.tolist())]))


@pytest.mark.unit
def test_conv_asr_decoder_forward_packed_refuses_adapters():
    """Check that a decoder with an enabled adapter raises instead of skipping the adapter in the packed path."""
    decoder = ConvASRDecoder(feat_in=8, num_classes=5)
    decoder.add_adapter("adapter", LinearAdapterConfig(in_features=8, dim=4))
    packed = pack_encoder_output(torch.randn(2, 3, 8), torch.tensor([3, 2]))
    with pytest.raises(NotImplementedError, match="Adapters"):
        decoder.forward_packed(packed)


@pytest.mark.unit
@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="pinning needs CUDA")
def test_packed_batch_pin_memory(asr_model, cuts):
    """Check that DataLoader(pin_memory=True) pins the tensors of a packed batch through its pin_memory method."""
    _, packed_ds = _make_datasets(asr_model)
    batch = packed_ds[cuts]
    pinned = pin_memory(batch)
    assert type(pinned) is type(batch)
    for name, value in vars(pinned).items():
        if torch.is_tensor(value):
            assert value.is_pinned(), name
            assert torch.equal(value, getattr(batch, name))
    assert pinned.cuts is batch.cuts


@pytest.fixture(name="asr_model_packed_recipe")
def make_asr_model_packed_recipe(asr_model):
    """Build the AED model of the padded tests with the components of the sequence-packed training recipe:
    8x ``dw_striding`` subsampling, SpecAugment, an auxiliary CTC loss and the loss mask of the prompt."""
    cfg = asr_model.cfg.copy()
    with open_dict(cfg):
        cfg.encoder.subsampling_factor = 8
        cfg.spec_augment = {
            "_target_": "nemo.collections.asr.modules.SpectrogramAugmentation",
            "freq_masks": 2,
            "time_masks": 10,
            "freq_width": 27,
            "time_width": 0.05,
        }
        cfg.aux_ctc = {"ctc_loss_weight": 0.3, "ctc_reduction": "mean_volume"}
        cfg.use_loss_mask_for_prompt = True
    model = EncDecMultiTaskModel(cfg=cfg)
    model.configure_optimizers()
    model.metric.eval = lambda **kwargs: {}
    return model


@pytest.mark.unit
def test_packed_training_step_reads_no_device_data(deterministic_rng, asr_model_packed_recipe, cuts):
    """Check that one packed training step (forward, both losses, backward) sizes its work from host metadata.

    On the CPU no read synchronizes anything, so ``_DeviceReadCounter`` records the reads that would:
    the host reads of data that lives on the GPU in training (see there).
    The one read left is the CPU's own: the dense reference attention, which runs instead of the
    FlashAttention and Triton kernels of CUDA, reads the decoder's sequence boundaries to slice its sequences.
    The CUDA counterpart below checks the kernels' path for every synchronization,
    including the host-to-device copies that wait for the stream, which this proxy does not see.
    """
    model = asr_model_packed_recipe.train()
    _, packed_ds = _make_datasets(model)
    batch = packed_ds[cuts]
    with _DeviceReadCounter(model, batch) as counter:
        model.training_step(batch, batch_nb=0)["loss"].backward()
    assert [read.function for read in counter.reads] == ["sequence_slices"], counter.reads


@pytest.mark.unit
def test_packed_frontend_without_host_lengths_reads_once(deterministic_rng, asr_model_packed_recipe, cuts):
    """Check that a caller without host lengths pays one device read in the frontend, which validates the metadata."""
    model = asr_model_packed_recipe.train()
    _, packed_ds = _make_datasets(model)
    batch = packed_ds[cuts]
    with _DeviceReadCounter(model, batch) as counter:
        features = model.preprocessor.forward_packed(batch.audio, batch.audio_lens, batch.audio_cu_seqlens)
        model.encoder.forward_sequence_packed(features)
    assert [read.function for read in counter.reads] == ["_validate_packed_waveforms"], counter.reads
    assert torch.equal(features.host_lengths, features.lengths)


@pytest.mark.unit
@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_packed_training_step_cuda_synchronizes_nothing(deterministic_rng, asr_model_packed_recipe, cuts):
    """Check that on CUDA a packed training step (forward, both losses and backward) never waits for the GPU."""
    model = asr_model_packed_recipe.cuda().train()
    _, packed_ds = _make_datasets(model)
    batch = _move_to_cuda(packed_ds[cuts])

    def step():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model.training_step(batch, batch_nb=0)["loss"]
        loss.backward()

    step()  # compiles the Triton kernels and initializes the CUDA libraries
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        step()
    finally:
        torch.cuda.set_sync_debug_mode("default")


@pytest.mark.unit
def test_packed_training_step_with_host_lengths_matches_without(deterministic_rng, asr_model_packed_recipe, cuts):
    """Check that the host metadata changes no value: without it (one validating read) the step gives the same loss."""
    model = asr_model_packed_recipe.eval()
    _, packed_ds = _make_datasets(model)
    with torch.no_grad():
        with_host = model.training_step(packed_ds[cuts], batch_nb=0)["loss"]
        batch = packed_ds[cuts]
        batch.host_audio_lens = None
        without_host = model.training_step(batch, batch_nb=0)["loss"]
    torch.testing.assert_close(with_host, without_host, rtol=0, atol=0)


@pytest.mark.unit
def test_packed_frontend_features_match_padded(asr_model, cuts):
    """Check that both frontends agree within 1e-4; they run in float32 and sum the normalization statistics in
    another order.

    In float64 the two agree to about 1e-13; the float32 difference is rounding of the per-feature mean and standard
    deviation, which shifts every frame of an utterance a little.
    """
    asr_model.eval()
    padded_ds, packed_ds = _make_datasets(asr_model)
    padded, packed = padded_ds[cuts], packed_ds[cuts]
    with torch.no_grad():
        features, lengths = asr_model.preprocessor(input_signal=padded.audio, length=padded.audio_lens)
        packed_features = asr_model.preprocessor.forward_packed(
            packed.audio, packed.audio_lens, packed.audio_cu_seqlens, host_length=packed.host_audio_lens
        )

    assert packed_features.lengths.tolist() == lengths.tolist()
    expected = torch.cat([features[i, :, :n].T for i, n in enumerate(lengths.tolist())])
    torch.testing.assert_close(packed_features.data, expected, rtol=0, atol=1e-4)


@pytest.mark.unit
def test_packed_backward_matches_padded_from_the_same_features(deterministic_rng, asr_model, cuts, monkeypatch):
    """Check that from identical float64 features the packed and the padded training step give every parameter its
    gradient.

    The frontends are replaced by the same precomputed features, so that their float32 rounding
    (test_packed_frontend_features_match_padded) does not hide a difference in the encoder, the decoder or the loss.
    """
    asr_model.eval().double()
    asr_model.metric.eval = lambda **kwargs: {}
    padded_ds, packed_ds = _make_datasets(asr_model)
    padded, packed = padded_ds[cuts], packed_ds[cuts]
    with torch.no_grad():
        features, lengths = asr_model.preprocessor(input_signal=padded.audio, length=padded.audio_lens)
        packed_features = asr_model.preprocessor.forward_packed(
            packed.audio, packed.audio_lens, packed.audio_cu_seqlens, host_length=packed.host_audio_lens
        )
    features = features.double()
    packed_features = packed_features.with_data(
        torch.cat([features[i, :, :n].T for i, n in enumerate(lengths.tolist())])
    )
    monkeypatch.setattr(asr_model.preprocessor, "forward", lambda input_signal, length: (features, lengths))
    monkeypatch.setattr(asr_model.preprocessor, "forward_packed", lambda *args, **kwargs: packed_features)

    grads = []
    for batch in (padded, packed):
        asr_model.zero_grad(set_to_none=True)
        asr_model.training_step(batch, batch_nb=0)["loss"].backward()
        grads.append({name: p.grad.clone() for name, p in asr_model.named_parameters() if p.grad is not None})

    padded_grads, packed_grads = grads
    assert packed_grads.keys() == padded_grads.keys()
    assert any(name.startswith("encoder.pre_encode.") for name in packed_grads)
    assert any(name.startswith("transf_decoder.") for name in packed_grads)
    for name, grad in packed_grads.items():
        torch.testing.assert_close(grad, padded_grads[name], rtol=1e-10, atol=1e-14, msg=name)


@pytest.mark.unit
@pytest.mark.parametrize("name", ["validation_ds", "test_ds"])
def test_evaluation_data_refuses_sequence_packing(asr_model, name):
    config = OmegaConf.create({"use_lhotse": True, "pack_sequences": True})
    setup = asr_model.setup_validation_data if name == "validation_ds" else asr_model.setup_test_data
    with pytest.raises(ValueError, match=f"{name}.pack_sequences"):
        setup(config)


@pytest.mark.unit
@pytest.mark.parametrize(
    "setting,match",
    [
        ({"loss": {"predict_last_k": 2}}, "predict_last_k"),
        ({"transf_encoder": {"num_layers": 1}}, "transf_encoder"),
    ],
)
def test_packed_training_refuses_unsupported_settings(asr_model, setting, match):
    with open_dict(asr_model.cfg):
        for key, value in setting.items():
            if isinstance(value, dict) and key in asr_model.cfg:
                asr_model.cfg[key].update(value)
            else:
                asr_model.cfg[key] = value
    config = OmegaConf.create({"use_lhotse": True, "pack_sequences": True})
    with pytest.raises(ValueError, match=match):
        asr_model.setup_training_data(config)


@pytest.mark.unit
def test_packed_training_refuses_an_encoder_without_a_packed_forward(asr_model, monkeypatch):
    """Check that setup refuses an encoder without forward_sequence_packed instead of failing at the first step."""
    monkeypatch.setattr(asr_model, "encoder", torch.nn.Linear(2, 2))
    with pytest.raises(ValueError, match="forward_sequence_packed"):
        asr_model.setup_training_data(OmegaConf.create({"use_lhotse": True, "pack_sequences": True}))


@pytest.mark.unit
def test_packed_training_refuses_adapters_on_the_ctc_head(asr_model_ctc):
    """Check that setup refuses a CTC head with adapters, which the packed CTC forward does not apply."""
    asr_model_ctc.ctc_decoder.add_adapter(
        "adapter", LinearAdapterConfig(in_features=asr_model_ctc.ctc_decoder._feat_in, dim=4)
    )
    with pytest.raises(ValueError, match="adapters"):
        asr_model_ctc.setup_training_data(OmegaConf.create({"use_lhotse": True, "pack_sequences": True}))


@pytest.mark.unit
def test_packed_training_loader_through_model_setup(tmp_path, monkeypatch, asr_model, cuts):
    """Check that the model setup's training loader (a worker process, collation of 3 cuts) yields the packed batch."""
    path = _save_cuts(cuts, tmp_path)
    warnings = []
    monkeypatch.setattr(lhotse_dataloader.logging, "warning", lambda msg, *args, **kwargs: warnings.append(str(msg)))
    asr_model.setup_training_data(_make_loader_config(path, pin_memory=False))
    batch = next(iter(asr_model._train_dl))

    expected = PromptedAudioToTextLhotseDataset(
        tokenizer=asr_model.tokenizer, prompt=asr_model.prompt, pack_sequences=True
    )[CutSet.from_file(path)]
    assert isinstance(batch, PackedPromptedAudioToTextMiniBatch)
    _assert_same_batch(batch, expected)
    # pack_sequences is a dataset option of the model, not an unknown key of the Lhotse dataloader config
    assert not any("pack_sequences" in message for message in warnings)


@pytest.mark.unit
def test_int16_audio_through_model_setup(tmp_path, asr_model, cuts):
    """Check that train_ds.int16_audio reaches the dataset of the model setup, which reads the audio files as int16."""
    path = _save_cuts(cuts, tmp_path)
    with open_dict(asr_model.cfg):
        asr_model.cfg.train_batch_metrics = False
    config = _make_loader_config(path, pin_memory=False)
    config.int16_audio = True

    asr_model.setup_training_data(config)
    batch = next(iter(asr_model._train_dl))

    expected = PromptedAudioToTextLhotseDataset(
        tokenizer=asr_model.tokenizer, prompt=asr_model.prompt, pack_sequences=True
    )[CutSet.from_file(path)]
    assert batch.audio.dtype == torch.int16
    assert torch.equal(batch.audio.to(torch.float32) / INT16_AUDIO_SCALE, expected.audio)


@pytest.mark.unit
@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="pinning needs CUDA")
def test_packed_training_loader_pins_memory(tmp_path, asr_model, cuts):
    path = _save_cuts(cuts, tmp_path)
    asr_model.setup_training_data(_make_loader_config(path, pin_memory=True))
    batch = next(iter(asr_model._train_dl))

    assert isinstance(batch, PackedPromptedAudioToTextMiniBatch)
    for name, value in vars(batch).items():
        if torch.is_tensor(value):
            assert value.is_pinned(), name


class _DeviceReadCounter(TorchDispatchMode):
    """Records the reads of a CPU run that would synchronize the host with the GPU in training.

    A tensor derived from a batch tensor or from a parameter or buffer of the model lives on the GPU in training,
    a tensor the code builds from host values does not; the counter follows the derivation through every op
    and records the host reads of derived tensors: scalar reads (``item``, ``int``, ``bool``),
    copies to the host (``tolist``, ``cpu``, ``numpy``, ``to("cpu")``), ``nonzero`` and boolean-mask indexing,
    ``equal``, ``masked_select``, ``unique`` and ``repeat_interleave`` without ``output_size``.
    Each read is recorded as the innermost NeMo frame that issued it.
    """

    _READ_OPS = {"_local_scalar_dense", "nonzero", "equal", "masked_select", "_unique2", "unique_dim"}
    _HOST_COPY_METHODS = ("tolist", "numpy", "cpu", "to")

    def __init__(self, model, batch):
        super().__init__()
        self.reads = []
        self._device_data = WeakIdKeyDictionary()
        for tensor in [
            *model.parameters(),
            *model.buffers(),
            *(v for v in vars(batch).values() if torch.is_tensor(v)),
        ]:
            self._device_data[tensor] = True
        self._originals = {}

    def __enter__(self):
        for name in self._HOST_COPY_METHODS:
            original = getattr(torch.Tensor, name)
            self._originals[name] = original
            setattr(torch.Tensor, name, self._wrap_host_copy(name, original))
        return super().__enter__()

    def __exit__(self, *exc):
        for name, original in self._originals.items():
            setattr(torch.Tensor, name, original)
        return super().__exit__(*exc)

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        inputs = [t for t in tree_flatten((args, kwargs))[0] if isinstance(t, torch.Tensor)]
        derived = any(t in self._device_data for t in inputs)
        if derived and self._is_read(func, args, kwargs):
            self._record(f"aten.{func._overloadpacket.__name__}")
        out = func(*args, **kwargs)
        if derived:
            for tensor in tree_flatten(out)[0]:
                if isinstance(tensor, torch.Tensor):
                    self._device_data[tensor] = True
        return out

    def _is_read(self, func, args, kwargs) -> bool:
        name = func._overloadpacket.__name__
        if name in self._READ_OPS:
            return True
        if name == "repeat_interleave" and func._overloadname == "Tensor":
            return kwargs.get("output_size") is None
        if name in ("index", "index_put", "index_put_", "_index_put_impl_"):
            return any(t is not None and t.dtype in (torch.bool, torch.uint8) for t in args[1])
        return False

    def _wrap_host_copy(self, name, original):
        def count_host_copy(tensor, *args, **kwargs):
            out = original(tensor, *args, **kwargs)
            if tensor in self._device_data and (name != "to" or _check_cpu_target(*args, **kwargs)):
                self._record(f"Tensor.{name}")
                if out is tensor:
                    # on the CPU the copy to the host is the tensor itself; in training it is a fresh host tensor
                    with _disable_current_modes():
                        out = tensor.clone()
            return out

        return count_host_copy

    def _record(self, op):
        frames = [f for f in traceback.extract_stack()[:-2] if "/nemo/" in f.filename]
        if not frames:
            # the autograd engine runs the backward ops without a Python frame of their forward
            self.reads.append(_DeviceRead(op, "backward", "?"))
            return
        self.reads.append(_DeviceRead(op, frames[-1].name, f"{frames[-1].filename}:{frames[-1].lineno}"))


# an op that reads device data on the host, with the NeMo function and the line that issued it
_DeviceRead = namedtuple("_DeviceRead", ["op", "function", "location"])


def _check_cpu_target(*args, **kwargs) -> bool:
    """Check whether ``Tensor.to(*args, **kwargs)`` names the CPU as a string.

    A ``torch.device`` argument is usually the device of another tensor (``x.device``),
    which is the GPU in training although it is the CPU in this run.
    """
    return any(isinstance(value, str) and torch.device(value).type == "cpu" for value in (*args, kwargs.get("device")))


def _make_datasets(model):
    prompt = CanaryPromptFormatter(model.tokenizer)
    padded = PromptedAudioToTextLhotseDataset(tokenizer=model.tokenizer, prompt=prompt)
    packed = PromptedAudioToTextLhotseDataset(tokenizer=model.tokenizer, prompt=prompt, pack_sequences=True)
    return padded, packed


def _calc_packed_features(model, batch):
    """Return the packed features of the preprocessor of ``model`` for a packed batch."""
    with torch.no_grad():
        return model.preprocessor.forward_packed(
            batch.audio, batch.audio_lens, batch.audio_cu_seqlens, host_length=batch.host_audio_lens
        )


def _make_int16_dataset(model):
    return PromptedAudioToTextLhotseDataset(
        tokenizer=model.tokenizer, prompt=CanaryPromptFormatter(model.tokenizer), pack_sequences=True, int16_audio=True
    )


def _move_to_cuda(batch):
    for name, value in vars(batch).items():
        if torch.is_tensor(value):
            setattr(batch, name, value.cuda())
    return batch


def _save_cuts(cuts, tmp_path):
    """Write the audio of ``cuts`` to files and the cuts to a manifest; return the manifest path."""
    path = tmp_path / "cuts.jsonl.gz"
    for cut in cuts:
        # keep the audio and the string fields of the prompt; the in-memory arrays of the dummy cuts do not serialize
        cut.features = None
        cut.custom = {key: value for key, value in cut.custom.items() if isinstance(value, str)}
    cuts.save_audios(tmp_path / "audio").to_file(path)
    return path


def _make_loader_config(path, pin_memory: bool):
    return OmegaConf.create(
        {
            "cuts_path": str(path),
            "use_lhotse": True,
            "sample_rate": 16000,
            "shuffle": False,
            "num_workers": 1,
            "pin_memory": pin_memory,
            "batch_size": 3,
            "seed": 0,
            "shard_seed": 0,
            "force_finite": True,
            "pack_sequences": True,
        }
    )


def _assert_same_batch(batch, expected):
    for name, value in vars(expected).items():
        if torch.is_tensor(value):
            assert torch.equal(getattr(batch, name), value), name
    assert batch.host_audio_lens == expected.host_audio_lens
    assert batch.max_decoder_len == expected.max_decoder_len


def _make_unreadable_cut(cut, tmp_path, text):
    """Return a copy of ``cut`` with its own id and transcript ``text`` whose recording names a missing file."""
    recording = Recording(
        id=f"{cut.id}-unreadable",
        sources=[AudioSource(type="file", channels=[0], source=str(tmp_path / f"{cut.id}-missing.wav"))],
        sampling_rate=cut.sampling_rate,
        num_samples=cut.recording.num_samples,
        duration=cut.recording.duration,
    )
    supervision = fastcopy(cut.supervisions[0], id=recording.id, recording_id=recording.id, text=text)
    return fastcopy(cut, id=recording.id, recording=recording, supervisions=[supervision])
