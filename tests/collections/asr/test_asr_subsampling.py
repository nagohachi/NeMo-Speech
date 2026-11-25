# SPDX-FileCopyrightText: Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
import os
import tempfile

import onnx
import pytest
import torch
from omegaconf import ListConfig

from nemo.collections.asr.models import ASRModel
from nemo.collections.asr.parts.packed_sequence import PackedEncoderActivations, pack_encoder_output
from nemo.collections.asr.parts.submodules import subsampling as subsampling_module
from nemo.collections.asr.parts.submodules.subsampling import (
    ConvSubsampling,
    SubsamplingReductionModule,
    _layer_padding,
)


@pytest.mark.gpu
class TestASRSubsamplingConvChunking:
    @pytest.mark.with_downloads()
    @pytest.mark.unit
    def test_forward(self):
        asr_model = ASRModel.from_pretrained("stt_en_fastconformer_ctc_large")
        asr_model = asr_model.eval()
        asr_model.preprocessor.featurizer.dither = 0.0
        asr_model.preprocessor.featurizer.pad_to = 0

        len = 512

        input_signal_batch1 = torch.randn(size=(1, len), device=asr_model.device)
        length_batch1 = torch.randint(low=321, high=500, size=[1], device=asr_model.device)

        input_signal_batch4 = torch.randn(size=(4, len), device=asr_model.device)
        length_batch4 = torch.randint(low=321, high=500, size=[4], device=asr_model.device)

        with torch.inference_mode():
            # regular inference
            logprobs_batch1_nosplit, _, _ = asr_model.forward(
                input_signal=input_signal_batch1, input_signal_length=length_batch1
            )
            logprobs_batch4_nosplit, _, _ = asr_model.forward(
                input_signal=input_signal_batch4, input_signal_length=length_batch4
            )

            # force chunking to 2
            asr_model.change_subsampling_conv_chunking_factor(subsampling_conv_chunking_factor=2)

            # chunked inference by channels as batch is 1
            logprobs_batch1_split, _, _ = asr_model.forward(
                input_signal=input_signal_batch1, input_signal_length=length_batch1
            )
            # chunked inference by batch as it is 4 [> 1]
            logprobs_batch4_split, _, _ = asr_model.forward(
                input_signal=input_signal_batch4, input_signal_length=length_batch4
            )

        diff = torch.mean(torch.abs(logprobs_batch1_split - logprobs_batch1_nosplit))
        assert diff <= 0.2
        diff = torch.mean(torch.abs(logprobs_batch4_split - logprobs_batch4_nosplit))
        assert diff <= 0.2


def _build_conv_subsampling(feat_in=16, conv_channels=8, factor=4):
    """A tiny dw_striding ConvSubsampling for unit-testing the 32-bit chunking logic."""
    return ConvSubsampling(
        subsampling="dw_striding",
        subsampling_factor=factor,
        feat_in=feat_in,
        feat_out=32,
        conv_channels=conv_channels,
        subsampling_conv_chunking_factor=1,
    ).eval()


def _install_split_spy(monkeypatch, sub):
    """Record the batch size of every conv_split_by_batch call; returns the list."""
    calls = []
    original_split = sub.conv_split_by_batch

    def spy_split(inp, lens):
        calls.append(int(inp.shape[0]))
        return original_split(inp, lens)

    monkeypatch.setattr(sub, "conv_split_by_batch", spy_split)
    return calls


class TestConvSubsampling32BitIndexing:
    """Guard and auto-chunking tests, run on small inputs with the limit lowered via monkeypatch."""

    @pytest.mark.unit
    @pytest.mark.parametrize("shape", [(1, 7, 16), (3, 50, 16), (5, 123, 16)])
    def test_first_conv_output_numel_matches_real_conv(self, shape):
        sub = _build_conv_subsampling()
        x = torch.randn(*shape)
        real_numel = sub.conv[0](x.unsqueeze(1)).numel()  # only the first Conv2d
        assert sub._first_conv_output_numel(x) == real_numel

    @pytest.mark.unit
    @pytest.mark.parametrize("batch_size", [1, 4])
    def test_guard_splits_at_exact_limit(self, monkeypatch, batch_size):
        # '>=' must split at output == limit; the old '>' let an INT_MAX tensor through.
        sub = _build_conv_subsampling()
        x = torch.randn(batch_size, 50, 16)
        lengths = torch.full((batch_size,), 50, dtype=torch.long)

        ref, ref_len = sub(x.clone(), lengths.clone())

        split_calls = _install_split_spy(monkeypatch, sub)
        monkeypatch.setattr(subsampling_module, "_MAX_CONV_NUMEL_32BIT", sub._first_conv_output_numel(x))

        out, out_len = sub(x.clone(), lengths.clone())

        assert split_calls, "the guard did not split when the first-conv output equals the limit"
        assert torch.allclose(out, ref, atol=1e-5)
        assert torch.equal(out_len, ref_len)

    @pytest.mark.unit
    def test_guard_does_not_split_below_limit(self, monkeypatch):
        # One element below the limit must not split.
        sub = _build_conv_subsampling()
        x = torch.randn(4, 50, 16)
        lengths = torch.full((4,), 50, dtype=torch.long)

        split_calls = _install_split_spy(monkeypatch, sub)
        monkeypatch.setattr(subsampling_module, "_MAX_CONV_NUMEL_32BIT", sub._first_conv_output_numel(x) + 1)

        sub(x.clone(), lengths.clone())
        assert not split_calls

    @pytest.mark.unit
    @pytest.mark.parametrize("batch_size", [4, 8, 16])
    def test_auto_chunking_keeps_each_chunk_below_limit(self, monkeypatch, batch_size):
        # Each chunk's first-conv output must end up strictly below the limit.
        sub = _build_conv_subsampling()
        x = torch.randn(batch_size, 40, 16)
        lengths = torch.full((batch_size,), 40, dtype=torch.long)
        limit = sub._first_conv_output_numel(x) // 3 + 1  # forces a multi-way split
        monkeypatch.setattr(subsampling_module, "_MAX_CONV_NUMEL_32BIT", limit)

        chunk_batches = []
        original_forward = sub.conv.forward

        def recording_forward(inp, lens):
            chunk_batches.append(int(inp.shape[0]))
            return original_forward(inp, lens)

        monkeypatch.setattr(sub.conv, "forward", recording_forward)

        sub(x.clone(), lengths.clone())

        assert len(chunk_batches) > 1, "expected the input to be split into multiple chunks"
        for chunk_batch in chunk_batches:
            assert sub._first_conv_output_numel(x[:chunk_batch]) < limit

    @pytest.mark.unit
    def test_split_by_batch_uses_size_one_when_factor_exceeds_batch(self, monkeypatch):
        # b=3 makes cf round up to 4, so b // cf == 0; must use single-sample batches instead
        # of the channel fallback (which misreads the batch as channels and errors out).
        sub = _build_conv_subsampling()
        x = torch.randn(3, 50, 16)
        lengths = torch.full((3,), 50, dtype=torch.long)

        ref, ref_len = sub(x.clone(), lengths.clone())

        # Limit above one sample but below the whole batch, so cf == 4 > b == 3 yet a sample fits.
        limit = sub._first_conv_output_numel(x) // 2
        assert sub._first_conv_output_numel(x[:1]) < limit
        monkeypatch.setattr(subsampling_module, "_MAX_CONV_NUMEL_32BIT", limit)

        channel_calls = []
        original_channel = sub.conv_split_by_channel

        def spy_channel(inp):
            channel_calls.append(int(inp.shape[0]))
            return original_channel(inp)

        monkeypatch.setattr(sub, "conv_split_by_channel", spy_channel)

        chunk_batches = []
        original_forward = sub.conv.forward

        def recording_forward(inp, lens):
            chunk_batches.append(int(inp.shape[0]))
            return original_forward(inp, lens)

        monkeypatch.setattr(sub.conv, "forward", recording_forward)

        out, out_len = sub(x.clone(), lengths.clone())

        assert not channel_calls, "should split by batch (size 1), not fall back to channel splitting"
        assert chunk_batches == [1, 1, 1], f"expected three single-sample chunks, got {chunk_batches}"
        assert torch.allclose(out, ref, atol=1e-5)
        assert torch.equal(out_len, ref_len)


class TestSubsamplingReductionModulePooling:
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    @pytest.mark.parametrize("reduction_factor", [2, 4, 8])
    def test_pooling_lengths_match_output(self, reduction_factor):
        """Pool-based reduction applies a single MaxPool1d(kernel_size=reduction_factor),
        so the returned lengths must match that one pooling step."""
        module = SubsamplingReductionModule(reduction='pooling', d_model=8, reduction_factor=reduction_factor)

        x = torch.randn(2, 100, 8)
        lengths = torch.tensor([100, 90])

        out, out_lengths = module(x, lengths)

        assert out.shape == (2, out.shape[1], 8)
        assert out_lengths[0].item() == out.shape[1]
        expected = torch.div(lengths - reduction_factor, reduction_factor, rounding_mode='floor') + 1
        assert out_lengths.tolist() == expected.tolist()


def _build_strided_conv_subsampling(**overrides):
    """A tiny striding ConvSubsampling with per-stage time strides and a frequency pool."""
    options = dict(
        subsampling="striding",
        subsampling_factor=6,
        feat_in=8,
        feat_out=16,
        conv_channels=[2, 3, 3],
        conv_strides=[1, 3, 2],
        freq_pools=[2, 1, 1],
        out_bias=False,
    )
    options.update(overrides)
    return ConvSubsampling(**options).eval()


class TestConvSubsamplingStrides:
    @pytest.mark.unit
    def test_lengths_shapes_and_padding_invariance(self):
        torch.manual_seed(0)
        module = _build_strided_conv_subsampling()
        features = torch.randn(2, 50, 8)
        lengths = torch.tensor([50, 31])
        with torch.no_grad():
            out, out_lengths = module(x=features, lengths=lengths)
            alone, alone_lengths = module(x=features[1:, :31], lengths=lengths[1:])
        # time strides 1, 3, 2 with kernel 3 and padding 1: ceil(ceil(T / 3) / 2)
        assert out.shape == (2, 9, 16)
        assert out_lengths.tolist() == [9, 6] and alone_lengths.tolist() == [6]
        assert module.out.bias is None
        # the frequency axis is halved once by the pool, so the projection sees 3 channels x 4 bins
        assert module.out.weight.shape == (16, 12)
        torch.testing.assert_close(out[1, :6], alone[0, :6], atol=1e-5, rtol=1e-5)

    @pytest.mark.unit
    def test_channels_as_omegaconf_list(self):
        module = _build_strided_conv_subsampling(conv_channels=ListConfig([2, 3, 3]), freq_pools=ListConfig([2, 1, 1]))
        assert module.out.weight.shape == (16, 12)

    @pytest.mark.unit
    def test_validation(self):
        with pytest.raises(ValueError, match="product of conv_strides"):
            ConvSubsampling("striding", 4, 8, 16, [2, 2], conv_strides=[1, 3])
        with pytest.raises(ValueError, match="only supported with subsampling='striding'"):
            ConvSubsampling("dw_striding", 6, 8, 16, [2, 2, 2], conv_strides=[1, 3, 2])
        with pytest.raises(ValueError, match="one entry per stride"):
            ConvSubsampling("striding", 6, 8, 16, [2, 2], conv_strides=[1, 3, 2])
        with pytest.raises(ValueError, match="needs conv_strides"):
            ConvSubsampling("striding", 4, 8, 16, [2, 3])
        with pytest.raises(ValueError, match="conv_strides must be positive integers"):
            ConvSubsampling("striding", 6, 8, 16, 4, conv_strides=[-1, -3, 2])
        with pytest.raises(ValueError, match="freq_pools must be positive integers"):
            ConvSubsampling("striding", 6, 8, 16, 4, conv_strides=[1, 3, 2], freq_pools=[0, 1, 1])

    @pytest.mark.unit
    @pytest.mark.parametrize("batch_size", [4, 1])
    def test_batch_chunking_matches_unchunked(self, batch_size):
        torch.manual_seed(0)
        module = _build_strided_conv_subsampling()
        features = torch.randn(batch_size, 50, 8)
        lengths = torch.tensor([50, 31, 50, 12][:batch_size])
        with torch.no_grad():
            reference, reference_lengths = module(x=features, lengths=lengths)
            # a factor above 1 always splits by batch; a batch of one falls back to the unsplit convolution
            module.change_subsampling_conv_chunking_factor(2)
            out, out_lengths = module(x=features, lengths=lengths)
        torch.testing.assert_close(out, reference, atol=1e-5, rtol=1e-5)
        assert out_lengths.tolist() == reference_lengths.tolist()

    @pytest.mark.unit
    def test_export_to_onnx(self):
        torch.manual_seed(0)
        module = _build_strided_conv_subsampling()
        features = torch.randn(2, 50, 8)
        lengths = torch.tensor([50, 31])
        with tempfile.TemporaryDirectory() as tmpdir:
            filename = os.path.join(tmpdir, 'subsampling.onnx')
            torch.onnx.export(
                module,
                (features, lengths),
                filename,
                input_names=['features', 'lengths'],
                output_names=['outputs', 'out_lengths'],
                dynamic_axes={
                    'features': {0: 'batch', 1: 'time'},
                    'lengths': {0: 'batch'},
                    'outputs': {0: 'batch', 1: 'time'},
                    'out_lengths': {0: 'batch'},
                },
                opset_version=17,
                dynamo=False,
            )
            onnx_model = onnx.load(filename)
            onnx.checker.check_model(onnx_model, full_check=True)  # throws when failed
            assert [node.name for node in onnx_model.graph.input] == ['features', 'lengths']
            try:
                import onnxruntime
            except ImportError:
                return
            session = onnxruntime.InferenceSession(filename, providers=['CPUExecutionProvider'])
            outputs, out_lengths = session.run(None, {'features': features.numpy(), 'lengths': lengths.numpy()})
        with torch.no_grad():
            reference, reference_lengths = module(x=features, lengths=lengths)
        torch.testing.assert_close(torch.from_numpy(outputs), reference, atol=1e-5, rtol=1e-5)
        assert out_lengths.tolist() == reference_lengths.tolist()

    @pytest.mark.unit
    def test_production_configuration(self):
        # 80 mel features, channels 32, 64, 64, time strides 1, 3, 2 and one frequency pool of 2
        module = _build_strided_conv_subsampling(feat_in=80, feat_out=256, conv_channels=[32, 64, 64])
        features = torch.randn(3, 101, 80)
        lengths = torch.tensor([101, 67, 13])
        with torch.no_grad():
            out, out_lengths = module(x=features, lengths=lengths)
        assert out.shape == (3, 17, 256) and out_lengths.tolist() == [17, 12, 3]
        assert module.out.weight.shape == (256, 64 * 40)
        # the 32-bit indexing guard counts the first convolution's output elements,
        # which in this configuration is the largest activation of the stack
        first_conv_numel = module.conv[0](features.unsqueeze(1)).numel()
        assert module._first_conv_output_numel(features) == first_conv_numel
        assert module._max_conv_numel(features) == first_conv_numel

    @pytest.mark.unit
    def test_max_conv_numel_counts_every_stage(self):
        # channels 2 then 8 with time strides 2, 1 and no pooling: the second convolution's output
        # is four times the first one's, so the first-conv estimate alone would under-count the stack
        module = _build_strided_conv_subsampling(
            subsampling_factor=2, conv_channels=[2, 8], conv_strides=[2, 1], freq_pools=[1, 1]
        )
        features = torch.randn(4, 50, 8)
        activation_numels = [features.numel()]
        x = features.unsqueeze(1)
        with torch.no_grad():
            for layer in module.conv:
                x = layer(x)
                if isinstance(layer, torch.nn.Conv2d):
                    activation_numels.append(x.numel())
        assert activation_numels == [1600, 1600, 6400]
        assert module._max_conv_numel(features) == max(activation_numels)
        assert module._first_conv_output_numel(features) == 1600
        # a frequency pool after the first stage halves the second convolution's map
        pooled = _build_strided_conv_subsampling(
            subsampling_factor=2, conv_channels=[2, 8], conv_strides=[2, 1], freq_pools=[2, 1]
        )
        assert pooled._max_conv_numel(features) == 3200

    @pytest.mark.unit
    def test_guard_splits_when_a_later_convolution_is_largest(self, monkeypatch):
        torch.manual_seed(0)
        module = _build_strided_conv_subsampling(
            subsampling_factor=2, conv_channels=[2, 8], conv_strides=[2, 1], freq_pools=[1, 1]
        )
        features = torch.randn(4, 50, 8)
        lengths = torch.tensor([50, 31, 50, 12])
        with torch.no_grad():
            reference, reference_lengths = module(x=features, lengths=lengths)
        # the limit sits at the second convolution's output, above the input and the first output
        limit = module._max_conv_numel(features)
        assert limit > max(features.numel(), module._first_conv_output_numel(features))
        monkeypatch.setattr(subsampling_module, "_MAX_CONV_NUMEL_32BIT", limit)
        split_calls = _install_split_spy(monkeypatch, module)
        chunk_batches = []
        original_forward = module.conv.forward

        def recording_forward(inp, lens):
            chunk_batches.append(int(inp.shape[0]))
            return original_forward(inp, lens)

        monkeypatch.setattr(module.conv, "forward", recording_forward)
        with torch.no_grad():
            out, out_lengths = module(x=features, lengths=lengths)
        assert split_calls == [4] and chunk_batches == [2, 2]
        for chunk_batch in chunk_batches:
            assert module._max_conv_numel(features[:chunk_batch]) < limit
        torch.testing.assert_close(out, reference, atol=1e-5, rtol=1e-5)
        assert out_lengths.tolist() == reference_lengths.tolist()
        # with a frequency pool after the first stage the stack stays below the same limit: no split
        pooled = _build_strided_conv_subsampling(
            subsampling_factor=2, conv_channels=[2, 8], conv_strides=[2, 1], freq_pools=[2, 1]
        )
        pooled_split_calls = _install_split_spy(monkeypatch, pooled)
        with torch.no_grad():
            pooled(x=features, lengths=lengths)
        assert not pooled_split_calls

    @pytest.mark.unit
    def test_defaults_unchanged(self):
        module = ConvSubsampling("striding", 4, 8, 16, 4)
        assert module.out.bias is not None and module.out.weight.shape == (16, 8)
        features = torch.randn(1, 20, 8)
        out, out_lengths = module(x=features, lengths=torch.tensor([20]))
        assert out.shape == (1, 5, 16) and out_lengths.tolist() == [5]

    @pytest.mark.unit
    def test_reset_parameters_without_output_bias(self):
        # reset_parameters re-initializes the dw_striding stages and the projection; no bias to draw without out_bias
        module = ConvSubsampling("dw_striding", 4, 8, 16, 4, out_bias=False)
        assert module.out.bias is None
        module.reset_parameters()
        with_bias = ConvSubsampling("dw_striding", 4, 8, 16, 4)
        torch.nn.init.zeros_(with_bias.out.bias)
        with_bias.reset_parameters()
        assert with_bias.out.bias.abs().sum() > 0

    @pytest.mark.unit
    def test_layer_padding_of_pooling_layers(self):
        assert _layer_padding(torch.nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))) == (0, 0)
        assert _layer_padding(torch.nn.Conv2d(1, 1, kernel_size=3, padding=1)) == (1, 1)


def _valid_frames(padded, lengths):
    """Rows of a padded (B, T, D) batch inside the lengths, in packing order."""
    mask = torch.arange(padded.shape[1], device=padded.device)[None, :] < lengths[:, None]
    return padded[mask]


def _strided_subsampling():
    return ConvSubsampling(
        subsampling="striding",
        subsampling_factor=6,
        feat_in=8,
        feat_out=12,
        conv_channels=[2, 3, 3],
        conv_strides=[1, 3, 2],
        freq_pools=[2, 1, 1],
        out_bias=False,
    )


def _packed_test_features(lengths, feat_in=8):
    torch.manual_seed(0)
    return torch.randn(len(lengths), int(max(lengths)), feat_in, dtype=torch.float64)


class TestPackedConvSubsampling:
    """forward_packed against forward at the valid frames: outputs, lengths and gradients."""

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "module",
        [
            _strided_subsampling,
            lambda: ConvSubsampling("striding", 4, 8, 12, 4),
            lambda: ConvSubsampling("dw_striding", 8, 8, 12, 4),
        ],
        ids=["strides_1_3_2_pool", "striding_4", "dw_striding_8"],
    )
    def test_matches_padded_outputs_lengths_and_gradients(self, module):
        module = module().double()
        lengths = torch.tensor([37, 1, 12, 36, 6])
        features = _packed_test_features(lengths).requires_grad_()
        padded_out, padded_lengths = module(x=features, lengths=lengths)
        packed_features = features.detach().clone().requires_grad_()
        packed_out = module.forward_packed(pack_encoder_output(packed_features, lengths))

        assert packed_out.lengths.tolist() == padded_lengths.tolist()
        assert packed_out.data.shape == (int(padded_lengths.sum()), 12)
        torch.testing.assert_close(packed_out.data, _valid_frames(padded_out, padded_lengths))

        _valid_frames(padded_out, padded_lengths).square().sum().backward()
        packed_out.data.square().sum().backward()
        torch.testing.assert_close(_valid_frames(packed_features.grad, lengths), _valid_frames(features.grad, lengths))

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "module",
        [
            _strided_subsampling,
            lambda: ConvSubsampling("striding", 4, 8, 12, 4),
            lambda: ConvSubsampling("dw_striding", 8, 8, 12, 4),
        ],
        ids=["strides_1_3_2_pool", "striding_4", "dw_striding_8"],
    )
    def test_gap_values_reach_no_output_or_gradient(self, module):
        """Check, with random biases that leave values in the gaps between utterances (a 1x1 convolution before a
        depthwise one, an activation before a strided convolution), that forward_packed matches forward."""
        module = module().double()
        torch.manual_seed(1)
        for layer in module.modules():
            if isinstance(layer, (torch.nn.Conv2d, torch.nn.Linear)) and layer.bias is not None:
                torch.nn.init.normal_(layer.bias)
        lengths = torch.tensor([37, 1, 12, 36, 6])
        features = _packed_test_features(lengths).requires_grad_()
        padded_out, padded_lengths = module(x=features, lengths=lengths)
        packed_features = features.detach().clone().requires_grad_()
        packed_out = module.forward_packed(pack_encoder_output(packed_features, lengths))
        torch.testing.assert_close(packed_out.data, _valid_frames(padded_out, padded_lengths))

        _valid_frames(padded_out, padded_lengths).square().sum().backward()
        reference_grads = {name: p.grad.clone() for name, p in module.named_parameters()}
        module.zero_grad()
        packed_out.data.square().sum().backward()
        torch.testing.assert_close(_valid_frames(packed_features.grad, lengths), _valid_frames(features.grad, lengths))
        for name, p in module.named_parameters():
            torch.testing.assert_close(p.grad, reference_grads[name], msg=name)

    @pytest.mark.unit
    def test_sequence_groups_match_one_group(self):
        module = _strided_subsampling().double()
        lengths = torch.tensor([37, 1, 12, 36, 6])
        packed = pack_encoder_output(_packed_test_features(lengths), lengths)
        whole = module.forward_packed(packed)
        module.subsampling_conv_chunking_factor = 2
        grouped = module.forward_packed(packed)
        torch.testing.assert_close(grouped.data, whole.data)
        assert grouped.lengths.tolist() == whole.lengths.tolist()

    @pytest.mark.unit
    @pytest.mark.parametrize("chunking_factor", [1, 2])
    def test_output_host_lengths(self, chunking_factor):
        """Check that the output carries a host copy of its lengths, computed as the device lengths are;
        an input without a host copy (one device read) gives the same result."""
        module = ConvSubsampling("dw_striding", 8, 8, 12, 4).double()
        module.subsampling_conv_chunking_factor = chunking_factor
        lengths = torch.tensor([37, 1, 12, 0, 36, 6])
        with_host_lengths = pack_encoder_output(_packed_test_features(lengths), lengths)
        without_host_lengths = PackedEncoderActivations.from_validated(
            with_host_lengths.data,
            with_host_lengths.lengths,
            with_host_lengths.cu_seqlens,
            with_host_lengths.max_seqlen,
        )
        expected = module.forward_packed(without_host_lengths)
        actual = module.forward_packed(with_host_lengths)
        torch.testing.assert_close(actual.data, expected.data, rtol=0, atol=0)
        for out in (actual, expected):
            assert out.host_lengths.device.type == "cpu"
            assert torch.equal(out.host_lengths, out.lengths)
            assert out.max_seqlen == int(out.lengths.max())

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "layer, reads",
        [
            (torch.nn.Conv2d(2, 2, kernel_size=3, stride=2, padding=1), True),
            (torch.nn.Conv2d(2, 2, kernel_size=3, stride=2, padding=1, groups=2), True),
            (torch.nn.Conv2d(2, 2, kernel_size=(1, 3), padding=(0, 1)), False),
            (torch.nn.Conv2d(2, 2, kernel_size=1), False),
            (torch.nn.Conv2d(2, 2, kernel_size=1, padding=(1, 0)), True),
            (torch.nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)), False),
            (torch.nn.ReLU(), False),
        ],
    )
    def test_reads_neighbouring_frames_detects_time_kernel_and_padding(self, layer, reads):
        """Check that a layer reads neighbouring frames when its time kernel is above one or it pads in time."""
        assert subsampling_module._reads_neighbouring_frames(layer) is reads

    @pytest.mark.unit
    def test_rejects_unsupported_stacks(self):
        packed = pack_encoder_output(torch.randn(1, 10, 8), torch.tensor([10]))
        with pytest.raises(NotImplementedError):
            ConvSubsampling("striding_conv1d", 4, 8, 12, 4).forward_packed(packed)
        with pytest.raises(NotImplementedError):
            ConvSubsampling("striding", 4, 8, 12, 4, is_causal=True).forward_packed(packed)
        with pytest.raises(NotImplementedError, match="vggnet"):
            ConvSubsampling("vggnet", 4, 8, 12, 4).forward_packed(packed)

    @pytest.mark.unit
    def test_empty_batch_keeps_gradients(self):
        """A batch without utterances (a rank's share may be empty) gives an empty output with every gradient edge."""
        module = ConvSubsampling("striding", 4, 8, 12, 4).double()
        features = torch.empty(0, 0, 8, dtype=torch.float64, requires_grad=True)
        out = module.forward_packed(pack_encoder_output(features, torch.empty(0, dtype=torch.long)))
        assert out.data.shape == (0, 12)
        assert out.lengths.tolist() == [] and out.cu_seqlens.tolist() == [0] and out.max_seqlen == 0
        out.data.sum().backward()
        assert features.grad is not None
        assert all(parameter.grad is not None for parameter in module.parameters())

    @pytest.mark.unit
    def test_rejects_a_stack_that_reads_across_utterances(self):
        module = ConvSubsampling("striding", 4, 8, 12, 4)
        # the second 3x3 stage has two empty frames after every utterance; a time padding of three would read the next
        module.conv[2].padding = (3, 3)
        packed = pack_encoder_output(torch.randn(1, 10, 8), torch.tensor([10]))
        with pytest.raises(NotImplementedError, match="reads beyond the 2 empty frames"):
            module.forward_packed(packed)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available() or not subsampling_module.TRITON_AVAILABLE, reason="CUDA, Triton")
@pytest.mark.timeout(900)  # the first call compiles the ReLU and mask kernel and autotunes the depthwise kernels
class TestPackedConvSubsamplingChannelsLast:
    """On CUDA the packed dw_striding stack with ReLU runs channels-last: the first stage in the fused kernel of the
    padded forward masked by the utterances' slots (or, for an input that needs a gradient, the first convolution as a
    matmul over the unfolded patches), the other depthwise convolutions in the Triton kernel of the padded forward,
    the ReLU and the mask of a depthwise input in one compiled kernel. It computes the function of the generic path."""

    @pytest.mark.unit
    @pytest.mark.parametrize("input_grad", [False, True], ids=["fused_first_stage", "unfolded_first_stage"])
    def test_matches_the_generic_path_in_float32(self, monkeypatch, input_grad):
        # float32 in both paths: no TF32 in cuDNN's convolutions of the generic path or in the matmuls
        monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", False)
        monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
        module, features, lengths = _make_cuda_packed_case()
        calls = []
        channels_last = subsampling_module.ConvSubsampling._forward_packed_group_channels_last
        monkeypatch.setattr(
            subsampling_module.ConvSubsampling,
            "_forward_packed_group_channels_last",
            lambda *args, **kwargs: calls.append(1) or channels_last(*args, **kwargs),
        )
        (out, grads), (reference, reference_grads) = (
            _run_packed_subsampling(module, features.requires_grad_(input_grad), lengths, channels_last=path)
            for path in (True, False)
        )
        assert calls, "the channels-last path did not run"
        # the fused first stage multiplies float32 with tl.dot, which rounds the operands to TF32 (10-bit mantissa),
        # as in the padded forward; the unfolded first stage computes in float32
        tolerance, floor = (1e-5, 1e-6) if input_grad else (1e-3, 1e-3)
        torch.testing.assert_close(out, reference, rtol=tolerance, atol=tolerance * float(reference.abs().max()))
        scale = max(float(g.abs().max()) for g in reference_grads.values())
        for name, grad in reference_grads.items():
            assert (grads[name] - grad).norm() <= 10 * tolerance * grad.norm() + floor * scale, name

    @pytest.mark.unit
    @pytest.mark.parametrize("input_grad", [False, True], ids=["fused_first_stage", "unfolded_first_stage"])
    def test_bfloat16_is_as_close_to_float32_as_the_generic_path(self, monkeypatch, input_grad):
        monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", False)
        monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
        module, features, lengths = _make_cuda_packed_case()
        features.requires_grad_(input_grad)
        reference, reference_grads = _run_packed_subsampling(module, features, lengths, channels_last=False)
        generic, generic_grads = _run_packed_subsampling(module, features, lengths, channels_last=False, bf16=True)
        out, grads = _run_packed_subsampling(module, features, lengths, channels_last=True, bf16=True)
        assert (out - reference).norm() <= 1.5 * (generic - reference).norm() + 1e-3 * reference.norm()
        scale = max(float(g.abs().max()) for g in reference_grads.values())
        for name, grad in reference_grads.items():
            error, generic_error = (grads[name] - grad).norm(), (generic_grads[name] - grad).norm()
            assert error <= 1.5 * generic_error + 2e-4 * scale, name


def _make_cuda_packed_case():
    torch.manual_seed(0)
    module = ConvSubsampling("dw_striding", 8, 16, 24, 8).cuda().train()
    assert module.conv.fuse_triton
    lengths = torch.tensor([300, 77, 160, 9, 241], device="cuda")
    features = torch.randn(len(lengths), int(lengths.max()), 16, device="cuda")
    return module, features, lengths


def _run_packed_subsampling(module, features, lengths, *, channels_last, bf16=False):
    """Return the packed output and the parameter gradients of a random projection of it, on one of the two paths."""
    module.conv.fuse_triton = channels_last
    module.zero_grad(set_to_none=True)
    try:
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
            out = module.forward_packed(pack_encoder_output(features, lengths)).data.float()
        weights = torch.randn(out.shape, generator=torch.Generator().manual_seed(3)).cuda()
        (out * weights).sum().backward()
    finally:
        module.conv.fuse_triton = True
    return out.detach(), {name: p.grad.float().clone() for name, p in module.named_parameters()}
