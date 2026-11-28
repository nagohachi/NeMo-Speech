# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""The Triton depthwise conv1d of packed rows against the gapped-slot convolution it replaces."""

import pytest
import torch
from torch._dynamo.testing import CompileCounterWithBackend

from nemo.collections.asr.parts.packed_sequence import pack_encoder_output, packed_slot_layout
from nemo.collections.asr.parts.submodules.causal_convs import CausalConv1D
from nemo.core.utils.optional_libs import TRITON_AVAILABLE
from tests.collections.asr.numerics_test_utils import calc_rel_error

CUDA_TRITON_AVAILABLE = torch.cuda.is_available() and TRITON_AVAILABLE

if CUDA_TRITON_AVAILABLE:
    from nemo.collections.asr.parts.triton.packed_depthwise_conv import run_packed_depthwise_conv1d

pytestmark = [
    pytest.mark.skipif(not CUDA_TRITON_AVAILABLE, reason="CUDA and Triton are required"),
    pytest.mark.gpu,
]

# lengths shorter than, equal to and longer than the kernels, single frames among them
LENGTHS = [300, 1, 2, 3, 77, 9, 160, 4]
# (num_channels, kernel_size, [left, right] context): d_model 512 with a symmetric kernel of 9, an even kernel,
# a causal one, channel counts that do not fill the channel tiles
GEOMETRIES = [(512, 9, [4, 4]), (48, 9, [4, 4]), (40, 4, [1, 2]), (24, 5, [4, 0])]


@pytest.mark.unit
@pytest.mark.parametrize("has_bias", [True, False])
@pytest.mark.parametrize("num_channels,kernel_size,context", GEOMETRIES)
@pytest.mark.parametrize(
    "extra_gap,alignment",
    [(0, 1), (4, 1), (0, 8)],
    ids=["smallest_gap", "wider_gap", "gaps_that_differ_per_utterance"],
)
def test_matches_gapped_convolution_in_float32(num_channels, kernel_size, context, has_bias, extra_gap, alignment):
    """Check the output and the input, weight and bias gradients against the float64 gapped-slot convolution, with
    the smallest gap the kernel needs (kernel_size - 1), a wider one, and slots aligned to 8 frames, whose gaps
    differ from utterance to utterance."""
    conv, layout, x, grad = _make_case(
        num_channels, kernel_size, context, has_bias, gap=kernel_size - 1 + extra_gap, alignment=alignment
    )
    expected, expected_grads = _run_gapped(conv.double(), layout, x.double(), grad.double())
    conv.float()
    out, grads = _run_kernel(conv, layout, x, grad)
    _assert_close(out, expected, rtol=1e-5)
    for actual, reference in zip(grads, expected_grads, strict=True):
        _assert_close(actual, reference, rtol=1e-5)


@pytest.mark.unit
@pytest.mark.parametrize("num_channels,kernel_size,context", GEOMETRIES[:2])
def test_bf16_autocast_error_at_most_that_of_the_gapped_convolution(num_channels, kernel_size, context):
    """Check both forms under bfloat16 autocast against the float64 result: the kernel's error is at most 1.1 times the
    gapped convolution's (the kernel keeps the weight in float32, nn.Conv1d rounds it to bfloat16)."""
    conv, layout, x, grad = _make_case(num_channels, kernel_size, context, has_bias=True)
    expected, expected_grads = _run_gapped(conv.double(), layout, x.double(), grad.double())
    conv.float()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        gapped, gapped_grads = _run_gapped(conv, layout, x, grad)
        out, grads = _run_kernel(conv, layout, x, grad)
    assert out.dtype == torch.bfloat16
    assert grads[1].dtype == grads[2].dtype == torch.float32
    pairs = [(out, gapped, expected)] + list(zip(grads, gapped_grads, expected_grads, strict=True))
    for actual, generic, reference in pairs:
        error, generic_error = calc_rel_error(actual, reference), calc_rel_error(generic, reference)
        assert error <= 1.1 * generic_error + 1e-6, (error, generic_error)


@pytest.mark.unit
def test_compiles_without_graph_breaks_for_new_frame_counts():
    """Check that with fullgraph and dynamic shapes a second frame count runs the same graph and matches eager."""
    conv, layout, x, grad = _make_case(48, 9, [4, 4], has_bias=True)
    torch._dynamo.reset()
    counter = CompileCounterWithBackend("inductor")
    compiled = torch.compile(_calc_squared_conv_sum, backend=counter, fullgraph=True, dynamic=True)
    for case in ((conv, layout, x), _make_case(48, 9, [4, 4], has_bias=True, lengths=[50, 3, 1, 120])[:3]):
        conv, layout, x = case
        x = x.detach().requires_grad_()
        out = compiled(x, conv.weight, conv.bias, layout.token_slots)
        grads = torch.autograd.grad(out, (x, conv.weight, conv.bias))
        x_eager = x.detach().requires_grad_()
        out_eager = _calc_squared_conv_sum(x_eager, conv.weight, conv.bias, layout.token_slots)
        eager_grads = torch.autograd.grad(out_eager, (x_eager, conv.weight, conv.bias))
        torch.testing.assert_close(out, out_eager)
        for actual, expected in zip(grads, eager_grads, strict=True):
            torch.testing.assert_close(actual, expected)
    assert counter.frame_count == 1


@pytest.mark.unit
def test_empty_input_returns_empty_output_and_zero_gradients():
    """Check that no frames give an empty output and zero weight and bias gradients."""
    weight = torch.randn(8, 1, 9, device="cuda", requires_grad=True)
    bias = torch.randn(8, device="cuda", requires_grad=True)
    x = torch.zeros(0, 8, device="cuda", requires_grad=True)
    out = run_packed_depthwise_conv1d(x, weight, bias, torch.zeros(0, dtype=torch.long, device="cuda"), 4)
    assert out.shape == (0, 8)
    out.sum().backward()
    assert torch.count_nonzero(weight.grad) == 0 and torch.count_nonzero(bias.grad) == 0


@pytest.mark.unit
@pytest.mark.parametrize(
    "x_shape,weight_shape,bias_shape,num_slots,left_padding,match",
    [
        ((5, 8), (8, 1, 9), None, 5, 9, "left_padding"),
        ((5, 8), (8, 1, 9), None, 5, -1, "left_padding"),
        ((5, 8), (4, 1, 9), None, 5, 4, "weight"),
        ((5, 8), (8, 1, 9), (4,), 5, 4, "bias must be"),
        ((5, 8, 1), (8, 1, 9), None, 5, 4, "x must be"),
        ((5, 8), (8, 1, 9), None, 4, 4, "token_slots"),
    ],
)
def test_rejects_inconsistent_arguments(x_shape, weight_shape, bias_shape, num_slots, left_padding, match):
    """Check that shapes and a left padding that do not fit together raise ValueError."""
    with pytest.raises(ValueError, match=match):
        run_packed_depthwise_conv1d(
            torch.zeros(x_shape, device="cuda"),
            torch.zeros(weight_shape, device="cuda"),
            None if bias_shape is None else torch.zeros(bias_shape, device="cuda"),
            torch.arange(num_slots, device="cuda"),
            left_padding,
        )


@pytest.mark.unit
def test_non_contiguous_input_and_bias_give_the_contiguous_result():
    """Check that a strided input and a strided bias (views whose rows or elements are not adjacent) give the result
    of their contiguous copies: the kernels index contiguous rows and read the bias with stride 1."""
    conv, layout, x, _ = _make_case(48, 9, [4, 4], has_bias=True)
    weight, bias = conv.weight.detach(), conv.bias.detach()
    strided_x = x.t().contiguous().t()
    strided_bias = torch.stack([bias, torch.zeros_like(bias)], dim=1)[:, 0]
    assert not strided_x.is_contiguous() and not strided_bias.is_contiguous()
    expected = run_packed_depthwise_conv1d(x, weight, bias, layout.token_slots, conv.left_padding)
    actual = run_packed_depthwise_conv1d(strided_x, weight, strided_bias, layout.token_slots, conv.left_padding)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def _calc_squared_conv_sum(x, weight, bias, token_slots):
    return run_packed_depthwise_conv1d(x, weight, bias, token_slots, 4).square().sum()


def _make_case(num_channels, kernel_size, context, has_bias, lengths=LENGTHS, gap=None, alignment=1):
    """Return a depthwise CausalConv1D, the slot layout of packed random features (``gap`` empty frames after every
    utterance, ``kernel_size - 1`` by default, and slots aligned to ``alignment``), the features and an output
    gradient."""
    torch.manual_seed(0)
    conv = CausalConv1D(
        num_channels, num_channels, kernel_size, padding=context, groups=num_channels, bias=has_bias
    ).cuda()
    lengths = torch.tensor(lengths, device="cuda")
    features = torch.randn(len(lengths), int(lengths.max()), num_channels, device="cuda")
    packed = pack_encoder_output(features, lengths)
    layout = packed_slot_layout(packed, alignment=alignment, gap=kernel_size - 1 if gap is None else gap)
    grad = torch.randn_like(packed.data)
    return conv, layout, packed.data, grad


def _run_gapped(conv, layout, x, grad):
    x = x.detach().requires_grad_()
    out = layout.gather(conv(layout.scatter(x).transpose(0, 1).unsqueeze(0)).squeeze(0).transpose(0, 1))
    return out, _calc_grads(out, x, conv, grad)


def _run_kernel(conv, layout, x, grad):
    x = x.detach().requires_grad_()
    out = run_packed_depthwise_conv1d(x, conv.weight, conv.bias, layout.token_slots, conv.left_padding)
    return out, _calc_grads(out, x, conv, grad)


def _calc_grads(out, x, conv, grad):
    inputs = [x, conv.weight] + ([conv.bias] if conv.bias is not None else [])
    return [g.detach() for g in torch.autograd.grad(out, inputs, grad.to(out.dtype))]


def _assert_close(actual, expected, rtol):
    # relative to the tensor's scale: sums of many terms of both signs make elementwise tolerances meaningless
    assert calc_rel_error(actual, expected) <= rtol
