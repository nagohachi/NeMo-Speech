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

"""Depthwise conv1d (stride 1, no dilation) over the frames of sequence-packed activations ``(T_total, C)``.

The utterances lie one after another along the frame axis, and ``token_slots`` gives every frame its slot on the gapped
time axis of a ``PackedSlotLayout``. Tap ``k`` of frame ``t`` reads frame ``t + d`` with ``d = k - left_padding``
when that frame sits ``d`` slots away, and zero otherwise:

    out[t, c] = bias[c] + sum over k of  weight[c, 0, k] * x[t + d, c] * [token_slots[t + d] - token_slots[t] == d]

With a layout that leaves at least ``kernel_size - 1`` empty slots after every utterance this is exactly
``layout.gather(conv(layout.scatter(x)))`` for the zero-padded convolution on the slots (``CausalConv1D`` with
``left_padding`` frames on the left): a tap that leaves the utterance lands on an empty slot, a tap that stays inside
reads the frame ``d`` slots away. The kernels read and write the ``(T_total, C)`` rows directly, so neither the
gapped copy nor the channel-first transposes of that form are materialized.

The input gradient is the same sum with the taps mirrored, and the weight and bias gradients are column sums over the
frames: every program sums ``ROWS_PER_PROGRAM`` frames into a float32 partial and the partials are added afterwards,
without atomics, so the backward is deterministic. Accumulation is float32; the inputs keep their dtypes, so under
autocast the activations are bfloat16 or float16 while the weight and bias stay float32 (no rounded copies).

The ops are ``torch.library`` custom ops with fake implementations and an autograd formula, so ``torch.compile``
traces them without graph breaks and dynamic frame counts do not recompile.
"""

from typing import Optional

import torch
import triton
import triton.language as tl
from torch import Tensor

__all__ = ["run_packed_depthwise_conv1d"]

# frames summed by one program of the weight-gradient kernel into its float32 partial
_ROWS_PER_PROGRAM = 64


def run_packed_depthwise_conv1d(
    x: Tensor, weight: Tensor, bias: Optional[Tensor], token_slots: Tensor, left_padding: int
) -> Tensor:
    """Run a depthwise conv1d over sequence-packed frames, reading zero beyond every utterance.

    Args:
        x: activations ``(T_total, C)`` on a CUDA device; under autocast they are cast to the autocast dtype,
            as ``nn.Conv1d`` casts its input.
        weight: ``(C, 1, kernel_size)``, the weight of a depthwise ``nn.Conv1d``.
        bias: ``(C,)`` or None.
        token_slots: slot of every frame on a gapped layout, ``(T_total,)``, increasing within an utterance by one
            per frame and by more than one between utterances (``PackedSlotLayout.token_slots``).
        left_padding: zero frames before every utterance, ``0 <= left_padding < kernel_size``;
            ``kernel_size - 1 - left_padding`` frames follow it.

    Returns:
        ``(T_total, C)`` in the dtype of ``x`` after the autocast cast.
    """
    if x.dim() != 2:
        raise ValueError(f"x must be (T_total, C), got shape {tuple(x.shape)}.")
    num_channels = x.shape[1]
    if weight.dim() != 3 or weight.shape[0] != num_channels or weight.shape[1] != 1:
        raise ValueError(f"weight must be (C, 1, kernel_size) with C = {num_channels}, got {tuple(weight.shape)}.")
    if bias is not None and tuple(bias.shape) != (num_channels,):
        raise ValueError(f"bias must be ({num_channels},), got {tuple(bias.shape)}.")
    if token_slots.dim() != 1 or token_slots.shape[0] != x.shape[0]:
        raise ValueError(f"token_slots must be ({x.shape[0]},), got {tuple(token_slots.shape)}.")
    if not 0 <= left_padding < weight.shape[2]:
        raise ValueError(f"left_padding must lie in [0, {weight.shape[2]}), got {left_padding}.")
    if torch.is_autocast_enabled(x.device.type):
        x = x.to(torch.get_autocast_dtype(x.device.type))
    return torch.ops.nemo_asr.packed_depthwise_conv1d(x, weight, bias, token_slots, left_padding)


@torch.library.custom_op("nemo_asr::packed_depthwise_conv1d", mutates_args=(), device_types="cuda")
def _run_conv_forward(
    x: Tensor, weight: Tensor, bias: Optional[Tensor], token_slots: Tensor, left_padding: int
) -> Tensor:
    # a compiled graph may feed the op any layout; the kernels index contiguous rows
    x, weight, token_slots = x.contiguous(), weight.contiguous(), token_slots.contiguous()
    out = torch.empty_like(x, memory_format=torch.contiguous_format)
    if x.numel() > 0:
        _launch_shifted_rows(x, weight, bias, token_slots, out, left_padding, mirrored=False)
    return out


@_run_conv_forward.register_fake
def _make_conv_forward_fake(
    x: Tensor, weight: Tensor, bias: Optional[Tensor], token_slots: Tensor, left_padding: int
) -> Tensor:
    del weight, bias, token_slots, left_padding
    return x.new_empty(tuple(x.shape))


@torch.library.custom_op("nemo_asr::packed_depthwise_conv1d_bwd", mutates_args=(), device_types="cuda")
def _run_conv_backward(
    grad_out: Tensor, x: Tensor, weight: Tensor, token_slots: Tensor, left_padding: int
) -> tuple[Tensor, Tensor, Tensor]:
    grad_out, x = grad_out.contiguous(), x.contiguous()
    weight, token_slots = weight.contiguous(), token_slots.contiguous()
    num_channels, _, kernel_size = weight.shape
    grad_x = torch.empty_like(x, memory_format=torch.contiguous_format)
    if x.numel() == 0:
        return grad_x, weight.new_zeros(tuple(weight.shape)), weight.new_zeros((num_channels,))
    _launch_shifted_rows(grad_out, weight, None, token_slots, grad_x, left_padding, mirrored=True)
    padded_num_taps = triton.next_power_of_2(kernel_size + 1)
    num_programs = triton.cdiv(x.shape[0], _ROWS_PER_PROGRAM)
    partial = torch.empty((num_programs, padded_num_taps, num_channels), dtype=torch.float32, device=x.device)
    grid = lambda meta: (num_programs, triton.cdiv(num_channels, meta["BLOCK_C"]))  # noqa: E731
    _sum_weight_grad_kernel[grid](
        x,
        grad_out,
        token_slots,
        partial,
        x.shape[0],
        num_channels,
        KERNEL=kernel_size,
        LEFT=left_padding,
        TAPS_PAD=padded_num_taps,
        ROWS_PER_PROGRAM=_ROWS_PER_PROGRAM,
    )
    sums = partial.sum(dim=0)
    # outputs of a custom op may not alias each other: both gradients are copies out of the sums
    grad_weight = sums[:kernel_size].t().to(weight.dtype, memory_format=torch.contiguous_format, copy=True)
    return grad_x, grad_weight.reshape(weight.shape), sums[kernel_size].to(weight.dtype, copy=True)


@_run_conv_backward.register_fake
def _make_conv_backward_fake(
    grad_out: Tensor, x: Tensor, weight: Tensor, token_slots: Tensor, left_padding: int
) -> tuple[Tensor, Tensor, Tensor]:
    del grad_out, token_slots, left_padding
    return x.new_empty(tuple(x.shape)), weight.new_empty(tuple(weight.shape)), weight.new_empty((weight.shape[0],))


def _setup_context(ctx, inputs, output):
    del output
    x, weight, bias, token_slots, left_padding = inputs
    ctx.save_for_backward(x, weight, token_slots)
    ctx.left_padding = left_padding
    ctx.bias_dtype = None if bias is None else bias.dtype


def _run_autograd_backward(ctx, grad_out):
    x, weight, token_slots = ctx.saved_tensors
    grad_x, grad_weight, grad_bias = torch.ops.nemo_asr.packed_depthwise_conv1d_bwd(
        grad_out, x, weight, token_slots, ctx.left_padding
    )
    grad_bias = None if ctx.bias_dtype is None else grad_bias.to(ctx.bias_dtype)
    return grad_x, grad_weight, grad_bias, None, None


torch.library.register_autograd(
    "nemo_asr::packed_depthwise_conv1d", _run_autograd_backward, setup_context=_setup_context
)


def _launch_shifted_rows(src, weight, bias, token_slots, out, left_padding, *, mirrored):
    num_rows, num_channels = src.shape
    grid = lambda meta: (
        triton.cdiv(num_rows, meta["BLOCK_T"]),
        triton.cdiv(num_channels, meta["BLOCK_C"]),
    )  # noqa: E731
    _sum_shifted_rows_kernel[grid](
        src,
        weight,
        src if bias is None else bias,  # any pointer when HAS_BIAS is False
        token_slots,
        out,
        num_rows,
        num_channels,
        KERNEL=weight.shape[2],
        LEFT=left_padding,
        SIGN=-1 if mirrored else 1,
        HAS_BIAS=bias is not None,
    )


def _make_row_configs():
    return [
        triton.Config({"BLOCK_T": block_t, "BLOCK_C": block_c}, num_warps=warps)
        for block_t, block_c, warps in ((32, 64, 4), (64, 64, 4), (64, 128, 4), (128, 64, 8), (128, 128, 8))
    ]


def _make_weight_grad_configs():
    return [
        triton.Config({"BLOCK_C": block_c}, num_warps=warps)
        for block_c, warps in ((32, 1), (64, 1), (64, 2), (128, 2), (128, 4), (256, 4))
    ]


@triton.autotune(configs=_make_row_configs(), key=["num_channels"])
@triton.jit
def _sum_shifted_rows_kernel(
    src_ptr,
    weight_ptr,
    bias_ptr,
    slots_ptr,
    out_ptr,
    num_rows,
    num_channels,
    KERNEL: tl.constexpr,
    LEFT: tl.constexpr,
    SIGN: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """Compute ``out[t] = bias + sum_k weight[:, k] * src[t + SIGN * (k - LEFT)]`` over the taps inside the utterance.

    SIGN 1 is the convolution, SIGN -1 its transpose (the input gradient from the output gradient).
    """
    rows = tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)
    cols = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    row_ok = rows < num_rows
    col_ok = cols < num_channels
    slots = tl.load(slots_ptr + rows, mask=row_ok, other=0)
    acc = tl.zeros((BLOCK_T, BLOCK_C), dtype=tl.float32)
    for tap in tl.static_range(KERNEL):
        shift = SIGN * (tap - LEFT)
        src_rows = rows + shift
        ok = row_ok & (src_rows >= 0) & (src_rows < num_rows)
        src_slots = tl.load(slots_ptr + src_rows, mask=ok, other=0)
        # the frame `shift` rows away belongs to the same utterance exactly when it sits `shift` slots away
        ok = ok & (src_slots - slots == shift)
        w = tl.load(weight_ptr + cols * KERNEL + tap, mask=col_ok, other=0.0).to(tl.float32)
        offsets = src_rows.to(tl.int64)[:, None] * num_channels + cols[None, :]
        values = tl.load(src_ptr + offsets, mask=ok[:, None] & col_ok[None, :], other=0.0)
        acc += values.to(tl.float32) * w[None, :]
    if HAS_BIAS:
        acc += tl.load(bias_ptr + cols, mask=col_ok, other=0.0).to(tl.float32)[None, :]
    offsets = rows.to(tl.int64)[:, None] * num_channels + cols[None, :]
    tl.store(out_ptr + offsets, acc.to(out_ptr.dtype.element_ty), mask=row_ok[:, None] & col_ok[None, :])


@triton.autotune(configs=_make_weight_grad_configs(), key=["num_channels"])
@triton.jit
def _sum_weight_grad_kernel(
    x_ptr,
    grad_ptr,
    slots_ptr,
    partial_ptr,
    num_rows,
    num_channels,
    KERNEL: tl.constexpr,
    LEFT: tl.constexpr,
    TAPS_PAD: tl.constexpr,
    ROWS_PER_PROGRAM: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """Write rows ``k < KERNEL`` of the partial, ``sum_t grad[t] * x[t + k - LEFT]`` over the taps inside the
    utterance, and row ``KERNEL``, ``sum_t grad[t]``; the remaining rows up to TAPS_PAD are zero.

    One frame per iteration: its gradient row times the window of input rows its taps read, added to a
    ``(TAPS_PAD, BLOCK_C)`` accumulator, so no reduction across rows is needed until the partials are summed.
    """
    cols = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    col_ok = cols < num_channels
    taps = tl.arange(0, TAPS_PAD)
    shifts = taps - LEFT
    acc = tl.zeros((TAPS_PAD, BLOCK_C), dtype=tl.float32)
    bias_acc = tl.zeros((BLOCK_C,), dtype=tl.float32)
    first = tl.program_id(0) * ROWS_PER_PROGRAM
    for i in range(ROWS_PER_PROGRAM):
        row = first + i
        row_ok = row < num_rows
        slot = tl.load(slots_ptr + row, mask=row_ok, other=0)
        grad = tl.load(grad_ptr + row.to(tl.int64) * num_channels + cols, mask=row_ok & col_ok, other=0.0).to(
            tl.float32
        )
        src_rows = row + shifts
        ok = row_ok & (taps < KERNEL) & (src_rows >= 0) & (src_rows < num_rows)
        src_slots = tl.load(slots_ptr + src_rows, mask=ok, other=0)
        ok = ok & (src_slots - slot == shifts)
        offsets = src_rows.to(tl.int64)[:, None] * num_channels + cols[None, :]
        window = tl.load(x_ptr + offsets, mask=ok[:, None] & col_ok[None, :], other=0.0)
        acc += window.to(tl.float32) * grad[None, :]
        bias_acc += grad
    acc += tl.where(taps[:, None] == KERNEL, bias_acc[None, :], 0.0)
    offsets = (tl.program_id(0) * TAPS_PAD + taps[:, None]).to(tl.int64) * num_channels + cols[None, :]
    tl.store(partial_ptr + offsets, acc, mask=col_ok[None, :])
