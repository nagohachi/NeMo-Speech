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


"""Numerical comparisons shared by the ASR tests."""

import torch


def calc_rel_error(actual: torch.Tensor, reference: torch.Tensor) -> float:
    """Return the norm of ``actual - reference`` relative to the norm of ``reference``, in float64.

    A tolerance relative to the tensor's scale suits sums of many terms of both signs, where elementwise
    tolerances are meaningless.
    """
    return float((actual.double() - reference).norm() / reference.norm().clamp(min=1e-30))
