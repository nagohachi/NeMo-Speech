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

"""Helpers shared by the tests of the compiled sequence-packed layers."""

from nemo.collections.asr.parts import packed_sequence as packed_sequence_module


def clear_packed_compile_caches():
    """Clear the cached compiled packed functions, so that the next call compiles, and checks, them again."""
    packed_sequence_module.compile_forward_packed.cache_clear()
    packed_sequence_module.compile_packed_function.cache_clear()
