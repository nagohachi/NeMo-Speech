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

"""Shared audio-loading policy helpers for Lhotse-backed NeMo datasets."""

from copy import copy

import torch
from lhotse import CutSet
from lhotse.dataset import AudioSamples
from lhotse.dataset.collation import read_audio_from_cuts
from lhotse.dataset.input_strategies import _get_executor


class LhotseAudioLoadingDatasetMixin:
    """Make a dataset's ``AudioSamples`` failure policy loader-configurable.

    Lhotse returns the surviving cuts as a third value only in fault-tolerant
    mode. ``load_audio_with_cuts`` preserves that stable three-value contract
    for NeMo datasets while allowing strict mode to propagate the original
    audio-loading exception.
    """

    def with_fault_tolerant_audio_loading(self, enabled: bool):
        """Return a shallow dataset copy with an independent audio loader."""
        enabled = bool(enabled)
        dataset = copy(self)
        dataset.load_audio = copy(self.load_audio)
        dataset.load_audio.fault_tolerant = enabled
        if dataset.load_audio.ais_batch_loader is not None:
            dataset.load_audio.ais_batch_loader = copy(dataset.load_audio.ais_batch_loader)
            dataset.load_audio.ais_batch_loader.skip_failed_fetches = enabled
        return dataset

    def load_audio_with_cuts(self, cuts):
        """Load audio and always return ``(audio, lengths, surviving_cuts)``."""
        loaded = self.load_audio(cuts)
        if self.load_audio.fault_tolerant:
            return loaded
        audio, audio_lens = loaded
        return audio, audio_lens, cuts


def configure_dataset_audio_loading(dataset, enabled: bool):
    """Apply a loader-level audio policy when the dataset exposes the protocol."""
    configure = getattr(dataset, "with_fault_tolerant_audio_loading", None)
    return configure(enabled) if callable(configure) else dataset


def read_unpadded_audio_rows(cuts: CutSet, load_audio: AudioSamples):
    """Load the mono waveform of every cut without padded collation.

    Returns the float32 waveforms, one 1D tensor per surviving cut, and the cuts that loaded;
    with a fault-tolerant ``load_audio`` the cuts that fail to load are dropped from both.
    """
    if load_audio.use_batch_loader and load_audio.ais_batch_loader is not None:
        cuts = load_audio.ais_batch_loader(cuts)
    executor = _get_executor(load_audio.num_workers, executor_type=load_audio._executor_type)
    audios, surviving = read_audio_from_cuts(
        cuts,
        executor=executor,
        suppress_errors=load_audio.fault_tolerant,
    )
    rows = [torch.as_tensor(audio) for audio in audios]
    mono_downmix = load_audio.mono_downmix
    if mono_downmix is None:
        mono_downmix = not rows or not all(row.ndim == 2 for row in rows)
    if not mono_downmix:
        raise ValueError("Unpadded audio rows require mono_downmix=True.")
    rows = [row.mean(dim=0) if row.ndim == 2 else row for row in rows]
    if any(row.ndim != 1 for row in rows):
        shapes = [tuple(row.shape) for row in rows]
        raise ValueError(f"Unpadded audio rows require one-dimensional waveforms, got {shapes}.")
    return [row.to(torch.float32) for row in rows], surviving
