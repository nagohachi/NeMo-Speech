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
from functools import partial
from io import BytesIO
from math import isclose

import numpy as np
import soundfile
import torch
from lhotse import CutSet, MonoCut
from lhotse.audio.recording import assert_and_maybe_fix_num_samples
from lhotse.audio.utils import get_audio_duration_mismatch_tolerance
from lhotse.dataset import AudioSamples
from lhotse.dataset.collation import read_audio_from_cuts
from lhotse.dataset.input_strategies import _get_executor
from lhotse.utils import compute_num_samples

from nemo.utils import logging


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


def read_unpadded_audio_rows(cuts: CutSet, load_audio: AudioSamples, *, dtype: torch.dtype = torch.float32):
    """Load the mono waveform of every cut without padded collation.

    Returns the waveforms, one 1D tensor per surviving cut, and the cuts that loaded;
    with a fault-tolerant ``load_audio`` the cuts that fail to load are dropped from both.

    ``dtype`` float32 gives Lhotse's samples in [-1, 1). ``dtype`` int16 decodes the 16-bit integer samples
    with soundfile instead, half the bytes to collate, pin and copy to the GPU; divided by 32768
    (``INT16_AUDIO_SCALE`` of the preprocessor that converts them) they are the float32 samples bit for bit. It reads mono 16-bit PCM recordings from a file or from memory as
    Lhotse would (same offset, sample count and length correction) and raises on cuts it cannot read that way
    (several sources or channels, audio transforms, resampling, video, other source types, other sample formats),
    also when ``load_audio`` is fault-tolerant.
    """
    if dtype not in (torch.float32, torch.int16):
        raise ValueError(f"Unpadded audio rows are float32 or int16, got {dtype}.")
    if load_audio.use_batch_loader and load_audio.ais_batch_loader is not None:
        cuts = load_audio.ais_batch_loader(cuts)
    executor = _get_executor(load_audio.num_workers, executor_type=load_audio._executor_type)
    if dtype == torch.int16:
        return _read_int16_rows(cuts, executor, suppress_errors=load_audio.fault_tolerant)
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


def _read_int16_rows(cuts: CutSet, executor, *, suppress_errors: bool):
    """Return the int16 waveforms of the cuts and the cuts that loaded (``suppress_errors`` drops the failures)."""
    map_fn = map if executor is None else executor.map
    rows, surviving = [], []
    for cut, samples in zip(cuts, map_fn(partial(_load_int16_audio, suppress_errors=suppress_errors), cuts)):
        if samples is not None:
            rows.append(torch.from_numpy(samples))
            surviving.append(cut)
    return rows, CutSet(surviving)


def _load_int16_audio(cut, suppress_errors: bool) -> np.ndarray | None:
    """Read the int16 samples of a mono cut as ``Recording.load_audio`` reads its float32 samples.

    A cut of a kind that this reader does not handle raises with or without ``suppress_errors`` (a configuration
    error); a failure to read supported audio drops the cut when ``suppress_errors`` is set.
    """
    recording = cut.recording if isinstance(cut, MonoCut) and cut.has_recording else None
    reason = _get_int16_unsupported_reason(cut, recording)
    if reason is not None:
        raise ValueError(f"int16 audio rows cannot read cut {cut.id}: {reason}.")
    try:
        source = recording.sources[0]
        offset, duration = cut.start, cut.duration
        # Recording.load_audio reads to the end of the recording when the cut covers nearly all of it
        read_duration = None if isclose(duration, recording.duration, abs_tol=1e-3) else duration
        with soundfile.SoundFile(BytesIO(source.source) if source.type == "memory" else source.source) as audio:
            if audio.subtype != "PCM_16":
                # soundfile would quantize other samples to int16, which no longer match the float32 path
                raise _UnsupportedInt16Audio(
                    f"int16 audio rows cannot read cut {cut.id}: its audio is {audio.subtype}, not 16-bit PCM."
                )
            if audio.samplerate != recording.sampling_rate or audio.channels != 1:
                # metadata that does not describe the file is a data error, not a failed read
                raise _UnsupportedInt16Audio(
                    f"int16 audio rows cannot read cut {cut.id}: the audio has {audio.channels} channel(s) at "
                    f"{audio.samplerate} Hz, the recording declares 1 channel at {recording.sampling_rate} Hz."
                )
            if offset > 0:
                audio.seek(compute_num_samples(offset, audio.samplerate))
            num_samples = -1 if read_duration is None else compute_num_samples(read_duration, audio.samplerate)
            samples = audio.read(frames=num_samples, dtype="int16", always_2d=True).T
        if read_duration is not None:
            available_duration = samples.shape[1] / recording.sampling_rate
            if available_duration < read_duration - get_audio_duration_mismatch_tolerance():
                raise ValueError(
                    f"int16 audio rows: cut {cut.id} requests {read_duration} s, the audio has {available_duration} s."
                )
        return assert_and_maybe_fix_num_samples(samples, offset=offset, duration=duration, recording=recording)[0]
    except _UnsupportedInt16Audio:
        raise
    except Exception as error:
        if not suppress_errors:
            raise
        logging.warning(f"Skipping cut {cut.id}: its int16 audio failed to load ({error}).")
        return None


def _get_int16_unsupported_reason(cut, recording) -> str | None:
    if recording is None:
        return f"a {type(cut).__name__} without a recording of its own"
    if len(recording.sources) != 1 or recording.num_channels != 1:
        return f"{len(recording.sources)} sources and {recording.num_channels} channels instead of one mono source"
    if recording.transforms:
        return "audio transforms (resampling, speed or volume perturbation, ...)"
    if recording.has_video:
        return "a recording with video"
    source = recording.sources[0]
    if source.type not in ("file", "memory"):
        return f"an audio source of type {source.type!r} (file and memory are supported)"
    if source.type == "file" and ".tar/" in str(source.source):
        return "a file inside a tar archive (Lhotse's .tar/ paths)"
    return None


class _UnsupportedInt16Audio(ValueError):
    """Audio that the int16 reader cannot read exactly: a configuration error, raised also when fault-tolerant."""
