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

"""The int16 reader of ``read_unpadded_audio_rows`` against the float32 samples that Lhotse reads."""

import numpy as np
import pytest
import soundfile
import torch
from lhotse import AudioSource, CutSet, MonoCut, Recording
from lhotse.dataset import AudioSamples

from nemo.collections.asr.modules.audio_preprocessing import INT16_AUDIO_SCALE
from nemo.collections.common.data.lhotse.audio_loading import read_unpadded_audio_rows

SAMPLING_RATE = 16000


@pytest.mark.unit
@pytest.mark.parametrize("source_type", ["file", "memory"])
def test_int16_rows_scale_to_the_float_rows_of_whole_and_partial_cuts(tmp_path, source_type):
    """Check that int16 rows divided by INT16_AUDIO_SCALE are Lhotse's float32 rows, for a whole cut and for cuts
    with an offset and a duration, read from a file or from memory."""
    recording = Recording.from_file(_write_audio(tmp_path / "talk.wav", num_samples=24000))
    if source_type == "memory":
        recording = recording.move_to_memory()
    whole_cut = recording.to_cut()
    cuts = CutSet([whole_cut, whole_cut.truncate(offset=0.37, duration=0.5), whole_cut.truncate(offset=1.0)])

    float_rows, _ = read_unpadded_audio_rows(cuts, AudioSamples())
    int16_rows, int16_cuts = read_unpadded_audio_rows(cuts, AudioSamples(), dtype=torch.int16)

    assert [cut.id for cut in int16_cuts] == [cut.id for cut in cuts]
    assert [row.numel() for row in int16_rows] == [24000, 8000, 8000]
    for int16_row, float_row in zip(int16_rows, float_rows):
        assert int16_row.dtype == torch.int16
        assert torch.equal(int16_row.to(torch.float32) / INT16_AUDIO_SCALE, float_row)


@pytest.mark.unit
def test_int16_rows_pad_a_slightly_short_file_as_lhotse_does(tmp_path):
    """Check that a file a few samples shorter than its recording declares gets Lhotse's length correction."""
    path = _write_audio(tmp_path / "short.wav", num_samples=15998)
    cuts = CutSet([_make_recording(path, num_samples=16000).to_cut()])

    float_rows, _ = read_unpadded_audio_rows(cuts, AudioSamples())
    int16_rows, _ = read_unpadded_audio_rows(cuts, AudioSamples(), dtype=torch.int16)

    assert int16_rows[0].numel() == float_rows[0].numel() == 16000
    assert torch.equal(int16_rows[0].to(torch.float32) / INT16_AUDIO_SCALE, float_rows[0])


@pytest.mark.unit
def test_int16_rows_raise_for_a_file_shorter_than_the_cut(tmp_path):
    """Check that a cut that requests more audio than the file has, beyond Lhotse's duration mismatch tolerance,
    raises instead of returning padded samples."""
    path = _write_audio(tmp_path / "short.wav", num_samples=16000)
    cut = _make_recording(path, num_samples=64000).to_cut().truncate(offset=0.5, duration=2.0)
    with pytest.raises(ValueError, match="requests 2.0 s"):
        read_unpadded_audio_rows(CutSet([cut]), AudioSamples(), dtype=torch.int16)


@pytest.mark.unit
@pytest.mark.parametrize("fault_tolerant", [False, True])
@pytest.mark.parametrize("subtype", ["PCM_24", "FLOAT"])
def test_int16_rows_refuse_audio_that_is_not_16_bit_pcm(tmp_path, subtype, fault_tolerant):
    """Check that the reader raises for samples that int16 would quantize, also when it drops unreadable cuts."""
    cuts = CutSet([Recording.from_file(_write_audio(tmp_path / "talk.wav", 8000, subtype=subtype)).to_cut()])
    with pytest.raises(ValueError, match=f"its audio is {subtype}, not 16-bit PCM"):
        read_unpadded_audio_rows(cuts, AudioSamples(fault_tolerant=fault_tolerant), dtype=torch.int16)


@pytest.mark.unit
@pytest.mark.parametrize("fault_tolerant", [False, True])
def test_int16_rows_refuse_a_file_that_differs_from_its_recording(tmp_path, fault_tolerant):
    """Check that a sampling rate that differs from the one the recording declares raises, also when the reader
    drops unreadable cuts."""
    path = _write_audio(tmp_path / "talk.wav", num_samples=8000)
    cut = _make_recording(path, num_samples=4000, sampling_rate=8000).to_cut()
    with pytest.raises(ValueError, match=r"1 channel\(s\) at 16000 Hz, the recording declares 1 channel at 8000 Hz"):
        read_unpadded_audio_rows(CutSet([cut]), AudioSamples(fault_tolerant=fault_tolerant), dtype=torch.int16)


@pytest.mark.unit
@pytest.mark.parametrize("fault_tolerant", [False, True])
def test_int16_rows_refuse_cuts_they_cannot_read_exactly(tmp_path, fault_tolerant):
    """Check that a stereo recording, a file inside a tar archive and cuts without a recording of their own raise,
    also when the reader drops unreadable cuts."""
    stereo_path = tmp_path / "stereo.wav"
    soundfile.write(stereo_path, np.zeros((8000, 2), dtype=np.int16), SAMPLING_RATE, subtype="PCM_16")
    mono_cut = Recording.from_file(_write_audio(tmp_path / "talk.wav", num_samples=8000)).to_cut()
    tar_member = _make_recording(f"{tmp_path}/shard.tar/talk.wav", num_samples=8000).to_cut()
    stereo_recording = Recording.from_file(stereo_path)
    load_audio = AudioSamples(fault_tolerant=fault_tolerant)
    for cut, message in (
        (
            MonoCut("stereo", start=0, duration=0.5, channel=0, recording=stereo_recording),
            "instead of one mono source",
        ),
        (stereo_recording.to_cut(), "a MultiCut without a recording of its own"),
        (tar_member, "a file inside a tar archive"),
        (mono_cut.pad(duration=1.0), "a MixedCut without a recording of its own"),
    ):
        with pytest.raises(ValueError, match=message):
            read_unpadded_audio_rows(CutSet([cut]), load_audio, dtype=torch.int16)


def _write_audio(path, num_samples, subtype="PCM_16"):
    """Write a mono file of random samples and return its path."""
    samples = np.random.default_rng(0).uniform(-0.5, 0.5, num_samples)
    soundfile.write(path, samples, SAMPLING_RATE, subtype=subtype)
    return str(path)


def _make_recording(path, num_samples, sampling_rate=SAMPLING_RATE):
    """Return a mono recording of ``path`` that declares ``num_samples`` at ``sampling_rate``."""
    return Recording(
        id="talk",
        sources=[AudioSource(type="file", channels=[0], source=str(path))],
        sampling_rate=sampling_rate,
        num_samples=num_samples,
        duration=num_samples / sampling_rate,
    )
