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


"""``read_unpadded_audio_rows``: one waveform per loaded cut, with the cuts that loaded."""

import numpy as np
import pytest
import soundfile
import torch
from lhotse import AudioSource, CutSet, MonoCut, Recording
from lhotse.dataset import AudioSamples

from nemo.collections.common.data.lhotse.audio_loading import read_unpadded_audio_rows

SAMPLING_RATE = 16000


@pytest.mark.unit
def test_rows_keep_the_order_of_the_cuts_that_load_without_the_unreadable_ones(tmp_path):
    """Check that with fault-tolerant loading a cut whose file is missing is dropped from the rows and the cuts,
    and that every other row is the audio of the cut returned at its position."""
    cuts = CutSet(
        [
            Recording.from_file(_write_audio(tmp_path / "first.wav", num_samples=8000, seed=1)).to_cut(),
            _make_recording(tmp_path / "missing.wav", num_samples=8000).to_cut(),
            Recording.from_file(_write_audio(tmp_path / "last.wav", num_samples=4000, seed=2)).to_cut(),
        ]
    )
    rows, surviving = read_unpadded_audio_rows(cuts, AudioSamples(fault_tolerant=True))
    assert [cut.recording.id for cut in surviving] == ["first", "last"]
    assert [row.numel() for row in rows] == [8000, 4000]
    for row, cut in zip(rows, surviving):
        assert row.dtype == torch.float32
        assert torch.equal(row, torch.from_numpy(cut.load_audio()[0]))


@pytest.mark.unit
def test_rows_raise_for_an_unreadable_cut_without_fault_tolerance(tmp_path):
    """Check that strict loading raises for a cut whose file is missing instead of dropping it."""
    cuts = CutSet([_make_recording(tmp_path / "missing.wav", num_samples=8000).to_cut()])
    with pytest.raises(Exception):
        read_unpadded_audio_rows(cuts, AudioSamples(fault_tolerant=False))


@pytest.mark.unit
@pytest.mark.parametrize("mono_downmix", [True, None])
def test_rows_downmix_multichannel_audio(tmp_path, mono_downmix):
    """Check that a cut with two channels gives the mean of its channels, with downmixing on and, next to a mono
    cut, with Lhotse's default that downmixes unless every cut has several channels."""
    stereo_cut = _make_stereo_cut(tmp_path)
    mono_cut = Recording.from_file(_write_audio(tmp_path / "talk.wav", num_samples=8000)).to_cut()
    rows, _ = read_unpadded_audio_rows(CutSet([stereo_cut, mono_cut]), AudioSamples(mono_downmix=mono_downmix))
    torch.testing.assert_close(rows[0], torch.from_numpy(stereo_cut.load_audio()).mean(dim=0))
    assert torch.equal(rows[1], torch.from_numpy(mono_cut.load_audio()[0]))


@pytest.mark.unit
@pytest.mark.parametrize("mono_downmix", [False, None])
def test_rows_refuse_multichannel_rows(tmp_path, mono_downmix):
    """Check that the reader raises instead of returning multichannel rows: with downmixing off, and with Lhotse's
    default when every cut has several channels."""
    cuts = CutSet([_make_stereo_cut(tmp_path)])
    with pytest.raises(ValueError, match="mono_downmix=True"):
        read_unpadded_audio_rows(cuts, AudioSamples(mono_downmix=mono_downmix))


def _make_stereo_cut(tmp_path):
    """Write a stereo file of random samples and return a cut of both channels."""
    path = tmp_path / "stereo.wav"
    soundfile.write(path, np.random.default_rng(0).uniform(-0.5, 0.5, (8000, 2)), SAMPLING_RATE, subtype="FLOAT")
    return MonoCut("stereo", start=0, duration=0.5, channel=[0, 1], recording=Recording.from_file(path))


def _write_audio(path, num_samples, seed=0):
    """Write a mono file of random samples and return its path."""
    samples = np.random.default_rng(seed).uniform(-0.5, 0.5, num_samples)
    soundfile.write(path, samples, SAMPLING_RATE, subtype="FLOAT")
    return str(path)


def _make_recording(path, num_samples):
    """Return a mono recording of ``path``, which need not exist, that declares ``num_samples``."""
    return Recording(
        id=path.stem,
        sources=[AudioSource(type="file", channels=[0], source=str(path))],
        sampling_rate=SAMPLING_RATE,
        num_samples=num_samples,
        duration=num_samples / SAMPLING_RATE,
    )
