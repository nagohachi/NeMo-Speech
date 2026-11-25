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

"""The manifest of scripts/speech_recognition/convert_to_tarred_audio_dataset.py names the members of its tars."""

import importlib
import sys
import tarfile
from pathlib import Path

import numpy as np
import pytest
import soundfile

from nemo.collections.common.data.lhotse.nemo_tar_routing import nemo_tar_audio_member_name

SCRIPTS = Path(__file__).resolve().parents[3] / "scripts" / "speech_recognition"


@pytest.fixture(scope="module", name="converter")
def load_converter():
    sys.path.insert(0, str(SCRIPTS))
    try:
        yield importlib.import_module("convert_to_tarred_audio_dataset")
    finally:
        sys.path.remove(str(SCRIPTS))


def _write_audio(path: Path, seconds: float) -> str:
    rng = np.random.default_rng(0)
    soundfile.write(path, rng.uniform(-0.5, 0.5, int(16000 * seconds)), 16000, subtype="PCM_16")
    return str(path)


@pytest.mark.unit
@pytest.mark.parametrize(
    "source_ext,force_codec,slice_with_offset",
    [
        ("wav", "flac", True),  # cropped and transcoded
        ("wav", "wav", True),  # cropped, same codec
        ("flac", "wav", False),  # whole files, transcoded
        ("wav", None, False),  # whole files stored as they are
    ],
)
def test_manifest_names_tar_members(tmp_path, converter, source_ext, force_codec, slice_with_offset):
    audio = _write_audio(tmp_path / f"talk.{source_ext}", 3.0)
    other = _write_audio(tmp_path / f"other.{source_ext}", 1.0)
    if slice_with_offset:
        entries = [
            {"audio_filepath": audio, "offset": 0.5, "duration": 1.0, "text": "a"},
            {"audio_filepath": audio, "offset": 1.75, "duration": 1.25, "text": "b"},
        ]
    else:
        # the second entry of the same file becomes a -sub entry of the first member
        entries = [
            {"audio_filepath": audio, "duration": 3.0, "text": "a"},
            {"audio_filepath": audio, "duration": 3.0, "text": "b"},
            {"audio_filepath": other, "duration": 1.0, "text": "c"},
        ]
    config = converter.ASRTarredDatasetConfig(
        num_shards=1,
        max_duration=40.0,
        min_duration=0.1,
        sort_in_shards=False,
        slice_with_offset=slice_with_offset,
        force_codec=force_codec,
    )
    builder = converter.ASRTarredDatasetBuilder()
    builder.configure(config)

    new_entries = builder._create_shard([dict(e) for e in entries], str(tmp_path), shard_id=0)
    manifest_only = builder._create_shard(
        [dict(e) for e in entries], str(tmp_path / "unused"), shard_id=0, only_manifests=True
    )

    with tarfile.open(tmp_path / "audio_0.tar") as tar:
        members = set(tar.getnames())
    for entry in new_entries:
        assert nemo_tar_audio_member_name(entry["audio_filepath"]) in members, (entry, members)
    expected_ext = f".{force_codec or source_ext}"
    assert all(name.endswith(expected_ext) for name in members)
    assert [e["audio_filepath"] for e in manifest_only] == [e["audio_filepath"] for e in new_entries]
