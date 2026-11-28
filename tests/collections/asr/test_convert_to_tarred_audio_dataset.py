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
    subtype = "PCM_16" if path.suffix in (".wav", ".flac") else None
    soundfile.write(path, rng.uniform(-0.5, 0.5, int(16000 * seconds)), 16000, subtype=subtype)
    return str(path)


def _get_squashed_base(audio_filepath: str) -> str:
    """Return the member name stem that the converter makes from a source path, without the extension."""
    return str(Path(audio_filepath).with_suffix("")).replace("/", "_").replace(".", "_")


def _make_builder(converter, force_codec, slice_with_offset):
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
    return builder


def _make_entries(talk_filepath: str, other_filepath: str, slice_with_offset: bool) -> list[dict]:
    if slice_with_offset:
        return [
            {"audio_filepath": talk_filepath, "offset": 0.5, "duration": 1.0, "text": "a"},
            {"audio_filepath": talk_filepath, "offset": 1.75, "duration": 1.25, "text": "b"},
        ]
    # the second entry of the same file becomes a -sub entry of the first member
    return [
        {"audio_filepath": talk_filepath, "duration": 3.0, "text": "a"},
        {"audio_filepath": talk_filepath, "duration": 3.0, "text": "b"},
        {"audio_filepath": other_filepath, "duration": 1.0, "text": "c"},
    ]


@pytest.mark.unit
@pytest.mark.parametrize(
    "source_ext,force_codec,slice_with_offset,member_ext",
    [
        ("wav", "flac", True, "flac"),  # cropped and transcoded
        ("wav", "wav", True, "wav"),  # cropped, same codec
        ("wav", "opus", True, "opus"),  # cropped into an ogg container named .opus
        ("ogg", None, True, "ogg"),  # cropped and re-encoded in the format of the source
        ("flac", "wav", False, "wav"),  # whole files, transcoded
        ("flac", "opus", False, "opus"),  # whole files, transcoded into an ogg container named .opus
        ("wav", None, False, "wav"),  # whole files stored as they are
    ],
)
def test_manifest_names_tar_members(tmp_path, converter, source_ext, force_codec, slice_with_offset, member_ext):
    """Check that the manifest entries, also with only_manifests, name the members written to the tar."""
    talk_filepath = _write_audio(tmp_path / f"talk.{source_ext}", 3.0)
    other_filepath = _write_audio(tmp_path / f"other.{source_ext}", 1.0)
    entries = _make_entries(talk_filepath, other_filepath, slice_with_offset)
    builder = _make_builder(converter, force_codec, slice_with_offset)

    new_entries = builder._create_shard([dict(e) for e in entries], str(tmp_path), shard_id=0)
    manifest_only_entries = builder._create_shard(
        [dict(e) for e in entries], str(tmp_path / "unused"), shard_id=0, only_manifests=True
    )

    talk_base, other_base = _get_squashed_base(talk_filepath), _get_squashed_base(other_filepath)
    if slice_with_offset:
        expected_names = [f"{talk_base}_0_5_1_0.{member_ext}", f"{talk_base}_1_75_1_25.{member_ext}"]
        expected_members = set(expected_names)
    else:
        expected_names = [f"{talk_base}.{member_ext}", f"{talk_base}-sub1.{member_ext}", f"{other_base}.{member_ext}"]
        expected_members = {f"{talk_base}.{member_ext}", f"{other_base}.{member_ext}"}
    with tarfile.open(tmp_path / "audio_0.tar") as tar:
        assert set(tar.getnames()) == expected_members
    assert [e["audio_filepath"] for e in new_entries] == expected_names
    assert [e["audio_filepath"] for e in manifest_only_entries] == expected_names
    assert {nemo_tar_audio_member_name(name) for name in expected_names} == expected_members


@pytest.mark.unit
def test_only_manifests_without_source_audio_raises_when_the_name_needs_its_format(tmp_path, converter):
    """Check that cropping without force_codec needs the source file to name the member."""
    missing_filepath = str(tmp_path / "missing.opus")
    entries = _make_entries(missing_filepath, missing_filepath, slice_with_offset=True)
    builder = _make_builder(converter, force_codec=None, slice_with_offset=True)
    with pytest.raises(FileNotFoundError, match="missing.opus"):
        builder._create_shard(entries, str(tmp_path), shard_id=0, only_manifests=True)


@pytest.mark.unit
def test_only_manifests_without_source_audio_names_forced_codec_members(tmp_path, converter):
    """Check that with force_codec the member names do not need the source file."""
    missing_filepath = str(tmp_path / "missing.wav")
    entries = _make_entries(missing_filepath, missing_filepath, slice_with_offset=True)
    builder = _make_builder(converter, force_codec="flac", slice_with_offset=True)
    manifest_only_entries = builder._create_shard(entries, str(tmp_path), shard_id=0, only_manifests=True)
    missing_base = _get_squashed_base(missing_filepath)
    assert [e["audio_filepath"] for e in manifest_only_entries] == [
        f"{missing_base}_0_5_1_0.flac",
        f"{missing_base}_1_75_1_25.flac",
    ]


@pytest.mark.unit
def test_only_manifests_reads_the_format_of_a_relative_path_from_the_manifest_folder(tmp_path, converter, monkeypatch):
    """Check that only_manifests finds the source of a path relative to the manifest folder to name its member."""
    manifest_folder = tmp_path / "data"
    manifest_folder.mkdir()
    _write_audio(manifest_folder / "talk.OGG", 3.0)
    monkeypatch.chdir(tmp_path)  # the relative path does not exist from the working directory
    entries = _make_entries("talk.OGG", "talk.OGG", slice_with_offset=True)
    builder = _make_builder(converter, force_codec=None, slice_with_offset=True)
    manifest_only_entries = builder._create_shard(
        entries, str(tmp_path), shard_id=0, manifest_folder=str(manifest_folder), only_manifests=True
    )
    assert [e["audio_filepath"] for e in manifest_only_entries] == ["talk_0_5_1_0.ogg", "talk_1_75_1_25.ogg"]
