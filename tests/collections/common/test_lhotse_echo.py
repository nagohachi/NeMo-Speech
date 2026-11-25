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

from collections import Counter

import pytest
import torch
from lhotse import CutSet
from lhotse.testing.dummies import DummyManifest
from omegaconf import OmegaConf

from nemo.collections.common.data.lhotse import get_lhotse_dataloader_from_config
from nemo.collections.common.data.lhotse.echo import LazyEchoer


class _Identity(torch.utils.data.Dataset):
    def __getitem__(self, cuts):
        return cuts


@pytest.fixture(name="cuts")
def make_cuts():
    return DummyManifest(CutSet, begin_id=0, end_id=20, with_data=True)


@pytest.mark.unit
def test_echo_factor_one_passes_through(cuts):
    assert [c.id for c in LazyEchoer(cuts, factor=1, buffer_size=4, seed=0)] == [c.id for c in cuts]


@pytest.mark.unit
def test_echo_yields_every_example_factor_times(cuts):
    echoed = list(LazyEchoer(cuts, factor=3, buffer_size=8, seed=0))
    assert len(echoed) == 3 * len(cuts) == len(LazyEchoer(cuts, factor=3, buffer_size=8, seed=0))
    counts = Counter(c.id.split("_repeat")[0] for c in echoed)
    assert counts == {c.id: 3 for c in cuts}
    ids = {c.id for c in echoed}
    assert all(f"{c.id}_repeat{k}" in ids for c in cuts for k in (1, 2))
    # the copies share the data of the original: no extra read or copy of the audio
    original = {c.id: c for c in cuts}
    for copy in echoed:
        assert copy.recording is original[copy.id.split("_repeat")[0]].recording


@pytest.mark.unit
def test_echo_order_is_seeded_and_mixed(cuts):
    """Check that the order depends only on the seed and that copies leave the buffer apart from their originals."""
    echoed_ids = [c.id for c in LazyEchoer(cuts, factor=2, buffer_size=10, seed=1)]
    assert echoed_ids == [c.id for c in LazyEchoer(cuts, factor=2, buffer_size=10, seed=1)]
    assert echoed_ids != [c.id for c in LazyEchoer(cuts, factor=2, buffer_size=10, seed=2)]
    position = {cut_id: i for i, cut_id in enumerate(echoed_ids)}
    distances = [abs(position[f"{c.id}_repeat1"] - position[c.id]) for c in cuts]
    assert max(distances) > 1


@pytest.mark.unit
def test_echo_order_changes_between_iterations(cuts):
    """Check that a second pass over the same echoer has another order, reproduced by a new echoer."""
    echoer = LazyEchoer(cuts, factor=2, buffer_size=10, seed=1)
    first_pass_ids = [c.id for c in echoer]
    second_pass_ids = [c.id for c in echoer]
    assert sorted(first_pass_ids) == sorted(second_pass_ids)
    assert first_pass_ids != second_pass_ids
    replayed_echoer = LazyEchoer(cuts, factor=2, buffer_size=10, seed=1)
    assert [[c.id for c in replayed_echoer] for _ in range(2)] == [first_pass_ids, second_pass_ids]


@pytest.mark.unit
@pytest.mark.parametrize("factor,buffer_size", [(0, 10), (2, 0)])
def test_echo_rejects_invalid_arguments(cuts, factor, buffer_size):
    with pytest.raises(ValueError):
        LazyEchoer(cuts, factor=factor, buffer_size=buffer_size)


@pytest.mark.unit
def test_dataloader_echo_factor(tmp_path, cuts):
    config = _make_config(tmp_path, cuts, echo_factor=2, echo_buffer_size=6)
    dl = get_lhotse_dataloader_from_config(config=config, global_rank=0, world_size=1, dataset=_Identity())
    seen = [c.id for batch in dl for c in batch]
    assert Counter(i.split("_repeat")[0] for i in seen) == {c.id: 2 for c in cuts}


@pytest.mark.unit
@pytest.mark.parametrize("echo_factor", [0, -1])
def test_dataloader_rejects_echo_factor_below_one(tmp_path, cuts, echo_factor):
    config = _make_config(tmp_path, cuts, echo_factor=echo_factor)
    with pytest.raises(ValueError, match="factor must be at least 1"):
        get_lhotse_dataloader_from_config(config=config, global_rank=0, world_size=1, dataset=_Identity())


@pytest.mark.unit
def test_dataloader_splits_echoed_copies_across_ranks(tmp_path, cuts):
    """Check that two ranks with an integer shard_seed together yield every original and copy exactly once."""
    config = _make_config(tmp_path, cuts, echo_factor=2, echo_buffer_size=6)
    seen_ids = []
    for rank in (0, 1):
        dl = get_lhotse_dataloader_from_config(config=config, global_rank=rank, world_size=2, dataset=_Identity())
        seen_ids.extend(c.id for batch in dl for c in batch)
    assert sorted(seen_ids) == sorted([c.id for c in cuts] + [f"{c.id}_repeat1" for c in cuts])


@pytest.mark.unit
def test_dataloader_rejects_echo_buffer_size_below_one_without_echoing(tmp_path, cuts):
    config = _make_config(tmp_path, cuts, echo_factor=1, echo_buffer_size=0)
    with pytest.raises(ValueError, match="echo_buffer_size must be positive"):
        get_lhotse_dataloader_from_config(config=config, global_rank=0, world_size=1, dataset=_Identity())


@pytest.mark.unit
def test_dataloader_rejects_echo_with_stateful_dataloader(tmp_path, cuts):
    config = _make_config(tmp_path, cuts, echo_factor=2, use_stateful_dataloader=True)
    with pytest.raises(ValueError, match="use_stateful_dataloader"):
        get_lhotse_dataloader_from_config(config=config, global_rank=0, world_size=1, dataset=_Identity())


def _make_config(tmp_path, cuts, **options):
    """Make a dataloader config that reads ``cuts`` from a manifest with their audio in files, plus ``options``."""
    path = tmp_path / "cuts.jsonl.gz"
    for c in cuts:  # keep only the audio, which save_audios writes to files
        c.features = None
        c.custom = None
        c.supervisions[0].custom = None
    cuts.save_audios(tmp_path / "audio").to_file(path)
    return OmegaConf.create(
        {
            "cuts_path": str(path),
            "sample_rate": 16000,
            "shuffle": True,
            "use_lhotse": True,
            "num_workers": 0,
            "batch_size": 4,
            "seed": 0,
            "shard_seed": 0,
            "force_finite": True,
            **options,
        }
    )
