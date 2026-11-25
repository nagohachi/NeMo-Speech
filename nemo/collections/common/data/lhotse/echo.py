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

"""Data echoing: reuse every example read from storage several times."""

import random
from typing import Any, Iterable

from lhotse.dataset.dataloading import resolve_seed
from lhotse.lazy import IteratorNode, LazyIteratorChain, attach_repeat_idx_to_id, resolve_iterator_source

__all__ = ["LazyEchoer"]


class LazyEchoer(IteratorNode):
    """Yield every item of ``iterator`` ``factor`` times, its copies spread at random over a buffer.

    Example echoing of Choi et al., "Faster Neural Network Training with Data Echoing" (2019):
    when reading the data bounds training throughput, every example that is read trains the model several times.
    The copies are inserted into a buffer of ``buffer_size`` items, from which items leave in random order,
    so they usually reach the sampler at different times. They can still share a mini-batch: a copy has the duration
    and token count of its original, so a bucketing sampler puts both into the same bucket.
    The copies share the in-memory data of the original (no extra storage reads or audio copies);
    the ``k``-th copy (``k >= 1``) gets the id suffix ``_repeat{k}``.
    Augmentations applied after sampling (SpecAugment on the GPU, for example) draw new randomness per copy.

    The echo buffer and its random order are not saved in a dataloader state, so the dataloader refuses
    ``echo_factor != 1`` together with ``use_stateful_dataloader=True``.

    Args:
        iterator: the source of examples.
        factor: how many times every example is yielded; 1 passes the source through.
        buffer_size: how many items the random reordering holds.
        seed: the seed of the reordering; ``"trng"`` and ``"randomized"`` resolve per worker as in Lhotse samplers.
            The RNG is created and seeded in ``__iter__``.
    """

    def __init__(self, iterator: Iterable, factor: int, buffer_size: int, seed: int | str = "trng") -> None:
        if factor < 1:
            raise ValueError(f"factor must be at least 1, got {factor}.")
        if buffer_size < 1:
            raise ValueError(f"buffer_size must be positive, got {buffer_size}.")
        self.source = resolve_iterator_source(iterator)
        self.factor = factor
        self.buffer_size = buffer_size
        self.seed = seed

    def __iter__(self):
        if self.factor == 1:
            yield from self.source
            return
        rng = random.Random(resolve_seed(self.seed))
        buffer: list[Any] = []
        for item in self.source:
            buffer.append(item)
            buffer.extend(attach_repeat_idx_to_id(item, k) for k in range(1, self.factor))
            while len(buffer) > self.buffer_size:
                yield _pop_random(buffer, rng)
        while buffer:
            yield _pop_random(buffer, rng)

    def __len__(self) -> int:
        return self.factor * len(self.source)

    def __add__(self, other):
        return LazyIteratorChain(self, other)


def _pop_random(buffer: list, rng: random.Random) -> Any:
    idx = rng.randrange(len(buffer))
    buffer[idx], buffer[-1] = buffer[-1], buffer[idx]
    return buffer.pop()
