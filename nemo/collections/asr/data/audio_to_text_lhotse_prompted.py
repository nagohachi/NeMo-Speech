# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
import dataclasses
import os
from dataclasses import dataclass
from typing import Optional, Sequence, Union

import torch.utils.data
from lhotse import CutSet
from lhotse.cut import MixedCut
from lhotse.dataset.collation import collate_vectors

from nemo.collections.asr.data.audio_to_text_lhotse import _make_audio_samples
from nemo.collections.common.data import apply_prompt_format_fn
from nemo.collections.common.data.lhotse.audio_loading import LhotseAudioLoadingDatasetMixin, read_unpadded_audio_rows
from nemo.collections.common.prompts import PromptFormatter
from nemo.collections.common.tokenizers import TokenizerSpec


@dataclass
class PromptedAudioToTextMiniBatch:
    audio: torch.Tensor
    audio_lens: torch.Tensor
    transcript: torch.Tensor
    transcript_lens: torch.Tensor
    prompt: torch.Tensor
    prompt_lens: torch.Tensor
    prompted_transcript: torch.Tensor
    prompted_transcript_lens: torch.Tensor
    cuts: Optional[CutSet] = None

    def get_decoder_inputs_outputs(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns the inputs and outputs of transformer decoder for training.
        The input is ``prompted_transcript`` (minus last token),
        and the output is ``prompted_transcript`` (minus first token).
        """
        return self.prompted_transcript[:, :-1], self.prompted_transcript[:, 1:]


@dataclass
class PackedPromptedAudioToTextMiniBatch:
    """A training mini-batch of a prompted AED model without inter-utterance padding.

    The waveforms are concatenated into one 1D tensor and the teacher-forcing sequences into token-flat tensors,
    with explicit boundaries instead of padding.
    The decoder input of an utterance is its prompted transcript without the last token,
    the target the prompted transcript without the first token, so both have ``prompted_transcript_lens - 1`` tokens.

    Args:
        audio: concatenated waveforms, shape ``(sum(audio_lens),)``: float32 samples in [-1, 1),
            or int16 samples (the float32 samples times 32768) from a dataset with ``int16_audio``.
        audio_lens: samples per utterance, shape ``(B,)``.
        audio_cu_seqlens: cumulative samples, shape ``(B + 1,)``.
        decoder_input_ids: concatenated decoder inputs, shape ``(sum(prompted_transcript_lens - 1),)``.
        decoder_targets: concatenated targets, same shape as ``decoder_input_ids``.
        prompted_transcript_lens: prompt plus transcript tokens per utterance, shape ``(B,)``.
        prompt_lens: prompt tokens per utterance, shape ``(B,)``.
        transcript: the answer tokens of the prompt format, padded, shape ``(B, U)``,
            as ``PromptedAudioToTextMiniBatch.transcript``; the targets of an auxiliary CTC loss.
        transcript_lens: transcript tokens per utterance, shape ``(B,)``.
        max_decoder_len: the longest decoder input of the batch, ``max(prompted_transcript_lens) - 1``,
            a host integer, so that a consumer needs no device read for it.
        cuts: the utterances of the batch, in order.
        host_audio_lens: ``audio_lens`` as a tuple of ints, which stays on the host when the batch moves to the GPU,
            so that the packed frontend sizes its computation without a device read.
    """

    audio: torch.Tensor
    audio_lens: torch.Tensor
    audio_cu_seqlens: torch.Tensor
    decoder_input_ids: torch.Tensor
    decoder_targets: torch.Tensor
    prompted_transcript_lens: torch.Tensor
    prompt_lens: torch.Tensor
    transcript: torch.Tensor
    transcript_lens: torch.Tensor
    max_decoder_len: int
    cuts: Optional[CutSet] = None
    host_audio_lens: Optional[tuple[int, ...]] = None

    def pin_memory(self) -> "PackedPromptedAudioToTextMiniBatch":
        """Return a copy with pinned tensors; see ``_pin_tensor_fields``."""
        return _pin_tensor_fields(self)


class PromptedAudioToTextLhotseDataset(LhotseAudioLoadingDatasetMixin, torch.utils.data.Dataset):
    """
    This dataset is based on :class:`~nemo.collections.asr.data.audio_to_text_lhotse.LhotseSpeechToTextBpeDataset`.
    It is a Lhotse-style dataset that converts a mini-batch of Cuts into tensors.
    The main difference from ``LhotseSpeechToTextBpeDataset`` is that we introduce
    a special prompt format for multitask encoder-decoder models.

    To perform the prompt formatting, we accept a ``prompt_format_fn``.
    It's expected to accept:
    * a ``CutSet`` which it will internally iterate over for utterances, and
    * a ``TokenizerWrapper`` object that will be internally used to tokenize the utterances

    Tokenized utterances will be extended with special prompt tokens according to ``prompt_format_fn`` logic.
    We support cuts with multiple supervision segments -- their tokenized texts will be concatenated before we add the prompt tokens.
    This is useful, for example, in code-switched scenarios where each segment is spoken in a different language.

    Chunking:
    If `enable_chunking` is True, each audio sample is split into optimally sized chunks
    (see `find_optimal_chunk_size` and `chunk_waveform`). This is useful for long audio inputs,
    allowing the model to process them in manageable segments.

    Sequence packing:
    With `pack_sequences` the dataset returns a ``PackedPromptedAudioToTextMiniBatch``: the waveforms of all
    utterances in one row, without padding. With `int16_audio` as well, the waveforms stay 16-bit integers from the
    decoder to the GPU (half the bytes to collate, pin and copy); the model's preprocessor divides them by 32768,
    which gives the float32 samples of the default path bit for bit. It reads single-source mono recordings from
    files or from memory (tarred datasets) and raises on other cuts.

    NOTE:
    If the environment variable `USE_AIS_GET_BATCH` is set to `true` (case-insensitive),
    then batch audio loading from AIStore will be enabled for this dataset. This will use the
    AISBatchLoader to load the audio from AIStore. This can improve data loading efficiency in some setups.
    """

    def __init__(
        self,
        tokenizer: TokenizerSpec,
        prompt: PromptFormatter,
        enable_chunking: bool = False,
        pack_sequences: bool = False,
        int16_audio: bool = False,
    ):
        super().__init__()
        if pack_sequences and enable_chunking:
            raise ValueError("pack_sequences and enable_chunking cannot be combined.")
        if int16_audio and not pack_sequences:
            raise ValueError("int16_audio needs pack_sequences: only the sequence-packed batches carry int16 audio.")
        self.pack_sequences = pack_sequences
        self.int16_audio = int16_audio
        self.tokenizer = tokenizer
        self.use_ais_get_batch = os.environ.get("USE_AIS_GET_BATCH", "False").lower() == "true"
        self.ais_force_individual = os.environ.get("USE_AIS_INDIVIDUAL_GETS", "False").lower() == "true"

        self.load_audio = _make_audio_samples(
            use_batch_loader=self.use_ais_get_batch,
            ais_force_individual=self.ais_force_individual,
        )

        self.padding_value = self.tokenizer.pad_id
        self.prompt = prompt
        self.enable_chunking = enable_chunking

    def __getitem__(self, cuts: CutSet) -> PromptedAudioToTextMiniBatch | PackedPromptedAudioToTextMiniBatch:
        if self.pack_sequences:
            return self._get_packed_batch(cuts)
        # Load the audio's from AIS and add them to the CutSet
        audio, audio_lens, cuts = self.load_audio_with_cuts(cuts)

        # Will work if batch_size is set to 1.
        if self.enable_chunking:
            # If dynamic chunking is enabled, split each audio sample into chunks.
            new_audio = []
            new_audio_lens = []
            for i in range(audio.shape[0]):
                waveform = audio[i, : audio_lens[i]]
                # Split the waveform into chunks and get their lengths.
                chunks, chunk_lens = self._chunk_waveform(waveform)
                new_audio.extend(chunks)
                new_audio_lens.extend(chunk_lens)
            # Stack all chunks into a batch.
            audio = torch.stack(new_audio)
            audio_lens = torch.tensor(new_audio_lens, dtype=torch.long)
            # Adding this to allow gathering results of the same audio from different batches
            if cuts[0].start != 0:
                cuts[0].id = cuts[0].id + '_cut_segmented'
        prompts_with_answers, prompts, answers = self._get_prompted_token_ids(cuts)

        transcript, transcript_lens = self._collate_tokens(answers)
        prompts_with_answers, prompts_with_answers_lens = self._collate_tokens(prompts_with_answers)
        prompts, prompt_lens = self._collate_tokens(prompts)

        return PromptedAudioToTextMiniBatch(
            audio=audio,
            audio_lens=audio_lens,
            transcript=transcript,
            transcript_lens=transcript_lens,
            prompt=prompts,
            prompt_lens=prompt_lens,
            prompted_transcript=prompts_with_answers,
            prompted_transcript_lens=prompts_with_answers_lens,
            cuts=_drop_in_memory_data(cuts),
        )

    def _get_packed_batch(self, cuts: CutSet) -> PackedPromptedAudioToTextMiniBatch:
        num_cuts = len(cuts)
        dtype = torch.int16 if self.int16_audio else torch.float32
        rows, cuts = read_unpadded_audio_rows(cuts, self.load_audio, dtype=dtype)
        if not rows:
            raise ValueError(
                f"The audio of none of the {num_cuts} cuts of the batch could be loaded: "
                "a sequence-packed batch needs at least one utterance."
            )
        host_audio_lens = tuple(row.numel() for row in rows)
        audio_lens = torch.tensor(host_audio_lens, dtype=torch.long)
        audio_cu_seqlens = torch.zeros(len(rows) + 1, dtype=torch.long)
        torch.cumsum(audio_lens, dim=0, out=audio_cu_seqlens[1:])
        audio = torch.cat(rows)

        prompts_with_answers, prompts, answers = self._get_prompted_token_ids(cuts)
        prompted_ids = [torch.as_tensor(ids, dtype=torch.long) for ids in prompts_with_answers]
        prompt_lens = [len(ids) for ids in prompts]
        answer_tensors = [torch.as_tensor(ids, dtype=torch.long) for ids in answers]
        if any(seq.numel() < 2 for seq in prompted_ids):
            raise ValueError("Every prompted transcript needs at least two tokens for teacher forcing.")
        transcript, transcript_lens = self._collate_tokens(answer_tensors)
        return PackedPromptedAudioToTextMiniBatch(
            audio=audio,
            audio_lens=audio_lens,
            audio_cu_seqlens=audio_cu_seqlens,
            decoder_input_ids=torch.cat([seq[:-1] for seq in prompted_ids]),
            decoder_targets=torch.cat([seq[1:] for seq in prompted_ids]),
            prompted_transcript_lens=torch.tensor([seq.numel() for seq in prompted_ids], dtype=torch.long),
            prompt_lens=torch.tensor(prompt_lens, dtype=torch.long),
            transcript=transcript,
            transcript_lens=transcript_lens,
            max_decoder_len=max(seq.numel() - 1 for seq in prompted_ids),
            cuts=_drop_token_ids(_drop_in_memory_data(cuts)),
            host_audio_lens=host_audio_lens,
        )

    def _get_prompted_token_ids(self, cuts: CutSet) -> tuple[Sequence, Sequence, Sequence]:
        """Return the prompted transcript, prompt and answer token ids of every cut, in the order of ``cuts``.

        The ids that the sampler attached to the cuts (tokenization and prompt formatting done before sampling) are
        used when every cut has them; otherwise the prompt format is applied here.
        """
        attrs = ("input_ids", "context_ids", "answer_ids")
        if all(hasattr(c, a) for c in cuts for a in attrs):
            prompts_with_answers, prompts, answers = zip(*((c.input_ids, c.context_ids, c.answer_ids) for c in cuts))
            return prompts_with_answers, prompts, answers
        formatted = [apply_prompt_format_fn(cut, self.prompt) for cut in cuts]
        return (
            [ex["input_ids"] for ex in formatted],
            [ex["context_ids"] for ex in formatted],
            [ex["answer_ids"] for ex in formatted],
        )

    def _collate_tokens(self, tokens: list[Union[list[int], torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = [torch.as_tensor(t) for t in tokens]
        token_lens = torch.tensor([t.size(0) for t in tokens], dtype=torch.long)
        tokens = collate_vectors(tokens, padding_value=self.padding_value)
        return tokens, token_lens

    def _find_optimal_chunk_size(
        self, total_len: int, min_sec: int = 30, max_sec: int = 40, sample_rate: int = 16000, overlap_sec: float = 1.0
    ) -> int:
        """
        Find the optimal chunk size for audio processing that minimizes paddings to the last chunk.

        Args:
            total_len (int): Total length of the audio waveform in samples
            min_sec (int, optional): Minimum chunk size in seconds. Defaults to 30.
            max_sec (int, optional): Maximum chunk size in seconds. Defaults to 40.
            sample_rate (int, optional): Audio sample rate in Hz. Defaults to 16000.
            overlap_sec (float, optional): Overlap duration between consecutive chunks in seconds.
                                         Defaults to 1.0.

        Returns:
            int: Optimal chunk size in samples that maximizes the last chunk length
        """
        best_chunk_size = min_sec * sample_rate
        best_last_chunk_len = 0
        if total_len < max_sec * sample_rate:
            return total_len
        # Try each possible chunk duration in the range
        for sec in range(min_sec, max_sec + 1):
            chunk_size = sec * sample_rate
            overlap_size = int(overlap_sec * sample_rate)
            step_size = chunk_size - overlap_size

            if step_size <= 0:  # Invalid overlap
                continue
            if chunk_size > total_len:
                continue

            # Calculate how many chunks we'd need and the last chunk's length
            n_chunks = (total_len + step_size - 1) // step_size
            last_chunk_len = total_len - step_size * (n_chunks - 1)

            if last_chunk_len > best_last_chunk_len:
                best_last_chunk_len = last_chunk_len
                best_chunk_size = chunk_size

        return best_chunk_size

    def _chunk_waveform(
        self, waveform: torch.Tensor, chunk_size: int = None, overlap_sec: float = 1.0, sample_rate: int = 16000
    ) -> tuple[list[torch.Tensor], list[int]]:
        """
        Split a waveform tensor into overlapping chunks.

        Args:
            waveform (torch.Tensor): Input audio waveform tensor of shape (time_samples,)
            chunk_size (int, optional): Size of each chunk in samples. If None, automatically
                                       determines optimal chunk size using find_optimal_chunk_size().
                                       Defaults to None.
            sample_rate (int, optional): Audio sample rate in Hz. Defaults to 16000.
            overlap_sec (float, optional): Overlap duration between consecutive chunks in seconds.
                                          Used to calculate step size. Defaults to 2.

        Returns:
            tuple[list[torch.Tensor], list[int]]: A tuple containing:
                - List of chunk tensors, each of shape (chunk_size,)
                - List of original lengths for each chunk before padding (useful for masking
                  padded regions during processing.
        """
        # If chunk_size is None, find the optimal chunk size for this waveform
        total_len = waveform.shape[0]
        if chunk_size is None:
            chunk_size = self._find_optimal_chunk_size(total_len, overlap_sec=overlap_sec)
        if chunk_size >= total_len:
            return [waveform], [total_len]
        overlap_size = int(overlap_sec * sample_rate)
        step_size = chunk_size - overlap_size
        chunks = []
        chunk_lens = []
        start = 0
        while start + overlap_size < total_len:
            end = min(start + chunk_size, total_len)
            chunk = waveform[start:end]
            length = chunk.shape[0]
            if length < chunk_size:
                pad = torch.zeros(chunk_size - length, dtype=chunk.dtype, device=chunk.device)
                chunk = torch.cat([chunk, pad], dim=0)
            chunks.append(chunk)
            chunk_lens.append(length)
            start += step_size

        return chunks, chunk_lens


class ProbablyIncorrectLanguageKeyError(RuntimeError):
    pass


def _pin_tensor_fields(batch):
    """Copy a mini-batch dataclass with every tensor field in page-locked memory.

    ``DataLoader(pin_memory=True)`` pins a batch through its ``pin_memory`` method only;
    without one the dataclass is passed on as it is, and the host-to-device copy of its audio
    is a synchronous copy from pageable memory that stalls the training loop.
    """
    pinned_fields = {
        f.name: value.pin_memory()
        for f in dataclasses.fields(batch)
        if isinstance(value := getattr(batch, f.name), torch.Tensor)
    }
    return dataclasses.replace(batch, **pinned_fields)


_TOKEN_ID_FIELDS = ("input_ids", "context_ids", "answer_ids", "mask")


def _drop_token_ids(cuts: CutSet) -> CutSet:
    """Remove the token tensors of pretokenized cuts, which the mini-batch carries collated.

    A batch leaves the dataloader worker by pickling, and every tensor in it crosses the process boundary on a
    file descriptor of its own: three per cut would exhaust the default limit of 1024 descriptors on batches of
    a few hundred cuts. ``cuts`` are copies (see ``_drop_in_memory_data``), so the cuts of the sampler keep theirs.
    """
    for cut in cuts:
        custom_fields = getattr(cut, "custom", None)
        if custom_fields and any(key in custom_fields for key in _TOKEN_ID_FIELDS):
            cut.custom = {k: v for k, v in custom_fields.items() if k not in _TOKEN_ID_FIELDS}
        for key in _TOKEN_ID_FIELDS:
            if key in cut.__dict__:
                delattr(cut, key)
    return cuts


def _drop_in_memory_data(
    cuts: CutSet,
    _fields=frozenset(MixedCut.__dataclass_fields__.keys()),
) -> CutSet:
    """Workaround for an edge case in cuts.drop_in_memory_data() on MixedCut with Lhotse<1.29.0"""
    ans = []
    for c in cuts:
        # Not a mixed cut or a mixed cut that wasn't assigned any extra attributes.
        if not isinstance(c, MixedCut) or _fields.issuperset(c.__dict__.keys()):
            ans.append(c.drop_in_memory_data())
        else:
            extra_attrs = {k: v for k, v in c.__dict__.items() if k not in _fields}
            for k in extra_attrs:
                delattr(c, k)
            ans.append(c.drop_in_memory_data())
            for k, v in extra_attrs.items():
                setattr(ans[-1], k, v)
                setattr(c, k, v)
    return CutSet(ans)
