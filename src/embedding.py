import abc
import itertools
import math
import random
from collections import defaultdict
from typing import Iterable, Callable, Iterator, Optional

import numpy as np
import torch
import voyager
from pathlib import Path
from torch import nn
from tqdm.auto import tqdm, trange

import utils
import data


class TracePreprocessor(abc.ABC):
    @abc.abstractmethod
    def preprocess_real(self, trace: utils.Trace) -> np.ndarray:
        raise NotImplementedError

    @abc.abstractmethod
    def preprocess_proj(self, word: str, grid_name: str) -> np.ndarray:
        raise NotImplementedError

    @abc.abstractmethod
    def merge_batch(self, samples: list[np.ndarray]) -> dict:
        raise NotImplementedError


class SeqPreprocessor(TracePreprocessor):
    dim = 5

    def __init__(
            self,
            keyboard_grids: dict[str, utils.KeyboardGrid],
            vocabulary: utils.Vocabulary,
            coordinates_max: np.ndarray = np.array([1080, 667]),
            max_len: int = 96,
    ):
        super().__init__()
        self.max_len = max_len
        self.keyboard_grids = keyboard_grids
        self.max_len = max_len
        self.coordinates_max = coordinates_max
        self.vocabulary = vocabulary

    def preprocess_real(self, trace: utils.Trace) -> np.ndarray:
        curve = trace.coordinates[:self.max_len] / self.coordinates_max
        time = np.expand_dims(trace.times[:self.max_len], 1)
        grid_mask = np.full((len(curve),  1), utils.GRID_NAMES.index(trace.grid_name), np.float32)
        real_mask = np.ones((len(curve), 1), np.float32)
        return np.concatenate((curve, time, grid_mask, real_mask), axis=1)

    def preprocess_proj(self, word: str, grid_name: str) -> np.ndarray:
        coordinates = self.keyboard_grids[grid_name].make_curve(word)
        curve = coordinates[:self.max_len] / self.coordinates_max
        time = np.full((len(curve), 1), -1, np.float32)
        grid_mask = np.full((len(curve), 1), utils.GRID_NAMES.index(grid_name), np.float32)
        real_mask = np.zeros((len(curve), 1), np.float32)
        return np.concatenate((curve, time, grid_mask, real_mask), axis=1)

    def merge_batch(self, samples: list[np.ndarray]) -> dict:
        lengths = np.array(list(map(len, samples)), np.int32)
        traces = np.zeros(((len(samples)), lengths.max(), self.dim))
        for i, t in enumerate(samples):
            traces[i, :len(t)] = t
        return {
            'traces': torch.FloatTensor(traces),
            'lengths': torch.IntTensor(lengths),
        }


class MetricLearningDataset(abc.ABC):
    def __init__(
            self,
            dataset: data.Dataset,
            vocabulary: utils.Vocabulary,
            preprocessor: TracePreprocessor,
            # The actual BS is x2 because an actual and a projected tracks are returned
            batch_size: int,
    ):
        self.preprocessor = preprocessor
        self.dataset = dataset
        self.vocabulary = vocabulary
        self.batch_size = batch_size
        self.length = math.ceil(len(self.dataset) / batch_size)

    @abc.abstractmethod
    def _get_samples(self, start_index: int, end_index: int) -> Iterator[np.ndarray]:
        raise NotImplementedError

    def __len__(self) -> int:
        return self.length

    @abc.abstractmethod
    def __getitem__(self, index: int) -> dict:
        raise NotImplementedError

    def __iter__(self) -> Iterable[dict]:
        for i in range(self.length):
            yield self[i]


class TargetPairDataset(MetricLearningDataset):
    def _get_samples(self, start_index: int, end_index: int) -> Iterator[np.ndarray]:
        for index in range(start_index, end_index):
            trace, word = self.dataset[index]
            yield self.preprocessor.preprocess_real(trace)
            yield self.preprocessor.preprocess_proj(word, trace.grid_name)

    @staticmethod
    def _sample_pairs(n: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        anchor_pairs = torch.arange(n)
        neg_pairs = (anchor_pairs + torch.randint(1, n, (n,))) % n
        anchor_flags = torch.rand(n) < .5
        anchors = anchor_pairs * 2 + anchor_flags
        positives = anchor_pairs * 2 + ~anchor_flags
        negatives = neg_pairs * 2 + ~anchor_flags
        return anchors, positives, anchors, negatives

    def __getitem__(self, index: int) -> dict:
        start_index = index * self.batch_size
        end_index = min(start_index + self.batch_size, len(self.dataset))
        return {
            **self.preprocessor.merge_batch(list(self._get_samples(start_index, end_index))),
            'indices_tuple': self._sample_pairs(end_index - start_index),
        }


class TripletDataset(MetricLearningDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.words_index = defaultdict(list)
        self.words_to_use = {w for w, _, c in self.vocabulary if c > 1}
        indexes_to_use = []
        for i, w in enumerate(self.dataset.words):
            if w in self.words_to_use:
                self.words_index[w].append(i)
                indexes_to_use.append(i)
        self.indexes_to_use = np.array(indexes_to_use)
        self.length = len(indexes_to_use)

    def _get_samples(self, start_index: int, end_index: int) -> Iterator[np.ndarray]:
        for i in self.indexes_to_use[start_index:end_index]:
            anchor_trace, word = self.dataset[i]
            yield self.preprocessor.preprocess_real(anchor_trace)
            positive_index = i
            while positive_index == i:
                positive_index = random.choice(self.words_index[word])
            positive_trace, _ = self.dataset[positive_index]
            yield self.preprocessor.preprocess_real(positive_trace)
            negative_trace, negative_word = None, word
            while negative_word == word:
                negative_trace, negative_word = self.dataset[random.choice(self.indexes_to_use)]
            yield self.preprocessor.preprocess_real(negative_trace)

    @staticmethod
    def _make_indices_tuple(n: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        anchors = torch.arange(0, n * 3, 3)
        positives = torch.arange(1, n * 3, 3)
        negatives = torch.arange(2, n * 3, 3)
        return anchors, positives, anchors, negatives

    def __getitem__(self, index: int) -> dict:
        start_index = index * self.batch_size
        end_index = min(start_index + self.batch_size, len(self.dataset))
        return {
            **self.preprocessor.merge_batch(list(self._get_samples(start_index, end_index))),
            'indices_tuple': self._make_indices_tuple(end_index - start_index),
        }


class Embedder(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, batch: dict) -> torch.Tensor:
        raise NotImplementedError

    @property
    def device(self) -> str:
        return next(self.parameters()).device


class LSTMEmbedder(Embedder):
    def __init__(self, in_dim: int = 5, out_dim: int = 64, n_layers: int = 4):
        super().__init__()
        self.rnn = nn.LSTM(in_dim, out_dim, n_layers, batch_first=True)

    def forward(self, batch: dict) -> torch.Tensor:
        x = nn.utils.rnn.pack_padded_sequence(batch['traces'], batch['lengths'], True, enforce_sorted=False)
        x = x.to(self.device)
        rnn_out, _ = self.rnn(x)
        padded_out, _ = nn.utils.rnn.pad_packed_sequence(rnn_out, True)
        res = padded_out[torch.arange(len(batch['lengths'])), batch['lengths'] - 1]
        return nn.functional.normalize(res, dim=1)


class PositionalEncoder(nn.Module):
    def __init__(self, dim: int = 3, max_size: int = 96):
        super().__init__()
        matrix = torch.arange(max_size).unsqueeze(1) * torch.arange(1, dim + 1) * (2 * torch.pi / max_size)
        embeddings = torch.cat((torch.sin(matrix), torch.cos(matrix)), dim=1)
        self.embeddings = torch.nn.Parameter(embeddings.unsqueeze(0), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.concatenate(
            (
                x,
                self.embeddings[:, :x.shape[1]].repeat(len(x), 1, 1),
            ),
            dim=2
        )


class TransformerEmbedder(Embedder):
    def __init__(
            self,
            in_dim: int = 5, emb_dim: int = 32,
            n_heads: int = 4, n_layers: int = 4,
            max_size: int = 96, pos_dim: int = 3
    ):
        super().__init__()
        self.positional_encoder = PositionalEncoder(pos_dim, max_size)
        self.proj = nn.Conv1d(in_dim + 2 * pos_dim, emb_dim, 1)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(emb_dim, n_heads, emb_dim, batch_first=True, activation='gelu'),
            n_layers
        )

    def forward(self, batch: dict) -> torch.Tensor:
        x = self.positional_encoder(batch['traces'].to(self.device))
        in_emb = self.proj(x.permute(0, 2, 1)).permute(0, 2, 1)
        mask = torch.ones(batch['traces'].shape[:2], dtype=torch.bool)
        for i, sl in enumerate(batch['lengths']):
            mask[i, :sl] = False
        mask = mask.to(self.device)
        embeddings = self.transformer(in_emb, src_key_padding_mask=mask)[:, 0]
        return nn.functional.normalize(embeddings, dim=1)


def center_dist_penalty(embeddings: torch.Tensor) -> torch.Tensor:
    distances = torch.norm(embeddings - embeddings.mean(axis=0), dim=1)
    return torch.mean((distances - 1) ** 2)


# noinspection PyTypeChecker
# noinspection PyUnresolvedReferences
def train(
        model: Embedder, opt: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler.LRScheduler,
        train_data: data.Dataset, val_data: data.Dataset,
        batches_per_epoch: int, n_epochs: int,
        loss_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        penalty_function: Callable[[torch.Tensor], torch.Tensor], penalty_weight: float = 0.,
        saving_path: Optional[Path] = None,
):
    train_iterator = utils.cycle(train_data)
    best_val_loss = float('inf')
    for epoch in range(1, n_epochs + 1):
        train_losses, val_losses = [], []
        train_penalties, val_penalties = [], []
        model.train()
        for batch in tqdm(
                itertools.islice(train_iterator, batches_per_epoch),
                total=batches_per_epoch,
                leave=False
        ):
            opt.zero_grad()
            embeddings = model(batch)
            loss = loss_function(embeddings, indices_tuple=batch['indices_tuple'])
            penalty = penalty_function(embeddings) if penalty_weight else torch.tensor(0)
            total_loss = loss + penalty * penalty_weight
            total_loss.backward()
            opt.step()
            train_losses.append(loss.item())
            train_penalties.append(penalty.item())
            del total_loss, penalty, loss, embeddings
        model.eval()
        with torch.no_grad():
            for batch in tqdm(val_data, leave=False):
                embeddings = model(batch)
                loss = loss_function(embeddings, indices_tuple=batch['indices_tuple'])
                val_losses.append(loss.item())
        train_loss = np.mean(train_losses)
        train_penalty = np.mean(train_penalties)
        val_loss = np.mean(val_losses)
        print(f'Epoch {epoch} train loss: {train_loss:.3f} train penalty: {train_penalty:.3f} val loss: {val_loss:.3f}')
        if scheduler:
            scheduler.step()
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if saving_path:
                torch.save(model.state_dict(), saving_path)


def calculate_word_embeddings(
        model: nn.Module,
        words: list[str],
        keyboard_grids: dict[str, utils.KeyboardGrid],
        preprocessor: TracePreprocessor,
        batch_size: int = 1000,
) -> dict[str, np.ndarray]:
    embeddings = {}
    with torch.no_grad():
        for grid_name, _ in keyboard_grids.items():
            grid_embeddings = []
            # noinspection PyTypeChecker
            for start_index in trange(0, len(words), batch_size, leave=False, desc=grid_name):
                end_index = min(len(words), start_index + batch_size)
                curves = [
                    preprocessor.preprocess_proj(w, grid_name)
                    for w in words[start_index:end_index]
                ]
                batch_embeddings = model(preprocessor.merge_batch(curves))
                grid_embeddings.append(batch_embeddings.cpu().numpy())
            embeddings[grid_name] = np.concatenate(grid_embeddings)
    return embeddings


class EmbeddingCandidateGenerator:
    def __init__(
            self,
            model: nn.Module,
            grid_vectors: dict[str, np.ndarray],
            preprocessor: TracePreprocessor,
            vocabulary: utils.Vocabulary,
            keyboard_grids: dict[str, utils.KeyboardGrid],
            n_candidates: int,
            dim: int = 32,
            min_freq: int = 0,
    ):
        self.keyboard_grids = keyboard_grids
        self.n_candidates = n_candidates
        self.vocabulary = vocabulary
        self.preprocessor = preprocessor
        self.model = model
        self.grid_indexes = {
            gn: voyager.Index(voyager.Space.Euclidean, num_dimensions=dim)
            for gn in grid_vectors
        }
        rel_indexes = np.array([
            i
            for w, i, c in vocabulary
            if c >= min_freq
        ])
        print(f'Using {len(rel_indexes)} words')
        for gn, vs in grid_vectors.items():
            self.grid_indexes[gn].add_items(vs[rel_indexes], rel_indexes)

    def __call__(self, traces: list[utils.Trace]) -> list[list[utils.Candidate]]:
        prep_curves = list(map(self.preprocessor.preprocess_real, traces))
        batch = self.preprocessor.merge_batch(prep_curves)
        with torch.no_grad():
            embeddings = self.model(batch).cpu().numpy()
        word_indexes = np.ndarray((len(traces), self.n_candidates), np.int32)
        extra_grid_mask = np.array([utils.GRID_NAMES.index(t.grid_name) for t in traces], bool)
        if extra_grid_mask.any():
            word_indexes[extra_grid_mask], _ = self.grid_indexes['extra'] \
                .query(embeddings[extra_grid_mask], self.n_candidates)
        if not extra_grid_mask.all():
            word_indexes[~extra_grid_mask], _ = self.grid_indexes['default'] \
                .query(embeddings[~extra_grid_mask], self.n_candidates)
        return [
            [
                utils.Candidate(w := self.vocabulary.words[wi], self.keyboard_grids[trace.grid_name].make_curve(w))
                for wi in wis
            ]
            for trace, wis in zip(traces, word_indexes)
        ]


class EmbeddingDistCalculator:
    def __init__(
            self,
            preprocessor: TracePreprocessor,
            model: Embedder,
            vocabulary_embeddings: dict[str, np.ndarray],
            vocabulary: utils.Vocabulary,
    ):
        self.model = model
        self.vocabulary_embeddings = vocabulary_embeddings
        self.vocabulary = vocabulary
        self.preprocessor = preprocessor

    def __call__(self, traces: list[utils.Trace], candidates: list[list[utils.Candidate]]) -> np.ndarray:
        batch = self.preprocessor.merge_batch(list(map(self.preprocessor.preprocess_real, traces)))
        with torch.no_grad():
            trace_embeddings = self.model(batch)
            word_indices = np.array([
                [self.vocabulary.word_codes[c.word] for c in cs]
                for cs in candidates
            ])
            word_embeddings = np.array([
                self.vocabulary_embeddings[t.grid_name][wis]
                for t, wis in zip(traces, word_indices)
            ])
            trace_embeddings = trace_embeddings.cpu().numpy()
            dists = np.linalg.norm(word_embeddings - np.expand_dims(trace_embeddings, 1), axis=2)
        return dists
