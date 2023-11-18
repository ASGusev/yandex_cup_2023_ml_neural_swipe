import faiss
import numpy as np
import torch
import voyager
from torch import nn
from tqdm.asyncio import trange

import utils
from embeddings.data import TracePreprocessor
from embeddings.models import Embedder


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


class FAISSEmbeddingCandidateGenerator:
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
        self.grid_indexes = {}
        rel_indexes = np.array([
            i
            for w, i, c in vocabulary
            if c >= min_freq
        ])
        print(f'Using {len(rel_indexes)} words')
        for gn, vs in grid_vectors.items():
            self.grid_indexes[gn] = faiss.IndexFlatL2(dim)
            self.grid_indexes[gn].add(vs[rel_indexes])
        self.rel_indexes = rel_indexes

    def __call__(self, traces: list[utils.Trace]) -> list[list[utils.Candidate]]:
        prep_curves = list(map(self.preprocessor.preprocess_real, traces))
        batch = self.preprocessor.merge_batch(prep_curves)
        with torch.no_grad():
            embeddings = self.model(batch).cpu().numpy()
        word_indexes = np.ndarray((len(traces), self.n_candidates), np.int32)
        extra_grid_mask = np.array([utils.GRID_NAMES.index(t.grid_name) for t in traces], bool)
        if extra_grid_mask.any():
            _, word_indexes[extra_grid_mask] = self.grid_indexes['extra'] \
                .search(embeddings[extra_grid_mask], self.n_candidates)
        if not extra_grid_mask.all():
            _, word_indexes[~extra_grid_mask] = self.grid_indexes['default'] \
                .search(embeddings[~extra_grid_mask], self.n_candidates)
        return [
            [
                utils.Candidate(w := self.vocabulary.words[wi], self.keyboard_grids[trace.grid_name].make_curve(w))
                for wi in self.rel_indexes[wis]
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
