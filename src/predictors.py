import itertools
import multiprocessing
import random
from typing import Callable, Iterable

import numpy as np
import pandas as pd
from tslearn.metrics import dtw

import utils


class FirstLetterCandidateGenerator:
    def __init__(self, vocabulary: utils.Vocabulary, keyboard_grids: dict[str, utils.KeyboardGrid]):
        self.vocabulary = vocabulary
        self.keyboard_grids = keyboard_grids

    @utils.apply_to_batch
    def __call__(self, trace: utils.Trace) -> list[utils.Candidate]:
        grid = self.keyboard_grids[trace.grid_name]
        first_letter = grid.resolve_letter(*trace.coordinates[0])
        candidate_words = self.vocabulary.get_by_first_letter(first_letter)
        return [utils.Candidate(cw, grid.make_curve(cw)) for cw, _, _ in candidate_words]


class TopCandidateGenerator:
    def __init__(self, vocabulary: utils.Vocabulary, keyboard_grids: dict[str, utils.KeyboardGrid], n: int):
        self.vocabulary = vocabulary
        self.keyboard_grids = keyboard_grids
        words = sorted(vocabulary, key=lambda wt: -wt[2])
        words = [w for w, _, _ in words[:n]]
        self.candidates = {
            gn: [utils.Candidate(w, g.make_curve(w)) for w in words]
            for gn, g in keyboard_grids.items()
        }

    def __call__(self, traces: list[utils.Trace]) -> list[list[utils.Candidate]]:
        return [self.candidates[t.grid_name] for t in traces]


class FirstLetterLengthCandidateGenerator:
    def __init__(self,
                 vocabulary: utils.Vocabulary, keyboard_grids: dict[str, utils.KeyboardGrid],
                 min_share: float = .25, max_share: float = 1.75, min_max_len: float = 300.):
        self.vocabulary = vocabulary
        self.keyboard_grids = keyboard_grids
        self.min_share = min_share
        self.max_share = max_share
        self.min_max_len = min_max_len

    @utils.apply_to_batch
    def __call__(self, trace: utils.Trace) -> list[utils.Candidate]:
        grid = self.keyboard_grids[trace.grid_name]
        first_letter = grid.resolve_letter(*trace.coordinates[0])
        candidate_words = self.vocabulary.get_by_first_letter(first_letter)
        trace_len = utils.trace_len(trace.coordinates)
        min_trace_len = trace_len * self.min_share
        max_trace_len = max(trace_len * self.max_share, self.min_max_len)
        close_candidates = [
            cw
            for cw, _, _ in candidate_words
            if min_trace_len <= utils.trace_len(grid.make_curve(cw)) <= max_trace_len
        ]
        while len(close_candidates) < 4:
            close_candidates.append(random.choice(candidate_words))
        return [utils.Candidate(cw, grid.make_curve(cw)) for cw in close_candidates]


def rank_by_dtw(sample_trace: utils.Trace, candidates: list[utils.Candidate]) -> utils.Suggestion:
    collector = utils.NBestCollector()
    for c in candidates:
        distance = dtw(c.coordinates, sample_trace.coordinates)
        collector.add(distance, c.word)
    return collector.values


class InterpolatedDTWRanker:
    def __init__(self, step: float):
        self.step = step

    def __call__(self, sample_trace: utils.Trace, candidates: list[utils.Candidate]) -> utils.Suggestion:
        collector = utils.NBestCollector()
        interp_sample = utils.interpolate_line(sample_trace.coordinates, self.step)
        for c in candidates:
            interp_candidate = utils.interpolate_line(c.coordinates, self.step)
            distance = dtw(interp_candidate, interp_sample)
            collector.add(distance, c.word)
        return collector.values


class Predictor:
    def __init__(
            self,
            candidate_generator: Callable[[list[utils.Trace]], list[list[utils.Candidate]]],
            ranker: Callable[[list[utils.Trace], list[list[utils.Candidate]]], list[utils.Suggestion]],
            batch_size: int = 500,
    ):
        self.candidate_generator = candidate_generator
        self.ranker = ranker
        self.batch_size = batch_size

    def __call__(self, sample_traces: Iterable[utils.Trace]) -> list[utils.Suggestion]:
        res = []
        for batch in utils.batch_iterable(sample_traces, self.batch_size):
            candidates = self.candidate_generator(batch)
            res.extend(self.ranker(batch, candidates))
        return res


class PopularityCalculator:
    def __init__(self, vocabulary: utils.Vocabulary):
        self.vocabulary = vocabulary

    def __call__(self, trace: utils.Trace, candidates: list[utils.Candidate]) -> np.ndarray:
        return np.array([
            [self.vocabulary.word_counts[c.word] for c in cs]
            for cs in candidates
        ])


def _calc_task_interpolated_dtw(trace: utils.Trace, candidates: list[utils.Candidate], step: int) -> np.ndarray:
    interpolated_target = utils.interpolate_line(trace.coordinates, step)
    return np.array([
        dtw(utils.interpolate_line(c.coordinates, step), interpolated_target)
        for c in candidates
    ])


class InterpolatedDTWCalculator:
    def __init__(self, step: float, chunk_size: int = 4):
        self.chunk_size = chunk_size
        self.step = step

    def __call__(self, traces: list[utils.Trace], candidates: list[list[utils.Candidate]]) -> np.ndarray:
        args_gen = ((t, c, self.step) for t, c in zip(traces, candidates))
        if len(traces) >= 2 * self.chunk_size:
            with multiprocessing.Pool() as pool:
                dists = pool.starmap(_calc_task_interpolated_dtw, args_gen, self.chunk_size)
        else:
            dists = list(itertools.starmap(_calc_task_interpolated_dtw, args_gen))
        return np.array(dists)


def keyboard_grid(traces: list[utils.Trace], candidates: list[list[utils.Candidate]]) -> np.ndarray:
    return np.stack([
        np.full(len(cs), utils.GRID_NAMES.index(t.grid_name), np.float32)
        for t, cs in zip(traces, candidates)
    ])


def target_trace_length(traces: list[utils.Trace], candidates: list[list[utils.Candidate]]) -> np.ndarray:
    return np.stack([
        np.full(len(cs), utils.trace_len(t.coordinates))
        for t, cs in zip(traces, candidates)
    ])


def candidate_trace_length(_: list[utils.Trace], candidates: list[list[utils.Candidate]]) -> np.ndarray:
    return np.array([
        [utils.trace_len(c.coordinates) for c in cs]
        for cs in candidates
    ])


class FeaturesExtractor:
    def __init__(
            self,
            feature_calculators: dict[str, Callable[[list[utils.Trace], list[list[utils.Candidate]]], np.ndarray]],
    ):
        self.feature_calculators = feature_calculators

    def __call__(self, traces: list[utils.Trace], candidates: list[list[utils.Candidate]]) -> pd.DataFrame:
        return pd.DataFrame({
            fn: list(itertools.chain.from_iterable(fc(traces, candidates)))
            for fn, fc in self.feature_calculators.items()
        })


class FeaturesExtractorNP:
    def __init__(
            self,
            feature_calculators: list[Callable[[list[utils.Trace], list[list[utils.Candidate]]], np.ndarray]],
    ):
        self.feature_calculators = feature_calculators

    def __call__(self, traces: list[utils.Trace], candidates: list[list[utils.Candidate]]) -> np.ndarray:
        return np.stack(
            [
                fc(traces, candidates).astype(np.float32)
                for fc in self.feature_calculators
            ],
            axis=2,
        )


class ScoringRanker:
    def __init__(self, scorer: Callable, feature_extractor: FeaturesExtractor):
        self.scorer = scorer
        self.feature_extractor = feature_extractor

    def __call__(self, sample_traces: list[utils.Trace], candidates: list[list[utils.Candidate]]) -> list[utils.Suggestion]:
        res = []
        features = self.feature_extractor(sample_traces, candidates)
        scores = self.scorer(features)
        for ss, cs in zip(utils.batch_iterable(scores, len(scores) // len(candidates)), candidates):
            res.append(tuple(
                cs[i].word
                for i in np.array(ss).argsort()[:4]
            ))
        return res


def make_scorer_ds(
        original_dataset: Iterable[utils.Sample],
        candidate_generator: Callable[[list[utils.Trace]], list[list[utils.Candidate]]],
        features_extractor: FeaturesExtractor,
        keyboard_grids: dict[str, utils.KeyboardGrid],
        sampler: Callable[[list[utils.Candidate]], utils.Candidate] = random.choice,
        batch_size: int = 100,
) -> tuple[pd.DataFrame, np.ndarray]:
    features, targets = [], []
    for batch in utils.batch_iterable(original_dataset, batch_size):
        candidates = candidate_generator([dp.trace for dp in batch])
        negative_candidates = [sampler(cs) for cs in candidates]
        positive_candidates = [
            utils.Candidate(dp.word, keyboard_grids[dp.trace.grid_name].make_curve(dp.word))
            for dp in batch
        ]
        features.append(features_extractor(
            [dp.trace for dp in batch],
            list(map(list, zip(positive_candidates, negative_candidates))),
        ))
        for _ in batch:
            targets.extend([1, 0])
    features = pd.concat(features).reset_index(drop=True)
    targets = np.array(targets)
    return features, targets


def make_ranking_ds(
        original_dataset: Iterable[utils.Sample],
        candidate_generator: Callable[[list[utils.Trace]], list[list[utils.Candidate]]],
        features_extractor: FeaturesExtractorNP,
        keyboard_grids: dict[str, utils.KeyboardGrid],
        sampler: Callable[[list[utils.Candidate]], utils.Candidate] = random.choice,
        negatives_per_track: int = 5,
        batch_size: int = 100,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    features, labels, groups, pairs = [], [], [], []
    rows_per_track = 1 + negatives_per_track
    labels_pattern = np.zeros(rows_per_track, np.int32)
    labels_pattern[0] = 1
    for i, batch in enumerate(utils.batch_iterable(original_dataset, batch_size)):
        candidates = candidate_generator([dp.trace for dp in batch])
        negative_candidates = [[sampler(cs) for _ in range(negatives_per_track)] for cs in candidates]
        positive_candidates = [
            utils.Candidate(dp.word, keyboard_grids[dp.trace.grid_name].make_curve(dp.word))
            for dp in batch
        ]

        selected_candidates = [[pc, *ncs] for pc, ncs in zip(positive_candidates, negative_candidates)]
        batch_features = features_extractor([dp.trace for dp in batch], selected_candidates)
        features.append(batch_features.reshape(
            (batch_features.shape[0] * batch_features.shape[1], batch_features.shape[2])
        ))

        for j in range(len(batch)):
            labels.append(labels_pattern)
            groups.append(np.full(rows_per_track, i * batch_size + j))
            true_row_index = (i * batch_size + j) * rows_per_track
            pairs.append(np.stack(
                (
                    np.full(negatives_per_track, true_row_index),
                    np.arange(true_row_index + 1, true_row_index + rows_per_track)
                ),
                axis=1,
            ))
    features = np.concatenate(features)
    labels = np.concatenate(labels)
    groups = np.concatenate(groups)
    pairs = np.concatenate(pairs)
    return features, labels, groups, pairs


class ExpSampler:
    def __init__(self, lam: float):
        self.lam = lam

    def __call__(self, options: list):
        index = int(random.expovariate(self.lam)) % len(options)
        return options[index]
