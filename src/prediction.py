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


def trace_length_ratio(traces: list[utils.Trace], candidates: list[list[utils.Candidate]]) -> np.ndarray:
    return np.stack([
        np.array([utils.trace_len(c.coordinates) for c in cs]) / np.maximum(utils.trace_len(t.coordinates), 1)
        for t, cs in zip(traces, candidates)
    ])


def trace_length_diff(traces: list[utils.Trace], candidates: list[list[utils.Candidate]]) -> np.ndarray:
    return np.stack([
        np.array([utils.trace_len(c.coordinates) for c in cs]) - utils.trace_len(t.coordinates)
        for t, cs in zip(traces, candidates)
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
                fc(traces, candidates).astype(np.float32).ravel()
                for fc in self.feature_calculators
            ],
            axis=1,
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


class ExpSampler:
    def __init__(self, lam: float, return_index: bool = True):
        self.lam = lam
        self.return_index = return_index

    def __call__(self, options: list[utils.Candidate]) -> int | utils.Candidate:
        index = int(random.expovariate(self.lam)) % len(options)
        if self.return_index:
            return index
        return options[index]


def _find_candidate_index(word: str, candidates: list[utils.Candidate]) -> int:
    for i, c in enumerate(candidates):
        if c.word == word:
            return i
    return -1


def make_ranking_ds(
        original_dataset: Iterable[utils.Sample],
        candidate_generator: Callable[[list[utils.Trace]], list[list[utils.Candidate]]],
        features_extractor: FeaturesExtractorNP,
        keyboard_grids: dict[str, utils.KeyboardGrid],
        sampler: ExpSampler = random.choice,
        negatives_per_track: int = 5,
        batch_size: int = 100,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    features, labels, groups, pairs = [], [], [], []
    rows_per_track = 1 + negatives_per_track
    labels_pattern = np.zeros(rows_per_track, np.int32)
    labels_pattern[0] = 1
    batch_start_index = 0
    for i, batch in enumerate(utils.batch_iterable(original_dataset, batch_size)):
        candidates = candidate_generator([dp.trace for dp in batch])
        good_indices = [
            i
            for i, (s, cs) in enumerate(zip(batch, candidates))
            if any(s.word == c.word for c in cs)
        ]
        candidates = [candidates[i] for i in good_indices]
        batch = [batch[i] for i in good_indices]

        negative_ranks = [[sampler(cs) for _ in range(negatives_per_track)] for cs in candidates]
        negative_candidates = [
            [cs[i] for i in nis]
            for cs, nis in zip(candidates, negative_ranks)
        ]
        positive_candidates = [
            utils.Candidate(dp.word, keyboard_grids[dp.trace.grid_name].make_curve(dp.word))
            for dp in batch
        ]
        positive_ranks = [
            _find_candidate_index(dp.word, cs)
            for dp, cs in zip(batch, candidates)
        ]
        ranks = list(itertools.chain.from_iterable([pr, *nrs] for pr, nrs in zip(positive_ranks, negative_ranks)))

        selected_candidates = [[pc, *ncs] for pc, ncs in zip(positive_candidates, negative_candidates)]
        batch_features = features_extractor([dp.trace for dp in batch], selected_candidates)
        features.append(np.concatenate(
            (
                batch_features,
                np.array(ranks).reshape((-1, 1)),
            ),
            axis=1,
        ))

        for j in range(len(batch)):
            labels.append(labels_pattern)
            true_row_index = batch_start_index + j * rows_per_track
            groups.append(np.full(rows_per_track, true_row_index))
            pairs.append(np.stack(
                (
                    np.full(negatives_per_track, true_row_index),
                    np.arange(true_row_index + 1, true_row_index + rows_per_track)
                ),
                axis=1,
            ))
        batch_start_index += len(batch) * rows_per_track
    features = np.concatenate(features)
    labels = np.concatenate(labels)
    groups = np.concatenate(groups)
    pairs = np.concatenate(pairs)
    return features, labels, groups, pairs


def make_pairs_ds(
        original_dataset: Iterable[utils.Sample],
        candidate_generator: Callable[[list[utils.Trace]], list[list[utils.Candidate]]],
        trace_features_extractor: FeaturesExtractorNP,
        candidate_features_extractor: FeaturesExtractorNP,
        sampler: ExpSampler = random.choice,
        negatives_per_track: int = 10,
        batch_size: int = 1000,
) -> tuple[np.ndarray, np.ndarray]:
    features, labels = [], []
    rows_per_track = 1 + negatives_per_track
    for i, batch in enumerate(utils.batch_iterable(original_dataset, batch_size)):
        batch_candidates = candidate_generator([dp.trace for dp in batch])
        chosen_traces, chosen_candidates, chosen_ranks = [], [], []
        for sample, candidates in zip(batch, batch_candidates):
            good_rank = _find_candidate_index(sample.word, candidates)
            if good_rank == -1:
                continue
            negative_ranks = [sampler(candidates) for _ in range(negatives_per_track)]
            ranks = [good_rank, *negative_ranks]
            chosen_traces.append(sample.trace)
            chosen_candidates.append([candidates[r] for r in ranks])
            chosen_ranks.append(ranks)

        trace_features = trace_features_extractor(chosen_traces, chosen_candidates)
        candidate_features = candidate_features_extractor(chosen_traces, chosen_candidates)
        chosen_ranks = np.array(chosen_ranks).reshape((-1, 1))
        candidate_features = np.concatenate((candidate_features, chosen_ranks), axis=1)
        for pos_index in range(0, len(trace_features), rows_per_track):
            for neg_index in range(pos_index + 1, pos_index + rows_per_track):
                tgt_features = trace_features[pos_index]
                pos_features = candidate_features[pos_index]
                neg_features = candidate_features[neg_index]
                features.append(np.block([[tgt_features, pos_features, neg_features],
                                          [tgt_features, neg_features, pos_features]]))
                labels.append(np.array([1, 0]))

    features = np.concatenate(features)
    labels = np.concatenate(labels)
    return features, labels


class PairwiseRanker:
    def __init__(
            self,
            scorer: Callable,
            trace_feature_extractor: FeaturesExtractor,
            candidates_feature_extractor: FeaturesExtractor
    ):
        self.scorer = scorer
        self.trace_feature_extractor = trace_feature_extractor
        self.candidates_feature_extractor = candidates_feature_extractor

    def __call__(self, sample_traces: list[utils.Trace], candidates: list[list[utils.Candidate]]) -> list[utils.Suggestion]:
        trace_features = self.trace_feature_extractor(sample_traces, candidates)
        candidates_features = self.candidates_feature_extractor(sample_traces, candidates)
        candidates_per_trace = len(candidates[0])
        res_indexes = [(0,) for _ in sample_traces]
        for ci in range(1, candidates_per_trace):
            features = np.stack([
                np.concatenate((
                    trace_features[i * candidates_per_trace],
                    candidates_features[i * candidates_per_trace + ri],
                    [ri],
                    candidates_features[i * candidates_per_trace + ci],
                    [ci],
                ))
                for i, (ris, cs) in enumerate(zip(res_indexes, candidates))
                for ri in ris
            ])
            res = self.scorer(features)
            res = res.reshape((len(sample_traces), -1))
            new_indexes = res.sum(axis=1)
            res_indexes = [
                (*ris[:ni], ci, *ris[ni:])[:4]
                for ris, ni in zip(res_indexes, new_indexes)
            ]
        return [
            tuple(cs[i].word for i in ris)
            for cs, ris in zip(candidates, res_indexes)
        ]
