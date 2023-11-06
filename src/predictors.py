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

    def __call__(self, trace: utils.Trace) -> list[utils.Candidate]:
        return self.candidates[trace.grid_name]


class FirstLetterLengthCandidateGenerator:
    def __init__(self,
                 vocabulary: utils.Vocabulary, keyboard_grids: dict[str, utils.KeyboardGrid],
                 min_share: float = .25, max_share: float = 1.75, min_max_len: float = 300.):
        self.vocabulary = vocabulary
        self.keyboard_grids = keyboard_grids
        self.min_share = min_share
        self.max_share = max_share
        self.min_max_len = min_max_len

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


class ScoringRanker:
    def __init__(self, scorer: Callable, feature_extractor: Callable):
        self.scorer = scorer
        self.feature_extractor = feature_extractor

    def __call__(self, sample_trace: utils.Trace, candidates: list[utils.Candidate]) -> utils.Suggestion:
        features = self.feature_extractor(sample_trace, candidates)
        scores = self.scorer(features)
        return tuple(
            candidates[i].word
            for i in scores.argsort()[:4]
        )


class Predictor:
    def __init__(
            self,
            candidate_generator: Callable[[utils.Trace], list[utils.Candidate]],
            ranker: Callable[[utils.Trace, list[utils.Candidate]], utils.Suggestion],
    ):
        self.candidate_generator = candidate_generator
        self.ranker = ranker

    def __call__(self, sample_trace: utils.Trace) -> utils.Suggestion:
        candidates = self.candidate_generator(sample_trace)
        return self.ranker(sample_trace, candidates)


class PopularityCalculator:
    def __init__(self, vocabulary: utils.Vocabulary):
        self.vocabulary = vocabulary

    def __call__(self, trace: utils.Trace, candidates: list[utils.Candidate]) -> np.ndarray:
        return np.array([self.vocabulary.word_counts[c.word] for c in candidates])


class InterpolatedDTWCalculator:
    def __init__(self, step: float):
        self.step = step

    def __call__(self, trace: utils.Trace, candidates: list[utils.Candidate]) -> np.ndarray:
        interpolated_target = utils.interpolate_line(trace.coordinates, self.step)
        return np.array([
            dtw(utils.interpolate_line(c.coordinates, self.step), interpolated_target)
            for c in candidates
        ])


def target_trace_length(trace: utils.Trace, candidates: list[utils.Candidate]) -> np.ndarray:
    return np.full(len(candidates), utils.trace_len(trace.coordinates))


def candidate_trace_length(_: utils.Trace, candidates: list[utils.Candidate]) -> np.ndarray:
    return np.array([utils.trace_len(c.coordinates) for c in candidates])


class FeaturesExtractor:
    def __init__(self, feature_calculators: dict[str, Callable[[utils.Trace, list[utils.Candidate]], np.ndarray]]):
        self.feature_calculators = feature_calculators

    def __call__(self, trace: utils.Trace, candidates: list[utils.Candidate]) -> pd.DataFrame:
        return pd.DataFrame({
            fn: fc(trace, candidates)
            for fn, fc in self.feature_calculators.items()
        })


def make_scorer_ds(
        original_dataset: Iterable[utils.Sample],
        candidate_generator: Callable[[utils.Trace], list[utils.Candidate]],
        features_extractor: FeaturesExtractor,
        keyboard_grids: dict[str, utils.KeyboardGrid],
) -> tuple[pd.DataFrame, np.ndarray]:
    features, targets = [], []
    for dp in original_dataset:
        candidates = candidate_generator(dp.trace)
        negative_candidate = random.choice(candidates)
        positive_candidate = utils.Candidate(dp.word, keyboard_grids[dp.trace.grid_name].make_curve(dp.word))
        features.append(features_extractor(dp.trace, [positive_candidate, negative_candidate]))
        targets.extend([1, 0])
    features = pd.concat(features).reset_index(drop=True)
    targets = np.array(targets)
    return features, targets
