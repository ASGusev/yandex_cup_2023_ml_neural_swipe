import random
from typing import Callable

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
