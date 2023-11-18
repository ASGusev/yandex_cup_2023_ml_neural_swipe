import itertools
import multiprocessing
from typing import Callable

import numpy as np
import pandas as pd
from tslearn.metrics import dtw

import utils


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
