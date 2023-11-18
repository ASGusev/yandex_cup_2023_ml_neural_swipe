import abc
import math
import random
from collections import defaultdict
from typing import Iterator, Iterable

import cv2
import numpy as np
import torch

import data
import utils


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


class ImgPreprocessor(TracePreprocessor):
    n_channels = 3

    def __init__(
            self,
            keyboard_grids: dict[str, utils.KeyboardGrid],
            vocabulary: utils.Vocabulary,
            coordinates_max: np.ndarray = np.array([1080, 667]),
            res: int = 32,
    ):
        super().__init__()
        self.vocabulary = vocabulary
        self.coordinates_max = coordinates_max
        self.keyboard_grids = keyboard_grids
        self.res = res

    def _make_datapoint(self, points: np.ndarray, grid_name: str, origin_flag: int) -> np.ndarray:
        img = make_curve_image(points, self.res)
        grid_channel = np.full_like(img, utils.GRID_NAMES.index(grid_name))
        origin_channel = np.full_like(img, origin_flag)
        return np.stack((img, grid_channel, origin_channel))

    def preprocess_real(self, trace: utils.Trace) -> np.ndarray:
        points = trace.coordinates / self.coordinates_max
        return self._make_datapoint(points, trace.grid_name, 0)

    def preprocess_proj(self, word: str, grid_name: str) -> np.ndarray:
        points = self.keyboard_grids[grid_name].make_curve(word) / self.coordinates_max
        return self._make_datapoint(points, grid_name, 1)

    def merge_batch(self, samples: list[np.ndarray]) -> dict:
        return {
            'x': torch.FloatTensor(np.stack(samples)),
        }


def make_curve_image(points: np.ndarray, res: int) -> np.ndarray:
    points = np.maximum(points, 0)
    points = np.minimum(points, 1)
    points = points * res
    points = np.expand_dims(points, 0).astype(np.int32)
    canvas = np.zeros((res, res), np.int8)
    # noinspection PyTypeChecker
    cv2.drawContours(canvas, points, -1, 1)
    return canvas
