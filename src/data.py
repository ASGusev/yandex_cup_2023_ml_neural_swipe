import csv
from collections.abc import Sequence
from pathlib import Path
from typing import Iterable

import numpy as np

import utils


class TracesReader:
    def __init__(self, path: Path):
        self.path = path

    def __iter__(self) -> Iterable[utils.Trace]:
        with open(self.path) as f:
            reader = csv.reader(f, delimiter=' ')
            for line in reader:
                word, *track = line
                track = list(map(int, track))
                n = len(track) // 3
                x = np.array(track[:n], np.int16)
                y = np.array(track[n:2 * n], np.int16)
                time = np.array(track[2 * n:], np.int16)
                coordinates = np.stack((x, y)).T
                yield utils.Trace(coordinates, time, word)


def read_words(path: Path) -> Iterable[str]:
    return path.read_text().split()


def save_results(path: Path, results: Iterable[tuple[str, str, str, str]]):
    with open(path, 'wt') as f:
        writer = csv.writer(f)
        for r in results:
            writer.writerow(r)


class Dataset:
    def __init__(self, traces: Sequence[np.ndarray], words: list[str]):
        self.traces = traces
        self.words = words

    def __getitem__(self, index: int) -> utils.Sample:
        return utils.Sample(self.traces[index], self.words[index])

    def __len__(self):
        return len(self.traces)

    def __iter__(self):
        for t, w in zip(self.traces, self.words):
            yield utils.Sample(t, w)


class BinaryDataset:
    def __init__(self, coordinates: np.ndarray, times: np.ndarray, extra_grid: np.array, lens: np.ndarray):
        self.coordinates = coordinates
        self.times = times
        self.extra_grid = extra_grid
        self.coordinates_end_indexes = lens.cumsum()
        self.coordinates_start_indexes = self.coordinates_end_indexes - lens

    def __getitem__(self, index: int) -> utils.Trace:
        coordinates_start_index = self.coordinates_start_indexes[index]
        coordinates_end_index = self.coordinates_end_indexes[index]
        return utils.Trace(
            self.coordinates[coordinates_start_index:coordinates_end_index],
            self.times[coordinates_start_index:coordinates_end_index],
            utils.GRID_NAMES[self.extra_grid[index]],
        )

    def __len__(self) -> int:
        return len(self.extra_grid)

    def __iter__(self) -> Iterable[utils.Trace]:
        for i in range(len(self)):
            yield self[i]

    @classmethod
    def load(cls, path: str | Path) -> 'BinaryDataset':
        return BinaryDataset(**np.load(path))
