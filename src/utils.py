import json
from collections import defaultdict
from typing import Any, Iterable
from collections import namedtuple

import numpy as np
from pathlib import Path


A_ORD = ord('а')
ALPHABET = ''.join(chr(ord('а') + i) for i in range(32))
NOT_FIRST_LETTERS = 'ь', 'ъ'
ALLOWED_FIRST_LETTER_MASK = np.array([c not in NOT_FIRST_LETTERS for c in ALPHABET], dtype=bool)
INF = 1e6
EPS = 1e-6
CENTER_DIST_PENALTY = ~ALLOWED_FIRST_LETTER_MASK * INF
GRID_NAMES = ['default', 'extra']


Trace = namedtuple('Trace', ['coordinates', 'times', 'grid_name'])
Sample = namedtuple('Sample', ['trace', 'word'])
Candidate = namedtuple('Candidate', ['word', 'coordinates'])
Suggestion = tuple[str, str, str, str]


class KeyboardGrid:
    def __init__(self, key_configs: list[dict]):
        self.x0s = np.zeros(32, np.int16)
        self.x1s = np.zeros(32, np.int16)
        self.y0s = np.zeros(32, np.int16)
        self.y1s = np.zeros(32, np.int16)
        self.valid_keys = set()
        self.key_order = []
        for kc in key_configs:
            if 'label' not in kc:
                continue
            char_index = ord(kc['label']) - A_ORD
            if char_index < 0 or char_index >= 32:
                continue
            self.valid_keys.add(kc['label'])
            self.key_order.append(char_index)
            self.x0s[char_index] = kc['hitbox']['x']
            self.x1s[char_index] = kc['hitbox']['x'] + kc['hitbox']['w']
            self.y0s[char_index] = kc['hitbox']['y']
            self.y1s[char_index] = kc['hitbox']['y'] + kc['hitbox']['h']
        center_xs = np.array((self.x0s + self.x1s) / 2)
        center_ys = np.array((self.y0s + self.y1s) / 2)
        self.key_centers = np.stack([center_xs, center_ys]).T

    def resolve_letter(self, x: int, y: int) -> str:
        fit_mask = (self.x0s <= x) & (x < self.x1s) & (self.y0s <= y) & (y < self.y1s) & ALLOWED_FIRST_LETTER_MASK
        true_indexes = np.argwhere(fit_mask).ravel()
        if len(true_indexes) == 1:
            char_index = true_indexes[0]
        else:
            center_dists = np.linalg.norm(self.key_centers - (x, y), axis=1)
            char_index = np.argmin(center_dists + CENTER_DIST_PENALTY)
        return ALPHABET[char_index]

    def make_curve(self, word: str) -> np.ndarray:
        char_codes = [ord(c) - A_ORD for c in word if c in self.valid_keys]
        return self.key_centers[char_codes]


def load_grids(path: Path) -> dict[str, KeyboardGrid]:
    with open(path) as f:
        keyboards_data = json.load(f)
    return {
        name: KeyboardGrid(props)
        for name, props in keyboards_data.items()
    }


class NBestCollector:
    def __init__(self, n: int = 4):
        self.n = n
        self.keys = ()
        self.values = ()

    def add(self, key: float, value: Any):
        pos = 0
        while pos < len(self.keys) and key >= self.keys[pos]:
            pos += 1
        if pos < self.n:
            self.keys = (*self.keys[:pos], key, *self.keys[pos:self.n - 1])
            self.values = (*self.values[:pos], value, *self.values[pos:self.n - 1])


class Vocabulary:
    def __init__(self, words: Iterable[tuple[str, int]]):
        self.words = words
        self.word_lists = defaultdict(list)
        self.word_codes = {}
        self.word_counts = {}
        for i, (w, c) in enumerate(words):
            self.word_lists[w[0]].append((w, i, c))
            self.word_codes[w] = i
            self.word_counts[w] = c
        for wl in self.word_lists.values():
            wl.sort()

    def get_by_first_letter(self, c: str) -> list[tuple[str, int, int]]:
        return self.word_lists[c]

    @staticmethod
    def load(path: Path) -> 'Vocabulary':
        with open(path) as f:
            data = []
            for line in f:
                word, counter = line.split(',')
                counter = int(counter)
                data.append((word, counter))
        return Vocabulary(data)

    def __iter__(self) -> Iterable[tuple[str, int, int]]:
        for _, wl in sorted(self.word_lists.items()):
            yield from wl


def _trace_step_lens(trace: np.ndarray) -> np.ndarray:
    steps = np.float32(trace[1:] - trace[:-1])
    step_lens = np.sum(steps ** 2, axis=1) ** .5
    return step_lens


def trace_len(trace: np.ndarray) -> float:
    step_lens = _trace_step_lens(trace)
    return step_lens.sum()


def interpolate_line(points: np.ndarray, step: float = 1.) -> np.ndarray:
    if len(points) <= 1:
        return points

    points = np.float32(points)
    ans = [points[0:1]]
    for prev_p, next_p in zip(points[:-1], points[1:]):
        step_vector = next_p - prev_p
        step_len = np.sum(step_vector ** 2) ** .5
        if step_len > step:
            weights = np.linspace(0, 1, int(step_len // step + 2))[1:-1].reshape((-1, 1))
            ans.append(prev_p + weights * (next_p - prev_p))
        ans.append([next_p])
    return np.concatenate(ans)
