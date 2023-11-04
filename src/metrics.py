import csv
import itertools
from typing import Iterable

import numpy as np
from pathlib import Path


POS_WEIGHTS = 1., .1, .09, .08


def mrr_dp(prediction: tuple[str, str, str, str], target: str) -> float:
    for w, p in zip(POS_WEIGHTS, prediction):
        if p == target:
            return w
    return 0.


def mrr_iterables(predictions: Iterable[tuple[str, str, str, str]], targets: Iterable[str]) -> float:
    return np.mean(list(itertools.starmap(mrr_dp, zip(predictions, targets))))


def mrr_files(predictions_path: Path, targets_path: Path) -> float:
    with open(predictions_path) as predictions_file, open(targets_path) as targets_file:
        predictions_reader = csv.reader(predictions_file)
        targets = (i.strip() for i in targets_file)
        # noinspection PyTypeChecker
        return mrr_iterables(predictions_reader, targets)


if __name__ == '__main__':
    import sys

    score = mrr_files(*sys.argv[1:])
    print(score)
