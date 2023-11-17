import numpy as np

import utils


class PopularityCalculator:
    def __init__(self, vocabulary: utils.Vocabulary):
        self.vocabulary = vocabulary

    def __call__(self, trace: utils.Trace, candidates: list[utils.Candidate]) -> np.ndarray:
        return np.array([
            [self.vocabulary.word_counts[c.word] for c in cs]
            for cs in candidates
        ])
