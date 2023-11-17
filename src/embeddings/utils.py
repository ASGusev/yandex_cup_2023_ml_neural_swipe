import numpy as np
import torch
from torch import nn
from tqdm.asyncio import trange

import utils
from embeddings.data import TracePreprocessor


def calculate_word_embeddings(
        model: nn.Module,
        words: list[str],
        keyboard_grids: dict[str, utils.KeyboardGrid],
        preprocessor: TracePreprocessor,
        batch_size: int = 1000,
) -> dict[str, np.ndarray]:
    embeddings = {}
    with torch.no_grad():
        for grid_name, _ in keyboard_grids.items():
            grid_embeddings = []
            # noinspection PyTypeChecker
            for start_index in trange(0, len(words), batch_size, leave=False, desc=grid_name):
                end_index = min(len(words), start_index + batch_size)
                curves = [
                    preprocessor.preprocess_proj(w, grid_name)
                    for w in words[start_index:end_index]
                ]
                batch_embeddings = model(preprocessor.merge_batch(curves))
                grid_embeddings.append(batch_embeddings.cpu().numpy())
            embeddings[grid_name] = np.concatenate(grid_embeddings)
    return embeddings
