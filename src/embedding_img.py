import cv2
import numpy as np
import torch
from torch import nn

import utils
import embedding


class ImgPreprocessor(embedding.TracePreprocessor):
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


class ResidualLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.Tanh(),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.Tanh(),
        )
        self.shortcut = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        return self.layers(x) + self.shortcut(x)


class ConvImgEmbedder(embedding.Embedder):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            ResidualLayer(3, 32),
            nn.LayerNorm([32, 32, 32]),
            nn.MaxPool2d(2),

            ResidualLayer(32, 64),
            nn.LayerNorm([64, 16, 16]),
            nn.MaxPool2d(2),

            ResidualLayer(64, 64),
            nn.LayerNorm([64, 8, 8]),
            nn.MaxPool2d(2),

            ResidualLayer(64, 64),
            nn.LayerNorm([64, 4, 4]),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 64, 2),
            nn.Tanh(),
        )

    @property
    def device(self) -> str:
        return next(self.parameters()).device

    def forward(self, batch: dict) -> torch.Tensor:
        emb = self.layers(batch['x'].to(self.device))
        emb = emb[:, :, 0, 0]
        return nn.functional.normalize(emb, dim=1)


class MLPImgEmbedder(embedding.Embedder):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(1026, 768),
            nn.Tanh(),
            nn.LayerNorm(768),

            nn.Linear(768, 512, ),
            nn.Tanh(),
            nn.LayerNorm(512),

            nn.Linear(512, 256),
            nn.Tanh(),
            nn.LayerNorm(256),

            nn.Linear(256, 64),
        )

    def forward(self, batch: dict) -> torch.Tensor:
        x = batch['x'].to(self.device)
        x = torch.cat(
            [
                x[:, 0].reshape(-1, 1024),
                x[:, 1:].mean(axis=(2, 3)),
            ],
            dim=1,
        )
        emb = self.layers(x)
        return nn.functional.normalize(emb, dim=1)
