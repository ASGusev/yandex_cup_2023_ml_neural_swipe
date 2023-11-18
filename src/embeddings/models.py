import itertools
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
from torch import nn
from tqdm.asyncio import tqdm

import data
import utils


class Embedder(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, batch: dict) -> torch.Tensor:
        raise NotImplementedError

    @property
    def device(self) -> str:
        return next(self.parameters()).device


class LSTMEmbedder(Embedder):
    def __init__(self, in_dim: int = 5, out_dim: int = 64, n_layers: int = 4):
        super().__init__()
        self.rnn = nn.LSTM(in_dim, out_dim, n_layers, batch_first=True)

    def forward(self, batch: dict) -> torch.Tensor:
        x = nn.utils.rnn.pack_padded_sequence(batch['traces'], batch['lengths'], True, enforce_sorted=False)
        x = x.to(self.device)
        rnn_out, _ = self.rnn(x)
        padded_out, _ = nn.utils.rnn.pad_packed_sequence(rnn_out, True)
        res = padded_out[torch.arange(len(batch['lengths'])), batch['lengths'] - 1]
        return nn.functional.normalize(res, dim=1)


class PositionalEncoder(nn.Module):
    def __init__(self, dim: int = 3, max_size: int = 96):
        super().__init__()
        matrix = torch.arange(max_size).unsqueeze(1) * torch.arange(1, dim + 1) * (2 * torch.pi / max_size)
        embeddings = torch.cat((torch.sin(matrix), torch.cos(matrix)), dim=1)
        self.embeddings = torch.nn.Parameter(embeddings.unsqueeze(0), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.concatenate(
            (
                x,
                self.embeddings[:, :x.shape[1]].repeat(len(x), 1, 1),
            ),
            dim=2
        )


class TransformerEmbedder(Embedder):
    def __init__(
            self,
            in_dim: int = 5, emb_dim: int = 32,
            n_heads: int = 4, n_layers: int = 4,
            max_size: int = 96, pos_dim: int = 3
    ):
        super().__init__()
        self.positional_encoder = PositionalEncoder(pos_dim, max_size)
        self.proj = nn.Conv1d(in_dim + 2 * pos_dim, emb_dim, 1)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(emb_dim, n_heads, emb_dim, batch_first=True, activation='gelu'),
            n_layers
        )

    def forward(self, batch: dict) -> torch.Tensor:
        x = self.positional_encoder(batch['traces'].to(self.device))
        in_emb = self.proj(x.permute(0, 2, 1)).permute(0, 2, 1)
        mask = torch.ones(batch['traces'].shape[:2], dtype=torch.bool)
        for i, sl in enumerate(batch['lengths']):
            mask[i, :sl] = False
        mask = mask.to(self.device)
        embeddings = self.transformer(in_emb, src_key_padding_mask=mask)[:, 0]
        return nn.functional.normalize(embeddings, dim=1)


def center_dist_penalty(embeddings: torch.Tensor) -> torch.Tensor:
    distances = torch.norm(embeddings - embeddings.mean(axis=0), dim=1)
    return torch.mean((distances - 1) ** 2)


# noinspection PyTypeChecker
# noinspection PyUnresolvedReferences
def train(
        model: Embedder, opt: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler.LRScheduler,
        train_data: data.Dataset, val_data: data.Dataset,
        batches_per_epoch: int, n_epochs: int,
        loss_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        penalty_function: Callable[[torch.Tensor], torch.Tensor], penalty_weight: float = 0.,
        saving_path: Optional[Path] = None,
):
    train_iterator = utils.cycle(train_data)
    best_val_loss = float('inf')
    for epoch in range(1, n_epochs + 1):
        train_losses, val_losses = [], []
        train_penalties, val_penalties = [], []
        model.train()
        for batch in tqdm(
                itertools.islice(train_iterator, batches_per_epoch),
                total=batches_per_epoch,
                leave=False
        ):
            opt.zero_grad()
            embeddings = model(batch)
            loss = loss_function(embeddings, indices_tuple=batch['indices_tuple'])
            penalty = penalty_function(embeddings) if penalty_weight else torch.tensor(0)
            total_loss = loss + penalty * penalty_weight
            total_loss.backward()
            opt.step()
            train_losses.append(loss.item())
            train_penalties.append(penalty.item())
            del total_loss, penalty, loss, embeddings
        model.eval()
        with torch.no_grad():
            for batch in tqdm(val_data, leave=False):
                embeddings = model(batch)
                loss = loss_function(embeddings, indices_tuple=batch['indices_tuple'])
                val_losses.append(loss.item())
        train_loss = np.mean(train_losses)
        train_penalty = np.mean(train_penalties)
        val_loss = np.mean(val_losses)
        print(f'Epoch {epoch} train loss: {train_loss:.3f} train penalty: {train_penalty:.3f} val loss: {val_loss:.3f}')
        if scheduler:
            scheduler.step()
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if saving_path:
                torch.save(model.state_dict(), saving_path)


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


class ConvImgEmbedder(Embedder):
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


class MLPImgEmbedder(Embedder):
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
