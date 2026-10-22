"""The network: a GRU encoder-decoder emitting one quantile path per series.

The encoder reads the history as two channels, the level and its annual
change (zero at the first year, divided by ``diff_scale``), and its final
state initialises the decoder. Starting from the last level of the history,
the decoder runs one step per forecast year: each step receives the previous
predicted median and the step as a fraction of the horizon, and emits one
value per quantile. Everything is on the model scale.
"""
import torch
from torch import nn

from .training import QUANTILES

MEDIAN = QUANTILES.index(0.5)


class NeuralTFR(nn.Module):
    def __init__(self, horizon: int, hidden_size: int = 16, diff_scale: float = 1.0) -> None:
        super().__init__()
        self.horizon = horizon
        self.diff_scale = diff_scale
        # Attribute order is part of the specification: a seeded initialisation
        # draws the parameters in it.
        self.decoder = nn.GRU(2, hidden_size, batch_first=True)
        self.output = nn.Linear(hidden_size, len(QUANTILES))
        self.encoder = nn.GRU(2, hidden_size, batch_first=True)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        """``history`` is (batch, years); the result is (batch, horizon, quantiles)."""
        change = torch.zeros_like(history)
        change[:, 1:] = history[:, 1:] - history[:, :-1]
        _, state = self.encoder(torch.stack([history, change / self.diff_scale], dim=-1))

        previous, steps = history[:, -1], []
        for step in range(self.horizon):
            position = torch.full_like(previous, step / self.horizon)
            output, state = self.decoder(torch.stack([previous, position], dim=-1).unsqueeze(1), state)
            quantiles = self.output(output.squeeze(1))
            steps.append(quantiles)
            previous = quantiles[:, MEDIAN]
        return torch.stack(steps, dim=1)


def save(model: NeuralTFR, path) -> None:
    torch.save({"state_dict": model.state_dict(), "horizon": model.horizon,
                "diff_scale": model.diff_scale}, path)


def load(path) -> NeuralTFR:
    """A saved model, ready for inference."""
    checkpoint = torch.load(path)
    model = NeuralTFR(checkpoint["horizon"], checkpoint["state_dict"]["output.weight"].shape[1],
                      checkpoint["diff_scale"])
    model.load_state_dict(checkpoint["state_dict"])
    return model.eval()
