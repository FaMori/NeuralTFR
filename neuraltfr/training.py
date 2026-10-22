"""Quantile loss and the training loop.

The network emits one value per quantile in ``QUANTILES`` for every target
year. Training minimises the mean quantile (pinball) loss over the reported
target cells of each batch, on the model scale, with Adam for a fixed number
of epochs. Which epoch to keep is decided outside this module.
"""
import torch

QUANTILES = (0.025, 0.1, 0.5, 0.9, 0.975)
COLUMNS = tuple(f"y_hat_{round(q * 1000):03d}" for q in QUANTILES)   # y_hat_025 ... y_hat_975


def quantile_loss(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean pinball loss over the masked cells of ``target`` and all quantiles.

    ``prediction`` is (batch, years, quantiles); ``target`` and ``mask`` are
    (batch, years). Masked cells contribute neither value nor gradient.
    """
    quantiles = torch.tensor(QUANTILES, dtype=prediction.dtype)
    error = prediction - target.unsqueeze(-1)
    loss = torch.maximum(quantiles * -error, (1.0 - quantiles) * error)
    return (loss * mask.unsqueeze(-1)).sum() / mask.sum() / len(QUANTILES)


def train(model: torch.nn.Module, loader, epochs: int, lr: float, weight_decay: float,
          after_epoch=None) -> None:
    """Fit ``model`` on the batches of ``loader`` for ``epochs`` epochs.

    ``after_epoch(epoch, model)``, if given, runs after every epoch and is how
    the caller scores checkpoints.
    """
    optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    for epoch in range(1, epochs + 1):
        model.train()
        for encoder, target, mask in loader:
            optimiser.zero_grad()
            quantile_loss(model(encoder), target, mask).backward()
            optimiser.step()
        if after_epoch is not None:
            after_epoch(epoch, model)
