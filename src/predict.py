"""Utility helpers for lightweight single-sequence prediction."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from .data import ReviewSequenceDataset, ReviewCollator
from src.models import SPARKModel


def _ensure_device(device: torch.device | str) -> torch.device:
    if isinstance(device, torch.device):
        return device
    return torch.device(device)


def _move_batch_to_device(
    batch: dict[str, torch.Tensor], device: torch.device
) -> dict[str, torch.Tensor]:
    return {key: tensor.to(device) for key, tensor in batch.items()}


def predict(
    model: SPARKModel,
    dataset: ReviewSequenceDataset,
    device: torch.device | str = "cpu",
) -> dict[str, torch.Tensor | float]:
    """Run a single-step prediction (last valid position) for one sequence."""

    if len(dataset) == 0:
        msg = "Cannot run prediction on an empty dataset."
        raise ValueError(msg)

    resolved_device = _ensure_device(device)
    model = model.to(resolved_device)
    model.eval()

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=ReviewCollator(),
        pin_memory=resolved_device.type == "cuda",
    )

    first_batch = next(iter(dataloader), None)
    if first_batch is None:
        msg = "Unable to materialize a batch for prediction."
        raise RuntimeError(msg)

    batch = _move_batch_to_device(first_batch, resolved_device)

    with torch.no_grad():
        outputs = model(
            numerical_features=batch["numerical_features"],
            categorical_features=batch["categorical_features"],
            time_stamps=batch["time_stamps"],
            causal_mask=batch["causal_mask"],
            card_mask=batch["card_mask"],
            deck_mask=batch["deck_mask"],
            time_diff=batch["time_diff"],
            padding_mask=batch["padding_mask"],
        )

    seq_len = int(batch["seq_lens"].item())
    last_index = seq_len - 1
    last_probs = outputs["rating_probs"][0, last_index]
    last_duration = outputs["duration_pred"][0, last_index]

    rating_pred = model.predict_rating(outputs["rating_probs"])[0, last_index]
    rating_expected = model.predict_expected_rating(outputs["rating_probs"])[
        0, last_index
    ]
    correct_prob = model.predict_correct(outputs["rating_probs"])[0, last_index]

    return {
        "rating_probs": last_probs.detach().cpu(),
        "rating_pred": float(rating_pred.item()),
        "rating_expected": float(rating_expected.item()),
        "recall_prob": float(correct_prob.item()),
        "duration_pred": float(last_duration.item()),
    }
