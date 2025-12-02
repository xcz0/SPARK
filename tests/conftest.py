"""Pytest fixtures shared across model tests."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.architecture import ModelConfig


@pytest.fixture
def small_model_config() -> ModelConfig:
    """Provide a lightweight ModelConfig for unit tests."""
    return ModelConfig(
        d_model=32,
        n_heads=4,
        depth=2,
        dropout=0.0,
        num_numerical_features=6,
        categorical_embed_dim=8,
        num_rating_classes=4,
        card_head_ratio=0.5,
        deck_head_ratio=0.25,
    )


@pytest.fixture
def dummy_batch(small_model_config: ModelConfig) -> dict[str, torch.Tensor]:
    """Create a synthetic batch matching the model interface."""
    torch.manual_seed(42)
    batch_size, seq_len = 2, 5
    config = small_model_config

    numerical_features = torch.randn(batch_size, seq_len, config.num_numerical_features)

    categorical_features = torch.stack(
        [
            torch.randint(0, vocab_size, (batch_size, seq_len))
            for vocab_size in config.categorical_vocab_sizes.values()
        ],
        dim=-1,
    )

    time_stamps = torch.linspace(0.0, float(seq_len - 1), steps=seq_len)
    time_stamps = time_stamps.repeat(batch_size, 1)

    causal_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))
    card_mask = causal_mask.unsqueeze(0).repeat(batch_size, 1, 1)
    deck_mask = card_mask.clone()

    time_diff = torch.abs(time_stamps.unsqueeze(2) - time_stamps.unsqueeze(1))
    padding_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

    num_classes = config.num_rating_classes
    ratings = torch.arange(batch_size * seq_len, dtype=torch.int64)
    ratings = (ratings % num_classes + 1).view(batch_size, seq_len)
    thresholds = torch.arange(1, config.num_rating_classes).view(1, 1, -1)
    ordinal_targets = (ratings.unsqueeze(-1) > thresholds).float()

    duration_targets = torch.rand(batch_size, seq_len)

    return {
        "numerical_features": numerical_features,
        "categorical_features": categorical_features,
        "time_stamps": time_stamps,
        "causal_mask": causal_mask,
        "card_mask": card_mask,
        "deck_mask": deck_mask,
        "time_diff": time_diff,
        "padding_mask": padding_mask,
        "ordinal_targets": ordinal_targets,
        "duration_targets": duration_targets,
    }
