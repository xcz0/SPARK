"""Tests for the lightweight prediction helper."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from spark.data.config import FeatureConfig
from spark.data.dataset import ReviewSequenceDataset
from spark.models.architecture import ModelConfig, SPARKModel
from spark.predict import predict

TEST_FEATURE_CONFIG = FeatureConfig(
    numerical_features=(
        "log_elapsed_seconds",
        "log_elapsed_days",
        "log_time_spent_today",
        "log_prev_review_duration",
        "log_last_duration_on_card",
        "card_index_today",
    ),
    categorical_features=(
        "state",
        "prev_review_rating",
        "is_first_review",
    ),
    categorical_vocab_sizes={
        "state": 4,
        "prev_review_rating": 5,
        "is_first_review": 2,
    },
    ordinal_targets=("rating_gt1", "rating_gt2", "rating_gt3"),
    time_feature="day_offset",
    card_id_feature="card_id",
    deck_id_feature="deck_id",
    duration_target="log_duration",
)


def _build_sequence_dict(length: int) -> dict[str, np.ndarray]:
    base = np.arange(length, dtype=np.float32)
    base_int = base.astype(np.int64)

    numerical = {
        name: (base + idx + 1.0).astype(np.float32)
        for idx, name in enumerate(TEST_FEATURE_CONFIG.numerical_features)
    }

    categorical = {
        "state": (base_int % 4).astype(np.int64),
        "prev_review_rating": (base_int % 5).astype(np.int64),
        "is_first_review": (base_int % 2).astype(np.int64),
    }

    targets = {
        "rating_gt1": np.ones(length, dtype=np.float32),
        "rating_gt2": (base > 0).astype(np.float32),
        "rating_gt3": (base > 1).astype(np.float32),
        "log_duration": np.log1p(base).astype(np.float32),
    }

    ids = {
        "day_offset": base.astype(np.float32),
        "card_id": np.ones(length, dtype=np.int64),
        "deck_id": np.full(length, 2, dtype=np.int64),
        "user_id": np.zeros(length, dtype=np.int64),
    }

    return numerical | categorical | targets | ids


def _make_dataset(length: int, seq_len: int | None = None) -> ReviewSequenceDataset:
    return ReviewSequenceDataset(
        data=_build_sequence_dict(length),
        feature_config=TEST_FEATURE_CONFIG,
        seq_len=seq_len or length,
        stride=seq_len or length,
    )


def test_predict_returns_last_step_outputs(small_model_config: ModelConfig) -> None:
    model = SPARKModel(config=small_model_config)
    dataset = _make_dataset(length=4)

    result = predict(model, dataset, device="cpu")

    expected_keys = {
        "rating_probs",
        "rating_pred",
        "rating_expected",
        "recall_prob",
        "duration_pred",
    }
    assert expected_keys <= result.keys()

    rating_probs = result["rating_probs"]
    assert isinstance(rating_probs, torch.Tensor)

    num_thresholds = small_model_config.num_rating_classes - 1
    assert rating_probs.shape == (num_thresholds,)

    assert 1.0 <= result["rating_pred"] <= small_model_config.num_rating_classes
    assert result["rating_expected"] >= 1.0
    assert 0.0 <= result["recall_prob"] <= 1.0
    assert np.isfinite(result["duration_pred"])


def test_predict_raises_for_empty_dataset(small_model_config: ModelConfig) -> None:
    model = SPARKModel(config=small_model_config)
    dataset = _make_dataset(length=1, seq_len=4)
    assert len(dataset) == 0

    with pytest.raises(ValueError, match="empty dataset"):
        predict(model, dataset)
