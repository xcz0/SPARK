"""Tests for the src.models.architecture module."""

from __future__ import annotations

import torch

from spark.models.architecture import ModelConfig, SPARKModel

_MODEL_INPUT_KEYS = (
    "numerical_features",
    "categorical_features",
    "time_stamps",
    "causal_mask",
    "card_mask",
    "deck_mask",
    "time_diff",
    "padding_mask",
)


def _extract_model_inputs(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Select the tensors required by SPARKModel.forward."""
    return {key: batch[key] for key in _MODEL_INPUT_KEYS}


def test_model_config_defaults():
    """ModelConfig should populate sensible defaults when not provided."""
    config = ModelConfig()

    assert config.d_ff == 4 * config.d_model
    assert config.categorical_vocab_sizes == {
        "state": 4,
        "prev_review_rating": 5,
        "is_first_review": 2,
    }
    assert config.num_rating_classes == 4


def test_model_config_respects_overrides():
    """Explicit kwargs should remain unchanged after initialization."""
    custom_vocab = {"state": 3, "difficulty": 5}
    config = ModelConfig(categorical_vocab_sizes=custom_vocab, d_ff=256)

    assert config.categorical_vocab_sizes is custom_vocab
    assert config.d_ff == 256


def test_spark_model_updates_existing_config():
    """Passing kwargs with an existing config should override its fields."""
    base_config = ModelConfig(dropout=0.2)
    model = SPARKModel(config=base_config, dropout=0.05)

    assert model.config.dropout == 0.05


def test_spark_model_forward_outputs_have_expected_shapes(
    small_model_config: ModelConfig, dummy_batch: dict[str, torch.Tensor]
):
    """Forward pass should return all tensors with consistent shapes."""
    model = SPARKModel(config=small_model_config)
    model_inputs = _extract_model_inputs(dummy_batch)

    outputs = model(**model_inputs)

    batch_size, seq_len = dummy_batch["numerical_features"].shape[:2]
    num_thresholds = small_model_config.num_rating_classes - 1

    assert outputs["rating_probs"].shape == (batch_size, seq_len, num_thresholds)
    assert outputs["duration_pred"].shape == (batch_size, seq_len)
    assert outputs["hidden_states"].shape == (
        batch_size,
        seq_len,
        small_model_config.d_model,
    )
    probs = outputs["rating_probs"]
    assert torch.all((probs >= 0.0) & (probs <= 1.0))


def test_predict_helpers_match_manual_decoding(
    small_model_config: ModelConfig, dummy_batch: dict[str, torch.Tensor]
):
    """Helper methods should match the direct decoding formulas."""
    model = SPARKModel(config=small_model_config)
    model.eval()
    model_inputs = _extract_model_inputs(dummy_batch)

    with torch.no_grad():
        forward_outputs = model(**model_inputs)
    manual_ratings = 1 + (forward_outputs["rating_probs"] > 0.5).sum(dim=-1)
    manual_expectation = 1.0 + forward_outputs["rating_probs"].sum(dim=-1)

    rating_preds = model.predict_rating(forward_outputs["rating_probs"])
    expected_preds = model.predict_expected_rating(forward_outputs["rating_probs"])

    assert torch.equal(rating_preds, manual_ratings)
    assert torch.allclose(expected_preds, manual_expectation, atol=1e-5)


class TestFeatureNames:
    """Tests for feature name properties."""

    def test_numerical_feature_names_default_is_none(self):
        """numerical_feature_names should be None when not specified."""
        model = SPARKModel()
        assert model.numerical_feature_names is None

    def test_numerical_feature_names_from_config(self):
        """numerical_feature_names should return value from config."""
        feature_names = ("delta_t", "r_history_cnt", "n_learning_cnt")
        config = ModelConfig(numerical_feature_names=feature_names)
        model = SPARKModel(config=config)

        assert model.numerical_feature_names == feature_names

    def test_numerical_feature_names_from_kwargs(self):
        """numerical_feature_names should be settable via kwargs."""
        feature_names = ("feature_a", "feature_b")
        model = SPARKModel(numerical_feature_names=feature_names)

        assert model.numerical_feature_names == feature_names

    def test_categorical_feature_names_from_vocab_keys(self):
        """categorical_feature_names should return vocab dict keys."""
        vocab = {"state": 4, "difficulty": 5, "is_new": 2}
        model = SPARKModel(categorical_vocab_sizes=vocab)

        assert model.categorical_feature_names == tuple(vocab.keys())

    def test_time_feature_name_default(self):
        """time_feature_name should default to 'day_offset'."""
        model = SPARKModel()
        assert model.time_feature_name == "day_offset"

    def test_time_feature_name_custom(self):
        """time_feature_name should be customizable."""
        model = SPARKModel(time_feature_name="timestamp")
        assert model.time_feature_name == "timestamp"

    def test_all_feature_names_structure(self):
        """all_feature_names should return dict with all feature categories."""
        numerical = ("feat1", "feat2")
        vocab = {"cat1": 3, "cat2": 5}
        time_name = "my_time"

        model = SPARKModel(
            numerical_feature_names=numerical,
            categorical_vocab_sizes=vocab,
            time_feature_name=time_name,
        )

        result = model.all_feature_names

        assert result["numerical"] == numerical
        assert result["categorical"] == tuple(vocab.keys())
        assert result["time"] == time_name
