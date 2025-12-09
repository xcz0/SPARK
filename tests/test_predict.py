"""Tests for spark.predict module."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from spark.models import SPARKModel
from spark.models.architecture import ModelConfig
from spark.predict import (
    _prepare_batch,
    _update_sequence_for_next_step,
    predict,
    predict_batch,
)


# ============== Fixtures ==============


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
def model(small_model_config: ModelConfig) -> SPARKModel:
    """Create a small SPARK model for testing."""
    return SPARKModel(config=small_model_config)


@pytest.fixture
def numerical_features() -> list[str]:
    """Default numerical feature names."""
    return [
        "interval",
        "log_interval",
        "n_reviews",
        "n_success",
        "n_failure",
        "log_duration",
    ]


@pytest.fixture
def categorical_features() -> list[str]:
    """Default categorical feature names."""
    return ["state", "prev_review_rating", "is_first_review"]


@pytest.fixture
def sample_sequence(
    numerical_features: list[str], categorical_features: list[str]
) -> dict[str, np.ndarray]:
    """Create a sample review sequence for testing."""
    seq_len = 5
    np.random.seed(42)

    data: dict[str, np.ndarray] = {}

    # Numerical features
    data["interval"] = np.array([0, 1, 3, 7, 14], dtype=np.float32)
    data["log_interval"] = np.log1p(data["interval"])
    data["n_reviews"] = np.arange(1, seq_len + 1, dtype=np.float32)
    data["n_success"] = np.array([0, 1, 2, 3, 4], dtype=np.float32)
    data["n_failure"] = np.array([0, 0, 0, 0, 0], dtype=np.float32)
    data["log_duration"] = np.random.rand(seq_len).astype(np.float32)

    # Categorical features
    data["state"] = np.array([1, 2, 2, 2, 2], dtype=np.int64)
    data["prev_review_rating"] = np.array([0, 3, 3, 4, 3], dtype=np.int64)
    data["is_first_review"] = np.array([1, 0, 0, 0, 0], dtype=np.int64)

    # Time and ID features
    data["day_offset"] = np.cumsum(data["interval"]).astype(np.float32)
    data["card_id"] = np.ones(seq_len, dtype=np.int64) * 123
    data["deck_id"] = np.ones(seq_len, dtype=np.int64) * 456
    data["user_id"] = np.ones(seq_len, dtype=np.int64) * 789

    return data


# ============== Tests for _prepare_batch ==============


class TestPrepareBatch:
    """Tests for _prepare_batch function."""

    def test_output_keys(
        self,
        sample_sequence: dict[str, np.ndarray],
        numerical_features: list[str],
        categorical_features: list[str],
    ):
        """Test that output contains all required keys."""
        batch = _prepare_batch(
            sample_sequence, numerical_features, categorical_features
        )

        expected_keys = {
            "numerical_features",
            "categorical_features",
            "time_stamps",
            "causal_mask",
            "card_mask",
            "deck_mask",
            "time_diff",
            "padding_mask",
            "card_ids",
            "deck_ids",
        }
        assert set(batch.keys()) == expected_keys

    def test_output_shapes(
        self,
        sample_sequence: dict[str, np.ndarray],
        numerical_features: list[str],
        categorical_features: list[str],
    ):
        """Test that output tensors have correct shapes."""
        batch = _prepare_batch(
            sample_sequence, numerical_features, categorical_features
        )

        seq_len = len(sample_sequence["day_offset"])
        num_numerical = len(numerical_features)
        num_categorical = len(categorical_features)

        assert batch["numerical_features"].shape == (1, seq_len, num_numerical)
        assert batch["categorical_features"].shape == (1, seq_len, num_categorical)
        assert batch["time_stamps"].shape == (1, seq_len)
        assert batch["causal_mask"].shape == (seq_len, seq_len)
        assert batch["card_mask"].shape == (1, seq_len, seq_len)
        assert batch["deck_mask"].shape == (1, seq_len, seq_len)
        assert batch["time_diff"].shape == (1, seq_len, seq_len)
        assert batch["padding_mask"].shape == (1, seq_len)
        assert batch["card_ids"].shape == (1, seq_len)
        assert batch["deck_ids"].shape == (1, seq_len)

    def test_output_dtypes(
        self,
        sample_sequence: dict[str, np.ndarray],
        numerical_features: list[str],
        categorical_features: list[str],
    ):
        """Test that output tensors have correct dtypes."""
        batch = _prepare_batch(
            sample_sequence, numerical_features, categorical_features
        )

        assert batch["numerical_features"].dtype == torch.float32
        assert batch["categorical_features"].dtype == torch.int64
        assert batch["time_stamps"].dtype == torch.float32
        assert batch["causal_mask"].dtype == torch.bool
        assert batch["card_mask"].dtype == torch.bool
        assert batch["deck_mask"].dtype == torch.bool
        assert batch["time_diff"].dtype == torch.float32
        assert batch["padding_mask"].dtype == torch.bool
        assert batch["card_ids"].dtype == torch.int64
        assert batch["deck_ids"].dtype == torch.int64

    def test_causal_mask_is_lower_triangular(
        self,
        sample_sequence: dict[str, np.ndarray],
        numerical_features: list[str],
        categorical_features: list[str],
    ):
        """Test that causal mask is lower triangular."""
        batch = _prepare_batch(
            sample_sequence, numerical_features, categorical_features
        )

        causal_mask = batch["causal_mask"]
        seq_len = causal_mask.shape[0]

        # Check lower triangular structure
        for i in range(seq_len):
            for j in range(seq_len):
                if j <= i:
                    assert causal_mask[i, j].item() is True
                else:
                    assert causal_mask[i, j].item() is False

    def test_custom_feature_names(
        self,
        sample_sequence: dict[str, np.ndarray],
    ):
        """Test with custom time/card/deck feature names."""
        # Add custom named features
        sample_sequence["custom_time"] = sample_sequence["day_offset"].copy()
        sample_sequence["custom_card"] = sample_sequence["card_id"].copy()
        sample_sequence["custom_deck"] = sample_sequence["deck_id"].copy()

        batch = _prepare_batch(
            sample_sequence,
            numerical_features=["interval", "log_interval"],
            categorical_features=["state"],
            time_feature="custom_time",
            card_id_feature="custom_card",
            deck_id_feature="custom_deck",
        )

        assert batch["time_stamps"].shape == (1, 5)
        assert batch["card_ids"].shape == (1, 5)
        assert batch["deck_ids"].shape == (1, 5)


# ============== Tests for _update_sequence_for_next_step ==============


class TestUpdateSequenceForNextStep:
    """Tests for _update_sequence_for_next_step function."""

    def test_sequence_length_increases(
        self,
        sample_sequence: dict[str, np.ndarray],
        numerical_features: list[str],
        categorical_features: list[str],
    ):
        """Test that sequence length increases by 1."""
        original_len = len(sample_sequence["day_offset"])
        updated = _update_sequence_for_next_step(
            sample_sequence,
            predicted_rating=3,
            next_day_offset=50.0,
            numerical_features=numerical_features,
            categorical_features=categorical_features,
        )

        for key in updated:
            assert len(updated[key]) == original_len + 1

    def test_time_feature_updated(
        self,
        sample_sequence: dict[str, np.ndarray],
        numerical_features: list[str],
        categorical_features: list[str],
    ):
        """Test that time feature is updated correctly."""
        next_day_offset = 50.0
        updated = _update_sequence_for_next_step(
            sample_sequence,
            predicted_rating=3,
            next_day_offset=next_day_offset,
            numerical_features=numerical_features,
            categorical_features=categorical_features,
        )

        assert updated["day_offset"][-1] == pytest.approx(next_day_offset)

    def test_interval_calculated_correctly(
        self,
        sample_sequence: dict[str, np.ndarray],
        numerical_features: list[str],
        categorical_features: list[str],
    ):
        """Test that interval is calculated correctly."""
        last_day = sample_sequence["day_offset"][-1]
        next_day_offset = last_day + 7.0

        updated = _update_sequence_for_next_step(
            sample_sequence,
            predicted_rating=3,
            next_day_offset=next_day_offset,
            numerical_features=numerical_features,
            categorical_features=categorical_features,
        )

        assert updated["interval"][-1] == pytest.approx(7.0)
        assert updated["log_interval"][-1] == pytest.approx(np.log1p(7.0))

    def test_n_reviews_incremented(
        self,
        sample_sequence: dict[str, np.ndarray],
        numerical_features: list[str],
        categorical_features: list[str],
    ):
        """Test that n_reviews is incremented."""
        last_n_reviews = sample_sequence["n_reviews"][-1]

        updated = _update_sequence_for_next_step(
            sample_sequence,
            predicted_rating=3,
            next_day_offset=50.0,
            numerical_features=numerical_features,
            categorical_features=categorical_features,
        )

        assert updated["n_reviews"][-1] == last_n_reviews + 1

    @pytest.mark.parametrize(
        "rating,expected_success_delta,expected_failure_delta",
        [
            (1, 0, 1),  # Again - failure
            (2, 1, 0),  # Hard - success
            (3, 1, 0),  # Good - success
            (4, 1, 0),  # Easy - success
        ],
    )
    def test_success_failure_counters(
        self,
        sample_sequence: dict[str, np.ndarray],
        numerical_features: list[str],
        categorical_features: list[str],
        rating: int,
        expected_success_delta: int,
        expected_failure_delta: int,
    ):
        """Test that success/failure counters are updated based on rating."""
        last_n_success = sample_sequence["n_success"][-1]
        last_n_failure = sample_sequence["n_failure"][-1]

        updated = _update_sequence_for_next_step(
            sample_sequence,
            predicted_rating=rating,
            next_day_offset=50.0,
            numerical_features=numerical_features,
            categorical_features=categorical_features,
        )

        assert updated["n_success"][-1] == last_n_success + expected_success_delta
        assert updated["n_failure"][-1] == last_n_failure + expected_failure_delta

    def test_prev_review_rating_updated(
        self,
        sample_sequence: dict[str, np.ndarray],
        numerical_features: list[str],
        categorical_features: list[str],
    ):
        """Test that prev_review_rating is set to predicted rating."""
        predicted_rating = 4
        updated = _update_sequence_for_next_step(
            sample_sequence,
            predicted_rating=predicted_rating,
            next_day_offset=50.0,
            numerical_features=numerical_features,
            categorical_features=categorical_features,
        )

        assert updated["prev_review_rating"][-1] == predicted_rating

    @pytest.mark.parametrize(
        "rating,expected_state",
        [
            (1, 1),  # Again -> Learning
            (2, 2),  # Hard -> Review
            (3, 2),  # Good -> Review
            (4, 2),  # Easy -> Review
        ],
    )
    def test_state_updated_based_on_rating(
        self,
        sample_sequence: dict[str, np.ndarray],
        numerical_features: list[str],
        categorical_features: list[str],
        rating: int,
        expected_state: int,
    ):
        """Test that state is updated based on rating."""
        updated = _update_sequence_for_next_step(
            sample_sequence,
            predicted_rating=rating,
            next_day_offset=50.0,
            numerical_features=numerical_features,
            categorical_features=categorical_features,
        )

        assert updated["state"][-1] == expected_state

    def test_is_first_review_set_to_zero(
        self,
        sample_sequence: dict[str, np.ndarray],
        numerical_features: list[str],
        categorical_features: list[str],
    ):
        """Test that is_first_review is set to 0 for new record."""
        updated = _update_sequence_for_next_step(
            sample_sequence,
            predicted_rating=3,
            next_day_offset=50.0,
            numerical_features=numerical_features,
            categorical_features=categorical_features,
        )

        assert updated["is_first_review"][-1] == 0

    def test_ids_preserved(
        self,
        sample_sequence: dict[str, np.ndarray],
        numerical_features: list[str],
        categorical_features: list[str],
    ):
        """Test that card_id, deck_id, user_id are preserved."""
        updated = _update_sequence_for_next_step(
            sample_sequence,
            predicted_rating=3,
            next_day_offset=50.0,
            numerical_features=numerical_features,
            categorical_features=categorical_features,
        )

        assert updated["card_id"][-1] == sample_sequence["card_id"][-1]
        assert updated["deck_id"][-1] == sample_sequence["deck_id"][-1]
        assert updated["user_id"][-1] == sample_sequence["user_id"][-1]

    def test_original_data_not_modified(
        self,
        sample_sequence: dict[str, np.ndarray],
        numerical_features: list[str],
        categorical_features: list[str],
    ):
        """Test that original data is not modified."""
        original_len = len(sample_sequence["day_offset"])
        original_values = {k: v.copy() for k, v in sample_sequence.items()}

        _ = _update_sequence_for_next_step(
            sample_sequence,
            predicted_rating=3,
            next_day_offset=50.0,
            numerical_features=numerical_features,
            categorical_features=categorical_features,
        )

        for key in sample_sequence:
            assert len(sample_sequence[key]) == original_len
            np.testing.assert_array_equal(sample_sequence[key], original_values[key])


# ============== Tests for predict ==============


class TestPredict:
    """Tests for predict function."""

    def test_single_step_prediction_output_keys(
        self,
        model: SPARKModel,
        sample_sequence: dict[str, np.ndarray],
        numerical_features: list[str],
        categorical_features: list[str],
    ):
        """Test single step prediction returns all expected keys."""
        result = predict(
            model, sample_sequence, numerical_features, categorical_features
        )

        expected_keys = {
            "rating",
            "expected_rating",
            "correct_prob",
            "duration",
            "rating_probs",
        }
        assert set(result.keys()) == expected_keys

    def test_single_step_prediction_shapes(
        self,
        model: SPARKModel,
        sample_sequence: dict[str, np.ndarray],
        numerical_features: list[str],
        categorical_features: list[str],
    ):
        """Test single step prediction output shapes."""
        result = predict(
            model, sample_sequence, numerical_features, categorical_features
        )

        assert result["rating"].shape == (1,)
        assert result["expected_rating"].shape == (1,)
        assert result["correct_prob"].shape == (1,)
        assert result["duration"].shape == (1,)
        # num_thresholds = num_rating_classes - 1 = 3
        assert result["rating_probs"].shape == (1, 3)

    def test_rating_in_valid_range(
        self,
        model: SPARKModel,
        sample_sequence: dict[str, np.ndarray],
        numerical_features: list[str],
        categorical_features: list[str],
    ):
        """Test that predicted rating is in valid range [1, 4]."""
        result = predict(
            model, sample_sequence, numerical_features, categorical_features
        )

        assert 1 <= result["rating"][0] <= 4

    def test_correct_prob_in_valid_range(
        self,
        model: SPARKModel,
        sample_sequence: dict[str, np.ndarray],
        numerical_features: list[str],
        categorical_features: list[str],
    ):
        """Test that correct_prob is in valid range [0, 1]."""
        result = predict(
            model, sample_sequence, numerical_features, categorical_features
        )

        assert 0.0 <= result["correct_prob"][0] <= 1.0

    def test_multi_step_prediction(
        self,
        model: SPARKModel,
        sample_sequence: dict[str, np.ndarray],
        numerical_features: list[str],
        categorical_features: list[str],
    ):
        """Test multi-step prediction with custom intervals."""
        steps = 3
        intervals = [1.0, 3.0, 7.0]

        result = predict(
            model,
            sample_sequence,
            numerical_features,
            categorical_features,
            options={"steps": steps, "intervals": intervals},
        )

        assert result["rating"].shape == (steps,)
        assert result["expected_rating"].shape == (steps,)
        assert result["correct_prob"].shape == (steps,)
        assert result["duration"].shape == (steps,)
        assert result["rating_probs"].shape == (steps, 3)

    def test_output_type_rating_only(
        self,
        model: SPARKModel,
        sample_sequence: dict[str, np.ndarray],
        numerical_features: list[str],
        categorical_features: list[str],
    ):
        """Test output_type='rating' returns only rating."""
        result = predict(
            model,
            sample_sequence,
            numerical_features,
            categorical_features,
            options={"output_type": "rating"},
        )

        assert set(result.keys()) == {"rating"}

    def test_output_type_expected_rating_only(
        self,
        model: SPARKModel,
        sample_sequence: dict[str, np.ndarray],
        numerical_features: list[str],
        categorical_features: list[str],
    ):
        """Test output_type='expected_rating' returns only expected_rating."""
        result = predict(
            model,
            sample_sequence,
            numerical_features,
            categorical_features,
            options={"output_type": "expected_rating"},
        )

        assert set(result.keys()) == {"expected_rating"}

    def test_output_type_correct_prob_only(
        self,
        model: SPARKModel,
        sample_sequence: dict[str, np.ndarray],
        numerical_features: list[str],
        categorical_features: list[str],
    ):
        """Test output_type='correct_prob' returns only correct_prob."""
        result = predict(
            model,
            sample_sequence,
            numerical_features,
            categorical_features,
            options={"output_type": "correct_prob"},
        )

        assert set(result.keys()) == {"correct_prob"}

    def test_intervals_auto_extend(
        self,
        model: SPARKModel,
        sample_sequence: dict[str, np.ndarray],
        numerical_features: list[str],
        categorical_features: list[str],
    ):
        """Test that intervals list is auto-extended if shorter than steps."""
        steps = 5
        intervals = [1.0, 2.0]  # Only 2 intervals for 5 steps

        result = predict(
            model,
            sample_sequence,
            numerical_features,
            categorical_features,
            options={"steps": steps, "intervals": intervals},
        )

        # Should complete without error and return correct shapes
        assert result["rating"].shape == (steps,)

    def test_default_intervals(
        self,
        model: SPARKModel,
        sample_sequence: dict[str, np.ndarray],
        numerical_features: list[str],
        categorical_features: list[str],
    ):
        """Test that default intervals (1.0 per step) work correctly."""
        steps = 3

        result = predict(
            model,
            sample_sequence,
            numerical_features,
            categorical_features,
            options={"steps": steps},
        )

        assert result["rating"].shape == (steps,)

    def test_custom_feature_names_in_options(
        self,
        model: SPARKModel,
        sample_sequence: dict[str, np.ndarray],
        numerical_features: list[str],
        categorical_features: list[str],
    ):
        """Test custom feature names via options."""
        # Add alternative named features
        sample_sequence["time"] = sample_sequence["day_offset"].copy()
        sample_sequence["cid"] = sample_sequence["card_id"].copy()
        sample_sequence["did"] = sample_sequence["deck_id"].copy()

        result = predict(
            model,
            sample_sequence,
            numerical_features,
            categorical_features,
            options={
                "time_feature": "time",
                "card_id_feature": "cid",
                "deck_id_feature": "did",
            },
        )

        assert "rating" in result


# ============== Tests for predict_batch ==============


class TestPredictBatch:
    """Tests for predict_batch function."""

    def test_batch_prediction_returns_list(
        self,
        model: SPARKModel,
        sample_sequence: dict[str, np.ndarray],
        numerical_features: list[str],
        categorical_features: list[str],
    ):
        """Test that batch prediction returns a list."""
        sequences = [sample_sequence, sample_sequence]

        results = predict_batch(
            model, sequences, numerical_features, categorical_features
        )

        assert isinstance(results, list)
        assert len(results) == 2

    def test_batch_prediction_each_result_has_correct_keys(
        self,
        model: SPARKModel,
        sample_sequence: dict[str, np.ndarray],
        numerical_features: list[str],
        categorical_features: list[str],
    ):
        """Test that each result in batch has correct keys."""
        sequences = [sample_sequence, sample_sequence]

        results = predict_batch(
            model, sequences, numerical_features, categorical_features
        )

        expected_keys = {
            "rating",
            "expected_rating",
            "correct_prob",
            "duration",
            "rating_probs",
        }
        for result in results:
            assert set(result.keys()) == expected_keys

    def test_batch_prediction_with_different_sequences(
        self,
        model: SPARKModel,
        sample_sequence: dict[str, np.ndarray],
        numerical_features: list[str],
        categorical_features: list[str],
    ):
        """Test batch prediction with different length sequences."""
        # Create a shorter sequence
        short_sequence = {k: v[:3] for k, v in sample_sequence.items()}

        sequences = [sample_sequence, short_sequence]

        results = predict_batch(
            model, sequences, numerical_features, categorical_features
        )

        assert len(results) == 2
        # Both should have valid predictions
        assert results[0]["rating"].shape == (1,)
        assert results[1]["rating"].shape == (1,)

    def test_batch_prediction_with_options(
        self,
        model: SPARKModel,
        sample_sequence: dict[str, np.ndarray],
        numerical_features: list[str],
        categorical_features: list[str],
    ):
        """Test batch prediction with options passed through."""
        sequences = [sample_sequence, sample_sequence]

        results = predict_batch(
            model,
            sequences,
            numerical_features,
            categorical_features,
            options={"steps": 2, "output_type": "rating"},
        )

        for result in results:
            assert set(result.keys()) == {"rating"}
            assert result["rating"].shape == (2,)

    def test_empty_batch(
        self,
        model: SPARKModel,
        numerical_features: list[str],
        categorical_features: list[str],
    ):
        """Test empty batch returns empty list."""
        results = predict_batch(model, [], numerical_features, categorical_features)

        assert results == []


# ============== Integration Tests ==============


class TestIntegration:
    """Integration tests for the predict module."""

    def test_multi_step_prediction_uses_previous_predictions(
        self,
        model: SPARKModel,
        sample_sequence: dict[str, np.ndarray],
        numerical_features: list[str],
        categorical_features: list[str],
    ):
        """Test that multi-step prediction builds on previous predictions."""
        # Run 3-step prediction
        result_multi = predict(
            model,
            sample_sequence,
            numerical_features,
            categorical_features,
            options={"steps": 3, "intervals": [1.0, 1.0, 1.0]},
        )

        # Run single step prediction
        result_single = predict(
            model, sample_sequence, numerical_features, categorical_features
        )

        # First step should be identical
        assert result_multi["rating"][0] == result_single["rating"][0]
        np.testing.assert_allclose(
            result_multi["expected_rating"][0],
            result_single["expected_rating"][0],
            rtol=1e-5,
        )

    def test_model_eval_mode(
        self,
        model: SPARKModel,
        sample_sequence: dict[str, np.ndarray],
        numerical_features: list[str],
        categorical_features: list[str],
    ):
        """Test that model is set to eval mode during prediction."""
        model.train()  # Ensure model starts in train mode

        _ = predict(model, sample_sequence, numerical_features, categorical_features)

        # After predict, model should be in eval mode
        assert not model.training

    def test_deterministic_predictions(
        self,
        model: SPARKModel,
        sample_sequence: dict[str, np.ndarray],
        numerical_features: list[str],
        categorical_features: list[str],
    ):
        """Test that predictions are deterministic."""
        result1 = predict(
            model, sample_sequence, numerical_features, categorical_features
        )
        result2 = predict(
            model, sample_sequence, numerical_features, categorical_features
        )

        np.testing.assert_array_equal(result1["rating"], result2["rating"])
        np.testing.assert_allclose(
            result1["expected_rating"], result2["expected_rating"], rtol=1e-5
        )
        np.testing.assert_allclose(
            result1["correct_prob"], result2["correct_prob"], rtol=1e-5
        )
