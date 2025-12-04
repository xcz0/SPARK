"""Tests for the Lightning wrapper in src.models.module."""

from __future__ import annotations

from unittest import mock

import pytest
import torch

from spark.models.module import SPARKModule


@pytest.fixture
def spark_module(small_model_config):
    """Instantiate SPARKModule with the lightweight config fixture."""
    config = small_model_config
    return SPARKModule(
        d_model=config.d_model,
        n_heads=config.n_heads,
        depth=config.depth,
        dropout=config.dropout,
        num_numerical_features=config.num_numerical_features,
        categorical_vocab_sizes=config.categorical_vocab_sizes,
        categorical_embed_dim=config.categorical_embed_dim,
        num_rating_classes=config.num_rating_classes,
        card_head_ratio=config.card_head_ratio,
        deck_head_ratio=config.deck_head_ratio,
        learning_rate=1e-4,
        weight_decay=0.01,
        warmup_steps=10,
        rating_loss_weight=1.0,
        duration_loss_weight=0.1,
    )


def test_forward_returns_expected_keys(
    spark_module: SPARKModule, dummy_batch: dict[str, torch.Tensor]
):
    """Forward pass should proxy to SPARKModel and expose all outputs."""
    spark_module.eval()
    outputs = spark_module.forward(dummy_batch)

    assert set(outputs.keys()) == {"rating_probs", "duration_pred", "hidden_states"}

    batch_size, seq_len = dummy_batch["numerical_features"].shape[:2]
    num_thresholds = spark_module.hparams.num_rating_classes - 1

    assert outputs["rating_probs"].shape == (batch_size, seq_len, num_thresholds)
    assert outputs["duration_pred"].shape == (batch_size, seq_len)


def test_training_step_runs_backward(
    spark_module: SPARKModule, dummy_batch: dict[str, torch.Tensor]
):
    """training_step should produce a scalar loss usable for backprop."""
    spark_module.train()
    spark_module.log = mock.Mock()
    spark_module.log_dict = mock.Mock()
    loss = spark_module.training_step(dummy_batch, batch_idx=0)

    assert loss.dim() == 0
    assert loss.requires_grad

    loss.backward()
    grads_exist = any(
        parameter.grad is not None
        for parameter in spark_module.parameters()
        if parameter.requires_grad
    )
    assert grads_exist


def test_configure_optimizers_returns_scheduler(spark_module: SPARKModule):
    """configure_optimizers should yield both optimizer and scheduler settings."""
    config = spark_module.configure_optimizers()

    optimizer = config["optimizer"]
    scheduler_conf = config["lr_scheduler"]

    assert optimizer.__class__.__name__ == "AdamW"
    assert scheduler_conf["interval"] == "step"
    assert (
        scheduler_conf["scheduler"].__class__.__name__ == "CosineAnnealingWarmRestarts"
    )
