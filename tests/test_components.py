"""Unit tests for model component modules."""

from __future__ import annotations

import torch

from spark.models.components.attention import (
    DifferentialMultiHeadAttention,
    TimeDecayBias,
)
from spark.models.components.embeddings import (
    CategoricalEmbedding,
    NumericalProjection,
    Time2Vec,
)
from spark.models.components.heads import CORALHead, DurationHead
from spark.models.components.input_layer import InputLayer


class DummyMHA(torch.nn.Module):
    """Lightweight stand-in for torch.nn.MultiheadAttention."""

    def __init__(self):
        super().__init__()
        self.last_mask: torch.Tensor | None = None

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        need_weights: bool = False,
    ) -> tuple[torch.Tensor, None]:
        del key, value, need_weights
        self.last_mask = attn_mask.detach().clone() if attn_mask is not None else None
        return torch.zeros_like(query), None


class TestTimeDecayBias:
    def test_time_decay_bias_matches_closed_form(self):
        module = TimeDecayBias(num_heads=2)
        with torch.no_grad():
            module.weight.copy_(torch.tensor([1.0, 2.0]))
            module.decay_raw.copy_(torch.tensor([2.0, -0.5]))

        time_diff = torch.tensor([[[0.0, 1.0], [2.0, 0.5]]])
        bias = module(time_diff)

        decay = module._get_decay()
        expected = module.weight.view(1, 2, 1, 1) * torch.exp(
            -decay.view(1, 2, 1, 1) * time_diff.unsqueeze(1)
        )

        assert bias.shape == (1, 2, 2, 2)
        assert torch.allclose(bias, expected, atol=1e-6)


class TestDifferentialMultiHeadAttention:
    def test_grouped_masks_feed_multihead_attention(self):
        torch.manual_seed(0)
        module = DifferentialMultiHeadAttention(
            d_model=16,
            n_heads=4,
            dropout=0.0,
            card_head_ratio=0.5,
            deck_head_ratio=0.25,
        )
        dummy_mha = DummyMHA()
        module.mha = dummy_mha

        batch, seq_len = 1, 3
        x = torch.randn(batch, seq_len, module.d_model)
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))
        card_mask = torch.ones(batch, seq_len, seq_len, dtype=torch.bool)
        deck_mask = torch.ones_like(card_mask)
        time_diff = torch.tensor([[[0.0, 1.0, 2.0], [1.0, 0.0, 1.5], [2.0, 1.5, 0.0]]])

        with torch.no_grad():
            module.time_decay.weight.fill_(1.5)
            module.time_decay.decay_raw.fill_(0.7)

        output = module(
            x=x,
            causal_mask=causal_mask,
            card_mask=card_mask,
            deck_mask=deck_mask,
            time_diff=time_diff,
            padding_mask=None,
        )

        assert output.shape == (batch, seq_len, module.d_model)
        assert dummy_mha.last_mask is not None

        mask = dummy_mha.last_mask.view(batch, module.n_heads, seq_len, seq_len)
        time_bias = module.time_decay(time_diff)
        assert torch.allclose(mask[:, : module.n_card_heads], time_bias, atol=1e-5), (
            "Card heads should incorporate time bias"
        )

        deck_slice = mask[
            :, module.n_card_heads : module.n_card_heads + module.n_deck_heads
        ]
        assert torch.allclose(deck_slice, torch.zeros_like(deck_slice))

        global_slice = mask[:, -module.n_global_heads :]
        causal = torch.zeros(batch, seq_len, seq_len, dtype=x.dtype)
        causal = causal.masked_fill(~causal_mask.unsqueeze(0), float("-inf"))
        expected_global = causal.unsqueeze(1).expand_as(global_slice)
        assert torch.allclose(global_slice, expected_global)


class TestTime2Vec:
    def test_linear_and_periodic_terms(self):
        module = Time2Vec(embed_dim=3)
        with torch.no_grad():
            module.omega.copy_(torch.tensor([1.0, 2.0, 3.0]))
            module.phi.copy_(torch.tensor([0.0, 0.1, -0.2]))

        timestamps = torch.tensor([[0.0, 0.5]])
        output = module(timestamps)
        assert output.shape == (1, 2, 3)

        linear_term = module.omega * timestamps.unsqueeze(-1) + module.phi
        expected_linear = linear_term[..., :1]
        expected_periodic = torch.sin(linear_term[..., 1:])

        assert torch.allclose(output[..., :1], expected_linear, atol=1e-6)
        assert torch.allclose(output[..., 1:], expected_periodic, atol=1e-6)


class TestCategoricalEmbedding:
    def test_offsets_map_each_feature_segment(self):
        vocab_sizes = {"state": 3, "rating": 2}
        embed_dim = 2
        total_embed_dim = len(vocab_sizes) * embed_dim
        module = CategoricalEmbedding(
            vocab_sizes=vocab_sizes,
            embed_dim=embed_dim,
            output_dim=total_embed_dim,
        )

        total_vocab = sum(vocab_sizes.values())
        with torch.no_grad():
            module.embedding.weight.copy_(
                torch.arange(total_vocab * embed_dim)
                .view(total_vocab, embed_dim)
                .float()
            )
            module.projection.weight.copy_(torch.eye(total_embed_dim))
            module.projection.bias.zero_()

        categorical = torch.tensor([[[1, 0]]], dtype=torch.int64)
        output = module(categorical)

        # Feature 0 uses indices [0, 1, 2], feature 1 is offset by 3
        expected_feature0 = module.embedding.weight[1]
        expected_feature1 = module.embedding.weight[3]
        expected = torch.cat([expected_feature0, expected_feature1])

        assert output.shape == (1, 1, total_embed_dim)
        assert torch.allclose(output[0, 0], expected)


class TestNumericalProjection:
    def test_projection_shapes_match_configuration(self):
        module = NumericalProjection(
            num_features=4,
            output_dim=6,
            hidden_dim=8,
            dropout=0.0,
        )
        features = torch.randn(2, 5, 4)
        projected = module(features)

        assert projected.shape == (2, 5, 6)
        assert projected.dtype == features.dtype


class TestInputLayer:
    def test_fuses_all_feature_streams(self):
        layer = InputLayer(
            d_model=8,
            num_numerical_features=4,
            categorical_vocab_sizes={"state": 3, "rating": 2, "first": 2},
            categorical_embed_dim=2,
            dropout=0.0,
        )

        numerical = torch.randn(1, 3, 4)
        categorical = torch.tensor(
            [[[0, 1, 0], [1, 0, 1], [2, 1, 1]]], dtype=torch.int64
        )
        time_stamps = torch.linspace(0, 1, steps=3).view(1, -1)

        embeddings = layer(
            numerical_features=numerical,
            categorical_features=categorical,
            time_stamps=time_stamps,
        )

        assert embeddings.shape == (1, 3, 8)
        assert torch.isfinite(embeddings).all()


class TestHeads:
    def test_coral_head_outputs_monotonic_probabilities(self):
        head = CORALHead(d_model=8, num_classes=4)
        x = torch.randn(2, 4, 8)
        probs = head(x)

        assert probs.shape == (2, 4, 3)
        assert torch.all(probs >= 0) and torch.all(probs <= 1)
        assert torch.all(probs[..., :-1] >= probs[..., 1:] - 1e-6)

    def test_coral_head_predict_rating_range(self):
        head = CORALHead(d_model=8, num_classes=4)
        x = torch.randn(1, 2, 8)
        ratings = head.predict_rating(x)

        assert ratings.shape == (1, 2)
        assert torch.all((1 <= ratings) & (ratings <= 4))

    def test_duration_head_emits_scalar_per_token(self):
        head = DurationHead(d_model=8, hidden_dim=4, dropout=0.0)
        x = torch.randn(3, 2, 8)
        duration = head(x)

        assert duration.shape == (3, 2)
