"""Tests for ReviewSequenceDataset and StreamingReviewDataset."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.chdir(PROJECT_ROOT)

from spark.data.config import FeatureConfig
from spark.data.dataset import ReviewSequenceDataset, StreamingReviewDataset


@pytest.fixture
def feature_config() -> FeatureConfig:
    """提供测试用的特征配置。"""
    return FeatureConfig(
        numerical_features=(
            "log_elapsed_seconds",
            "log_elapsed_days",
            "log_time_spent_today",
            "log_prev_review_duration",
            "log_last_duration_on_card",
            "card_index_today",
        ),
        categorical_features=("state", "prev_review_rating", "is_first_review"),
        categorical_vocab_sizes={
            "state": 4,
            "prev_review_rating": 5,
            "is_first_review": 2,
        },
        ordinal_targets=("rating_gt1", "rating_gt2", "rating_gt3"),
    )


@pytest.fixture
def sample_data() -> dict[str, np.ndarray]:
    """生成测试用的数据字典。"""
    np.random.seed(42)
    n_samples = 20

    return {
        "user_id": np.array([1] * 10 + [2] * 10),
        "card_id": np.random.randint(1, 100, n_samples),
        "deck_id": np.random.randint(1, 10, n_samples),
        "day_offset": np.arange(n_samples, dtype=np.float32),
        # 数值特征
        "log_elapsed_seconds": np.random.randn(n_samples).astype(np.float32),
        "log_elapsed_days": np.random.randn(n_samples).astype(np.float32),
        "log_time_spent_today": np.random.randn(n_samples).astype(np.float32),
        "log_prev_review_duration": np.random.randn(n_samples).astype(np.float32),
        "log_last_duration_on_card": np.random.randn(n_samples).astype(np.float32),
        "card_index_today": np.random.randn(n_samples).astype(np.float32),
        # 类别特征
        "state": np.random.randint(0, 4, n_samples),
        "prev_review_rating": np.random.randint(0, 5, n_samples),
        "is_first_review": np.random.randint(0, 2, n_samples),
        # 目标
        "rating_gt1": np.random.rand(n_samples).astype(np.float32),
        "rating_gt2": np.random.rand(n_samples).astype(np.float32),
        "rating_gt3": np.random.rand(n_samples).astype(np.float32),
        "log_duration": np.random.rand(n_samples).astype(np.float32),
    }


class TestReviewSequenceDataset:
    """ReviewSequenceDataset 测试类。"""

    def test_dataset_creation_and_length(
        self, sample_data: dict[str, np.ndarray], feature_config: FeatureConfig
    ):
        """测试数据集创建和长度计算。"""
        dataset = ReviewSequenceDataset(
            sample_data, feature_config=feature_config, seq_len=8, stride=4
        )
        assert len(dataset) > 0

        # 更短的序列长度和步长应产生更多样本
        dataset_short = ReviewSequenceDataset(
            sample_data, feature_config=feature_config, seq_len=4, stride=2
        )
        assert len(dataset_short) >= len(dataset)

    def test_getitem_structure_and_types(
        self, sample_data: dict[str, np.ndarray], feature_config: FeatureConfig
    ):
        """测试 __getitem__ 返回正确的键、类型和数据类型。"""
        dataset = ReviewSequenceDataset(
            sample_data, feature_config=feature_config, seq_len=8
        )
        sample = dataset[0]

        expected_keys = {
            "numerical_features",
            "categorical_features",
            "time_stamps",
            "card_ids",
            "deck_ids",
            "ordinal_targets",
            "duration_targets",
            "seq_len",
        }
        assert set(sample.keys()) == expected_keys

        # 检查类型和数据类型
        expected_dtypes = {
            "numerical_features": torch.float32,
            "categorical_features": torch.int64,
            "time_stamps": torch.float32,
            "card_ids": torch.int64,
            "deck_ids": torch.int64,
            "ordinal_targets": torch.float32,
            "duration_targets": torch.float32,
            "seq_len": torch.int64,
        }
        for key, value in sample.items():
            assert isinstance(value, torch.Tensor), f"{key} 不是 Tensor"
            assert value.dtype == expected_dtypes[key], f"{key} dtype 不正确"

    def test_features_shape_consistency(
        self, sample_data: dict[str, np.ndarray], feature_config: FeatureConfig
    ):
        """测试所有特征形状与序列长度一致。"""
        dataset = ReviewSequenceDataset(
            sample_data, feature_config=feature_config, seq_len=8
        )
        sample = dataset[0]
        seq_len = sample["seq_len"].item()

        # 检查所有特征的第一维都等于 seq_len
        assert sample["numerical_features"].shape == (
            seq_len,
            len(feature_config.numerical_features),
        )
        assert sample["categorical_features"].shape == (
            seq_len,
            len(feature_config.categorical_features),
        )
        assert sample["ordinal_targets"].shape == (
            seq_len,
            len(feature_config.ordinal_targets),
        )
        assert sample["time_stamps"].shape == (seq_len,)
        assert sample["card_ids"].shape == (seq_len,)
        assert sample["deck_ids"].shape == (seq_len,)
        assert sample["duration_targets"].shape == (seq_len,)

    def test_stride_configuration(
        self,
        sample_data: dict[str, np.ndarray],
        feature_config: FeatureConfig,
    ):
        """测试步长配置：默认等于 seq_len，可自定义。"""
        dataset_default = ReviewSequenceDataset(
            sample_data, feature_config=feature_config, seq_len=8, stride=None
        )
        assert dataset_default.stride == 8

        dataset_custom = ReviewSequenceDataset(
            sample_data, feature_config=feature_config, seq_len=16, stride=4
        )
        assert dataset_custom.stride == 4

    def test_minimum_sequence_length_filter(
        self, sample_data: dict[str, np.ndarray], feature_config: FeatureConfig
    ):
        """测试最小序列长度过滤（seq_length > 1）。"""
        # 创建只有 1 条记录的用户数据
        small_data = {k: v[:1] for k, v in sample_data.items()}
        dataset = ReviewSequenceDataset(
            small_data, feature_config=feature_config, seq_len=8
        )
        assert len(dataset) == 0


class TestStreamingReviewDataset:
    """StreamingReviewDataset 测试类。"""

    def test_dataset_creation(self, feature_config: FeatureConfig, tmp_path: Path):
        """测试流式数据集创建。"""
        dataset = StreamingReviewDataset(
            data_dir=tmp_path,
            user_ids=[1, 2, 3],
            seq_len=8,
            feature_config=feature_config,
        )
        assert dataset.seq_len == 8
        assert list(dataset.user_ids) == [1, 2, 3]
        assert dataset.shuffle is True

    def test_configuration_options(
        self,
        feature_config: FeatureConfig,
        tmp_path: Path,
    ):
        """测试配置选项：stride 和 shuffle。"""
        dataset = StreamingReviewDataset(
            data_dir=tmp_path,
            user_ids=[1],
            seq_len=16,
            feature_config=feature_config,
            stride=4,
            shuffle=False,
        )
        assert dataset.stride == 4
        assert dataset.shuffle is False

    @patch("src.data.dataset.load_user_data")
    def test_iteration_yields_samples(
        self,
        mock_load: MagicMock,
        sample_data: dict[str, np.ndarray],
        feature_config: FeatureConfig,
        tmp_path: Path,
    ):
        """测试迭代生成样本。"""
        # 只取 user_id == 1 的数据
        mask = sample_data["user_id"] == 1
        user_data = {k: v[mask] for k, v in sample_data.items()}
        mock_load.return_value = user_data

        dataset = StreamingReviewDataset(
            data_dir=tmp_path,
            user_ids=[1],
            seq_len=8,
            feature_config=feature_config,
            shuffle=False,
        )

        samples = list(dataset)
        assert len(samples) > 0

        sample = samples[0]
        assert "numerical_features" in sample
        assert isinstance(sample["numerical_features"], torch.Tensor)

    @patch("src.data.dataset.load_user_data")
    def test_iteration_multiple_users(
        self,
        mock_load: MagicMock,
        sample_data: dict[str, np.ndarray],
        feature_config: FeatureConfig,
        tmp_path: Path,
    ):
        """测试多用户迭代。"""

        def side_effect(
            data_dir: Path, user_id: int, columns: list[str]
        ) -> dict[str, np.ndarray]:
            mask = sample_data["user_id"] == user_id
            return {k: v[mask] for k, v in sample_data.items()}

        mock_load.side_effect = side_effect

        dataset = StreamingReviewDataset(
            data_dir=tmp_path,
            user_ids=[1, 2],
            seq_len=8,
            feature_config=feature_config,
            shuffle=False,
        )

        samples = list(dataset)
        assert len(samples) > 0
        assert mock_load.call_count == 2

    @patch("src.data.dataset.get_worker_info")
    def test_worker_indices_distribution(
        self, mock_worker_info: MagicMock, feature_config: FeatureConfig, tmp_path: Path
    ):
        """测试 worker 索引分配。"""
        user_ids = [1, 2, 3, 4, 5]

        # 单 worker 模式
        mock_worker_info.return_value = None
        dataset = StreamingReviewDataset(
            data_dir=tmp_path,
            user_ids=user_ids,
            seq_len=8,
            feature_config=feature_config,
        )
        indices = dataset._get_worker_indices()
        assert list(indices) == [0, 1, 2, 3, 4]

        # 多 worker 模式 - worker 0 of 2
        worker_info = MagicMock()
        worker_info.num_workers = 2
        worker_info.id = 0
        mock_worker_info.return_value = worker_info
        indices = dataset._get_worker_indices()
        assert list(indices) == [0, 1, 2]

        # 多 worker 模式 - worker 1 of 2 (获取剩余)
        worker_info.id = 1
        indices = dataset._get_worker_indices()
        assert list(indices) == [3, 4]


class TestRealConfigAndData:
    """使用真实配置文件和数据的测试。"""

    @pytest.fixture
    def real_config(self) -> FeatureConfig:
        """从真实配置文件加载特征配置。"""
        config_path = Path("configs/data.yaml")
        if not config_path.exists():
            pytest.skip("配置文件 configs/data.yaml 不存在")
        return FeatureConfig.from_yaml(config_path)

    @pytest.fixture
    def real_data_path(self) -> Path:
        """获取真实数据路径。"""
        return Path("data/processed")

    def test_load_config_from_yaml(self):
        """测试从 YAML 文件加载配置。"""
        config_path = Path("configs/data.yaml")
        if not config_path.exists():
            pytest.skip("配置文件不存在")

        config = FeatureConfig.from_yaml(config_path)

        # 验证数值特征
        assert len(config.numerical_features) == 6
        assert "log_elapsed_seconds" in config.numerical_features
        assert "log_elapsed_days" in config.numerical_features

        # 验证类别特征
        assert len(config.categorical_features) == 3
        assert "state" in config.categorical_features
        assert "prev_review_rating" in config.categorical_features

        # 验证词表大小
        assert config.categorical_vocab_sizes["state"] == 4
        assert config.categorical_vocab_sizes["prev_review_rating"] == 5
        assert config.categorical_vocab_sizes["is_first_review"] == 2

        # 验证目标特征
        assert len(config.ordinal_targets) == 3
        assert config.duration_target == "log_duration"

    def test_load_real_user_data(
        self, real_config: FeatureConfig, real_data_path: Path
    ):
        """测试加载真实用户数据。"""
        from spark.data.loader import load_user_data

        data_file = real_data_path / "user_id=1.parquet"
        if not data_file.exists():
            pytest.skip("用户数据文件不存在")

        columns = list(real_config.all_columns)
        data = load_user_data(real_data_path, 1, columns)

        # 验证所有列都被加载
        for col in columns:
            assert col in data, f"缺少列: {col}"
            assert len(data[col]) > 0, f"列 {col} 为空"

        # 验证数值特征可以转换为浮点类型（数据集会自动转换）
        for feat in real_config.numerical_features:
            assert np.issubdtype(data[feat].dtype, np.number), f"{feat} 应为数值类型"

    def test_review_sequence_dataset_with_real_config_and_data(
        self, real_config: FeatureConfig, real_data_path: Path
    ):
        """使用真实配置和数据测试 ReviewSequenceDataset。"""
        from spark.data.loader import load_user_data

        data_file = real_data_path / "user_id=1.parquet"
        if not data_file.exists():
            pytest.skip("真实数据文件不存在")

        data = load_user_data(real_data_path, 1, list(real_config.all_columns))
        dataset = ReviewSequenceDataset(
            data, feature_config=real_config, seq_len=50, stride=25
        )

        assert len(dataset) > 0, "数据集不应为空"

        sample = dataset[0]

        # 验证特征维度与配置一致
        assert sample["numerical_features"].shape[1] == len(
            real_config.numerical_features
        )
        assert sample["categorical_features"].shape[1] == len(
            real_config.categorical_features
        )
        assert sample["ordinal_targets"].shape[1] == len(real_config.ordinal_targets)

        # 验证序列长度
        seq_len = sample["seq_len"].item()
        assert seq_len <= 50
        assert sample["time_stamps"].shape[0] == seq_len
        assert sample["card_ids"].shape[0] == seq_len

    def test_streaming_dataset_with_real_config_and_data(
        self, real_config: FeatureConfig, real_data_path: Path
    ):
        """使用真实配置和数据测试 StreamingReviewDataset。"""
        if not real_data_path.exists():
            pytest.skip("真实数据目录不存在")

        # 获取所有可用的用户 ID
        user_files = list(real_data_path.glob("user_id=*.parquet"))
        if not user_files:
            pytest.skip("没有找到用户数据文件")

        user_ids = [int(f.stem.split("=")[1]) for f in user_files]

        dataset = StreamingReviewDataset(
            data_dir=real_data_path,
            user_ids=user_ids[:2],  # 只测试前两个用户
            seq_len=50,
            feature_config=real_config,
            shuffle=False,
        )

        samples = list(dataset)
        assert len(samples) > 0, "应该生成至少一个样本"

        sample = samples[0]
        assert sample["numerical_features"].shape[1] == len(
            real_config.numerical_features
        )
        assert sample["categorical_features"].shape[1] == len(
            real_config.categorical_features
        )
