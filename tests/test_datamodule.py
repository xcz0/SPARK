"""测试 src.data.datamodule 模块。"""

import numpy as np
import pandas as pd
import pytest
import tempfile
from pathlib import Path

from src.data.datamodule import ReviewDataModule
from src.data.config import FeatureConfig


@pytest.fixture(scope="module")
def temp_data_dir():
    """创建一次临时数据目录并在模块结束后清理。"""
    tmpdir = tempfile.TemporaryDirectory()
    path = Path(tmpdir.name)

    for user_id in [1, 2, 3, 4, 5]:
        n_records = 50 + user_id * 10
        rng = np.random.default_rng(user_id)

        data = pd.DataFrame(
            {
                "user_id": [user_id] * n_records,
                "log_elapsed_seconds": rng.standard_normal(n_records),
                "log_elapsed_days": rng.standard_normal(n_records),
                "log_time_spent_today": rng.standard_normal(n_records),
                "log_prev_review_duration": rng.standard_normal(n_records),
                "log_last_duration_on_card": rng.standard_normal(n_records),
                "card_index_today": rng.integers(0, 10, n_records),
                "card_review_count": rng.integers(0, 100, n_records),
                "state": rng.integers(0, 4, n_records),
                "prev_review_rating": rng.integers(0, 5, n_records),
                "last_rating_on_card": rng.integers(0, 5, n_records),
                "is_first_review": rng.choice([True, False], n_records),
                "day_offset": np.arange(n_records, dtype=float),
                "card_id": rng.integers(1, 20, n_records),
                "deck_id": rng.integers(1, 5, n_records),
                "rating_gt1": rng.choice([0.0, 1.0], n_records),
                "rating_gt2": rng.choice([0.0, 1.0], n_records),
                "rating_gt3": rng.choice([0.0, 1.0], n_records),
                "log_duration": rng.standard_normal(n_records),
            }
        )
        data.to_parquet(path / f"user_id={user_id}.parquet", index=False)

    yield path
    tmpdir.cleanup()


class TestReviewDataModule:
    """测试 ReviewDataModule。"""

    def test_initialization(self, temp_data_dir):
        """测试初始化。"""
        dm = ReviewDataModule(
            data_dir=temp_data_dir,
            seq_len=32,
            batch_size=4,
            num_workers=0,
        )

        assert dm.hparams.seq_len == 32
        assert dm.hparams.batch_size == 4
        assert dm.hparams.stride == 32  # 默认等于 seq_len

    def test_setup_preload_mode(self, temp_data_dir):
        """测试预加载模式设置。"""
        dm = ReviewDataModule(
            data_dir=temp_data_dir,
            seq_len=32,
            batch_size=4,
            num_workers=0,
            streaming=False,
        )

        dm.setup("fit")
        assert dm.train_dataset is not None
        assert dm.val_dataset is not None

        dm.setup("test")
        assert dm.test_dataset is not None

    def test_setup_streaming_mode(self, temp_data_dir):
        """测试流式模式设置。"""
        dm = ReviewDataModule(
            data_dir=temp_data_dir,
            seq_len=32,
            batch_size=4,
            num_workers=0,
            streaming=True,
        )

        dm.setup("fit")

        assert dm.train_dataset is not None
        assert dm.val_dataset is not None

    def test_train_dataloader(self, temp_data_dir):
        """测试训练数据加载器。"""
        dm = ReviewDataModule(
            data_dir=temp_data_dir,
            seq_len=32,
            batch_size=4,
            num_workers=0,
        )
        dm.setup("fit")

        train_loader = dm.train_dataloader()
        batch = next(iter(train_loader))

        assert batch["numerical_features"].shape[0] == 4  # batch_size
        assert "padding_mask" in batch
        assert "card_mask" in batch

    def test_val_dataloader(self, temp_data_dir):
        """测试验证数据加载器。"""
        dm = ReviewDataModule(
            data_dir=temp_data_dir,
            seq_len=32,
            batch_size=4,
            num_workers=0,
        )
        dm.setup("fit")

        val_loader = dm.val_dataloader()
        batch = next(iter(val_loader))

        assert batch["numerical_features"].shape[0] <= 4

    def test_test_dataloader(self, temp_data_dir):
        """测试测试数据加载器。"""
        dm = ReviewDataModule(
            data_dir=temp_data_dir,
            seq_len=32,
            batch_size=4,
            num_workers=0,
        )
        dm.setup("test")

        test_loader = dm.test_dataloader()
        batch = next(iter(test_loader))

        assert "numerical_features" in batch

    def test_data_split_ratios(self, temp_data_dir):
        """测试数据划分比例。"""
        dm = ReviewDataModule(
            data_dir=temp_data_dir,
            seq_len=32,
            batch_size=4,
            num_workers=0,
            train_ratio=0.6,
            val_ratio=0.2,
        )
        dm.setup("fit")
        dm.setup("test")

        total = len(dm.train_dataset) + len(dm.val_dataset) + len(dm.test_dataset)

        # 验证比例大致正确（由于整数划分可能有微小差异）
        train_ratio = len(dm.train_dataset) / total
        assert 0.55 <= train_ratio <= 0.65

    def test_custom_feature_config(self, temp_data_dir):
        """测试特征配置属性和自定义配置。"""
        # 默认配置
        dm_default = ReviewDataModule(
            data_dir=temp_data_dir,
            seq_len=32,
            batch_size=4,
        )
        assert dm_default.num_numerical_features == 7
        assert dm_default.num_categorical_features == 3
        assert "state" in dm_default.categorical_vocab_sizes

        # 自定义配置
        custom_config = FeatureConfig(
            numerical_features=("log_elapsed_seconds", "log_elapsed_days"),
            categorical_features=("state",),
            categorical_vocab_sizes={"state": 4},
            ordinal_targets=("rating_gt1",),
        )
        dm_custom = ReviewDataModule(
            data_dir=temp_data_dir,
            seq_len=32,
            batch_size=4,
            feature_config=custom_config,
        )
        assert dm_custom.num_numerical_features == 2

    def test_reproducibility(self, temp_data_dir):
        """测试可重复性。"""
        dm1 = ReviewDataModule(
            data_dir=temp_data_dir,
            seq_len=32,
            batch_size=4,
            num_workers=0,
            seed=42,
        )
        dm1.setup("fit")

        dm2 = ReviewDataModule(
            data_dir=temp_data_dir,
            seq_len=32,
            batch_size=4,
            num_workers=0,
            seed=42,
        )
        dm2.setup("fit")

        # 相同种子应产生相同的数据划分
        assert len(dm1.train_dataset) == len(dm2.train_dataset)

    def test_stride_parameter(self, temp_data_dir):
        """测试步长参数。"""
        dm1 = ReviewDataModule(
            data_dir=temp_data_dir,
            seq_len=32,
            stride=32,  # 无重叠
            num_workers=0,
        )
        dm1.setup("fit")
        dm1.setup("test")

        dm2 = ReviewDataModule(
            data_dir=temp_data_dir,
            seq_len=32,
            stride=16,  # 50% 重叠
            num_workers=0,
        )
        dm2.setup("fit")
        dm2.setup("test")

        # 重叠应产生更多序列
        total1 = len(dm1.train_dataset) + len(dm1.val_dataset) + len(dm1.test_dataset)
        total2 = len(dm2.train_dataset) + len(dm2.val_dataset) + len(dm2.test_dataset)
        assert total2 > total1


class TestReviewDataModuleIntegration:
    """ReviewDataModule 集成测试。"""

    def test_from_config(self, temp_data_dir, tmp_path):
        """测试从配置文件创建数据模块及参数覆盖。"""
        config_content = f"""
data:
  processed_dir: "{temp_data_dir}"
  batch_size: 8
  window_size: 32
  num_workers: 0
  train_ratio: 0.7
  val_ratio: 0.15

features:
  numerical_features:
    - log_elapsed_seconds
    - log_elapsed_days
    - log_time_spent_today
    - log_prev_review_duration
    - log_last_duration_on_card
    - card_index_today
  categorical_features:
    - state
    - prev_review_rating
    - is_first_review
  categorical_vocab_sizes:
    state: 4
    prev_review_rating: 5
    is_first_review: 2
  time_feature: day_offset
  card_id_feature: card_id
  deck_id_feature: deck_id
  ordinal_targets:
    - rating_gt1
    - rating_gt2
    - rating_gt3
  duration_target: log_duration
"""
        config_path = tmp_path / "test_config.yaml"
        config_path.write_text(config_content)

        # 测试基本创建
        dm = ReviewDataModule.from_config(config_path)
        assert dm.hparams.batch_size == 8
        assert dm.hparams.seq_len == 32
        assert dm.hparams.train_ratio == 0.7

        # 测试参数覆盖
        dm_override = ReviewDataModule.from_config(
            config_path,
            overrides={"batch_size": 16, "seq_len": 64},
        )
        assert dm_override.hparams.batch_size == 16
        assert dm_override.hparams.seq_len == 64

    def test_training_batch_validity(self, temp_data_dir):
        """测试训练批次的有效性和一致性。"""
        dm = ReviewDataModule(
            data_dir=temp_data_dir,
            seq_len=32,
            batch_size=2,
            num_workers=0,
        )
        dm.setup("fit")

        train_loader = dm.train_dataloader()
        batch = next(iter(train_loader))

        # 验证批次有效性
        assert batch["numerical_features"].dim() == 3
        assert batch["padding_mask"].dtype == torch.bool

        # 验证所有张量的批次维度一致
        batch_size = batch["numerical_features"].shape[0]
        assert batch["categorical_features"].shape[0] == batch_size
        assert batch["padding_mask"].shape[0] == batch_size
        assert batch["card_mask"].shape[0] == batch_size
        assert batch["seq_lens"].shape[0] == batch_size


# 需要导入 torch 用于断言
import torch
