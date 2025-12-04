"""PyTorch Lightning DataModule，用于管理复习序列数据的加载流程。"""

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Subset, random_split
from lightning import LightningDataModule

from .collator import ReviewCollator
from .config import CollatorConfig, DEFAULT_FEATURE_CONFIG, FeatureConfig
from .dataset import ReviewSequenceDataset, StreamingReviewDataset
from .loader import load_all_users_data

# 数据集类型别名
type _DatasetType = (
    ReviewSequenceDataset
    | StreamingReviewDataset
    | Subset[dict[str, torch.Tensor]]
    | None
)


class ReviewDataModule(LightningDataModule):
    """复习序列数据模块。

    管理训练、验证和测试数据的加载，支持两种模式：
    1. 预加载模式：将所有数据加载到内存
    2. 流式模式：按需加载用户数据

    Example:
        >>> dm = ReviewDataModule.from_config("configs/data.yaml")
        >>> dm.setup("fit")
        >>> train_loader = dm.train_dataloader()
    """

    def __init__(
        self,
        data_dir: str | Path = "data/processed",
        streaming: bool = False,
        seed: int = 42,
        seq_len: int = 256,
        stride: int | None = None,
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = False,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        feature_config: FeatureConfig | None = None,
        collator_config: CollatorConfig | None = None,
    ):
        """初始化数据模块。

        Args:
            data_dir: 处理后数据的目录路径
            seq_len: 序列长度
            stride: 滑动窗口步长，默认等于 seq_len（无重叠）
            batch_size: 批次大小
            num_workers: DataLoader 工作进程数
            pin_memory: 是否将数据固定在 GPU 内存
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            test_ratio: 测试集比例
            feature_config: 特征配置
            collator_config: Collator 配置
            streaming: 是否使用流式加载模式
            seed: 随机种子
        """
        super().__init__()
        self.save_hyperparameters(ignore=["feature_config", "collator_config"])

        # 如果未指定 stride，则默认等于 seq_len（无重叠）
        if self.hparams.stride is None:
            self.hparams.stride = self.hparams.seq_len

        self.data_dir = Path(data_dir)
        self.feature_config = feature_config or DEFAULT_FEATURE_CONFIG
        self._collator = ReviewCollator(collator_config or CollatorConfig())

        # 数据集在 setup() 中按需初始化
        self.train_dataset: _DatasetType = None
        self.val_dataset: _DatasetType = None
        self.test_dataset: _DatasetType = None
        self.predict_dataset: _DatasetType = None

        # 预加载模式下缓存的完整数据集（用于按需划分）
        self._full_dataset: ReviewSequenceDataset | None = None
        self._dataset_splits: list[Subset[dict[str, torch.Tensor]]] | None = None

    @classmethod
    def from_config(
        cls,
        config_path: str | Path,
        *,
        overrides: dict[str, Any] | None = None,
    ) -> "ReviewDataModule":
        """从配置文件创建数据模块。

        Args:
            config_path: YAML 配置文件路径
            overrides: 覆盖配置的参数字典

        Returns:
            初始化后的 ReviewDataModule 实例
        """
        import yaml

        with open(config_path, encoding="utf-8") as f:
            config = yaml.safe_load(f)

        data_config = config.get("data", {})
        feature_config = FeatureConfig.from_dict(config)

        params = {
            "data_dir": data_config.get("processed_dir", "data/processed"),
            "seq_len": data_config.get("window_size", 256),
            "stride": data_config.get("stride"),
            "batch_size": data_config.get("batch_size", 32),
            "num_workers": data_config.get("num_workers", 4),
            "pin_memory": data_config.get("pin_memory", False),
            "train_ratio": data_config.get("train_ratio", 0.8),
            "val_ratio": data_config.get("val_ratio", 0.1),
            "streaming": data_config.get("streaming", False),
            "seed": data_config.get("seed", 42),
            "feature_config": feature_config,
        }

        if overrides:
            params.update(overrides)

        return cls(**params)

    def setup(self, stage: str) -> None:
        """设置数据集。

        根据 stage 参数按需设置数据集，避免不必要的内存占用。
        该方法会在每个 GPU/进程上被调用。

        Args:
            stage: 当前阶段，可选值为 "fit"、"validate"、"test"、"predict"
        """
        setup_method = (
            self._setup_streaming if self.hparams.streaming else self._setup_preload
        )
        setup_method(stage)

    def _setup_preload(self, stage: str) -> None:
        """预加载模式的数据设置。

        使用延迟划分策略：仅在首次需要时加载并划分数据集。
        """
        # 首次调用时加载完整数据集并进行划分
        if self._dataset_splits is None:
            all_data = load_all_users_data(
                self.data_dir, columns=list(self.feature_config.all_columns)
            )
            self._full_dataset = ReviewSequenceDataset(
                all_data,
                seq_len=self.hparams.seq_len,
                feature_config=self.feature_config,
                stride=self.hparams.stride,
            )

            # 划分数据集
            total_size = len(self._full_dataset)
            val_size = int(total_size * self.hparams.val_ratio)
            test_size = int(total_size * self.hparams.test_ratio)
            train_size = total_size - val_size - test_size

            self._dataset_splits = random_split(
                self._full_dataset,
                [train_size, val_size, test_size],
                generator=torch.Generator().manual_seed(self.hparams.seed),
            )

        train_split, val_split, test_split = self._dataset_splits

        # 根据 stage 设置对应数据集
        match stage:
            case "fit":
                self.train_dataset = train_split
                self.val_dataset = val_split
            case "validate":
                self.val_dataset = val_split
            case "test":
                self.test_dataset = test_split
            case "predict":
                self.predict_dataset = test_split

    def _setup_streaming(self, stage: str) -> None:
        """流式模式的数据设置。"""
        parquet_files = sorted(self.data_dir.glob("user_id=*.parquet"))
        user_ids = [int(f.stem.split("=")[1]) for f in parquet_files]

        # 使用 torch.Generator 保证可重复性
        generator = torch.Generator().manual_seed(self.hparams.seed)
        perm = torch.randperm(len(user_ids), generator=generator).tolist()
        user_ids = [user_ids[i] for i in perm]

        total_users = len(user_ids)
        train_end = int(total_users * self.hparams.train_ratio)
        val_end = train_end + int(total_users * self.hparams.val_ratio)

        dataset_kwargs = {
            "seq_len": self.hparams.seq_len,
            "feature_config": self.feature_config,
            "stride": self.hparams.stride,
        }

        match stage:
            case "fit":
                self.train_dataset = StreamingReviewDataset(
                    self.data_dir, user_ids[:train_end], shuffle=True, **dataset_kwargs
                )
                self.val_dataset = StreamingReviewDataset(
                    self.data_dir,
                    user_ids[train_end:val_end],
                    shuffle=False,
                    **dataset_kwargs,
                )
            case "validate":
                self.val_dataset = StreamingReviewDataset(
                    self.data_dir,
                    user_ids[train_end:val_end],
                    shuffle=False,
                    **dataset_kwargs,
                )
            case "test":
                self.test_dataset = StreamingReviewDataset(
                    self.data_dir, user_ids[val_end:], shuffle=False, **dataset_kwargs
                )
            case "predict":
                self.predict_dataset = StreamingReviewDataset(
                    self.data_dir, user_ids[val_end:], shuffle=False, **dataset_kwargs
                )

    def _create_dataloader(
        self, dataset: _DatasetType, *, shuffle: bool = False, drop_last: bool = False
    ) -> DataLoader:
        """创建 DataLoader 的通用方法。"""
        return DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            shuffle=shuffle and not self.hparams.streaming,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=self._collator,
            drop_last=drop_last,
            persistent_workers=self.hparams.num_workers > 0,
        )

    def train_dataloader(self) -> DataLoader:
        """返回训练数据加载器。"""
        return self._create_dataloader(self.train_dataset, shuffle=True, drop_last=True)

    def val_dataloader(self) -> DataLoader:
        """返回验证数据加载器。"""
        return self._create_dataloader(self.val_dataset)

    def test_dataloader(self) -> DataLoader:
        """返回测试数据加载器。"""
        return self._create_dataloader(self.test_dataset)

    def predict_dataloader(self) -> DataLoader:
        """返回预测数据加载器。"""
        return self._create_dataloader(self.predict_dataset)

    @property
    def num_numerical_features(self) -> int:
        """返回数值特征数量。"""
        return self.feature_config.num_numerical_features

    @property
    def num_categorical_features(self) -> int:
        """返回类别特征数量。"""
        return self.feature_config.num_categorical_features

    @property
    def categorical_vocab_sizes(self) -> dict[str, int]:
        """返回类别特征的词表大小。"""
        return self.feature_config.categorical_vocab_sizes
