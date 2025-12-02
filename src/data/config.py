"""数据处理配置类。

包含特征配置和 Collator 配置。
"""

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True, slots=True)
class CollatorConfig:
    """Collator 配置。"""

    pad_value: float = 0.0
    pad_id: int = -1


@dataclass(frozen=True)
class FeatureConfig:
    """特征配置，定义模型输入所需的各类特征。

    根据模型架构文档，特征分为：
    - 数值特征流 (Numerical Stream): 间隔、累计学习量、耗时历史等
    - 类别特征流 (Categorical Stream): state, rating, is_first_review 等
    - 时间特征: day_offset 用于 Time2Vec 编码
    - ID 特征: card_id, deck_id 用于构建 Attention Mask
    - 目标特征: rating_gt1/2/3 (Ordinal Regression), log_duration (Duration Regression)
    """

    numerical_features: tuple[str, ...]
    categorical_features: tuple[str, ...]
    categorical_vocab_sizes: dict[str, int]
    ordinal_targets: tuple[str, ...]
    time_feature: str = "day_offset"
    card_id_feature: str = "card_id"
    deck_id_feature: str = "deck_id"
    duration_target: str = "log_duration"

    @classmethod
    def from_yaml(cls, config_path: Path | str) -> "FeatureConfig":
        """从 YAML 配置文件加载特征配置。"""
        with open(config_path, encoding="utf-8") as f:
            config = yaml.safe_load(f)
        return cls.from_dict(config.get("features", {}))

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> "FeatureConfig":
        """从字典加载特征配置。"""
        features = config.get("features", config)

        return cls(
            numerical_features=tuple(features.get("numerical_features")),
            categorical_features=tuple(features.get("categorical_features")),
            categorical_vocab_sizes=features.get("categorical_vocab_sizes"),
            time_feature=features.get("time_feature", "day_offset"),
            card_id_feature=features.get("card_id_feature", "card_id"),
            deck_id_feature=features.get("deck_id_feature", "deck_id"),
            ordinal_targets=tuple(features.get("ordinal_targets")),
            duration_target=features.get("duration_target", "log_duration"),
        )

    @cached_property
    def num_numerical_features(self) -> int:
        return len(self.numerical_features)

    @cached_property
    def num_categorical_features(self) -> int:
        return len(self.categorical_features)

    @cached_property
    def all_input_features(self) -> tuple[str, ...]:
        """返回模型所需的所有输入特征列名。"""
        return (
            *self.numerical_features,
            *self.categorical_features,
            self.time_feature,
            self.card_id_feature,
            self.deck_id_feature,
        )

    @cached_property
    def all_target_features(self) -> tuple[str, ...]:
        """返回所有目标特征列名。"""
        return (*self.ordinal_targets, self.duration_target)

    @cached_property
    def all_columns(self) -> tuple[str, ...]:
        """返回所有需要的列名（包含 user_id）。"""
        return (*self.all_input_features, *self.all_target_features, "user_id")


# 默认配置实例
DEFAULT_FEATURE_CONFIG = FeatureConfig.from_yaml("config/data.yaml")
DEFAULT_COLLATOR_CONFIG = CollatorConfig()
