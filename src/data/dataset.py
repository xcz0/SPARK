"""复习序列数据集。

提供两种数据集实现：
- ReviewSequenceDataset: 预加载模式，将数据全部加载到内存
- StreamingReviewDataset: 流式模式，按需加载用户数据
"""

from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, IterableDataset, get_worker_info

from src.data.config import DEFAULT_FEATURE_CONFIG, FeatureConfig
from src.data.loader import load_user_data


class ReviewSequenceDataset(Dataset):
    """内存驻留型数据集。

    适用于数据量可以完全载入内存的场景。
    """

    def __init__(
        self,
        data: dict[str, np.ndarray],
        feature_config: FeatureConfig = DEFAULT_FEATURE_CONFIG,
        seq_len: int = 256,
        stride: int | None = None,
    ):
        self.seq_len = seq_len
        self.feature_config = feature_config
        self.stride = stride or seq_len

        # 预处理数据
        self._prepare_data(data)

    def _prepare_data(self, data: dict[str, np.ndarray]) -> None:
        """预处理数据：构建特征矩阵和索引。"""
        cfg = self.feature_config

        # 1. 提取并预先堆叠特征 (Pre-stacking)
        # 将分离的列合并为 (Total_Rows, Num_Features) 的矩阵
        self._numerical = np.stack(
            [data[f] for f in cfg.numerical_features], axis=1
        ).astype(np.float32)

        self._categorical = np.stack(
            [data[f] for f in cfg.categorical_features], axis=1
        ).astype(np.int64)

        # 提取其他单列特征
        self._time_stamps = data[cfg.time_feature].astype(np.float32)
        self._card_ids = data[cfg.card_id_feature].astype(np.int64)
        self._deck_ids = data[cfg.deck_id_feature].astype(np.int64)
        self._duration_targets = data[cfg.duration_target].astype(np.float32)

        self._ordinal_targets = np.stack(
            [data[t] for t in cfg.ordinal_targets], axis=1
        ).astype(np.float32)

        # 2. 构建序列索引
        # 假设数据已按 user_id 排序，使用 np.unique 快速获取每个用户的长度
        user_ids = data["user_id"]
        _, user_lens = np.unique(user_ids, return_counts=True)

        # 计算每个用户的起始行号
        user_starts = np.zeros(len(user_lens) + 1, dtype=np.int64)
        np.cumsum(user_lens, out=user_starts[1:])

        # 生成所有切片的 (start_idx, length)
        sequences = []
        for start, length in zip(user_starts[:-1], user_lens):
            # 对每个用户进行滑动窗口切分
            for offset in range(0, length, self.stride):
                seq_len = min(self.seq_len, length - offset)
                if seq_len > 1:
                    sequences.append((start + offset, seq_len))

        self._sequences = np.array(sequences, dtype=np.int64)

    def __len__(self) -> int:
        return len(self._sequences)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        """获取样本"""
        start, length = self._sequences[idx]
        end = start + length

        return {
            "numerical_features": torch.from_numpy(self._numerical[start:end]),
            "categorical_features": torch.from_numpy(self._categorical[start:end]),
            "time_stamps": torch.from_numpy(self._time_stamps[start:end]),
            "card_ids": torch.from_numpy(self._card_ids[start:end]),
            "deck_ids": torch.from_numpy(self._deck_ids[start:end]),
            "ordinal_targets": torch.from_numpy(self._ordinal_targets[start:end]),
            "duration_targets": torch.from_numpy(self._duration_targets[start:end]),
            "seq_len": torch.tensor(length, dtype=torch.long),
        }


class StreamingReviewDataset(IterableDataset):
    """流式数据集
    支持多 Worker 并行读取不同用户文件。
    """

    def __init__(
        self,
        data_dir: Path | str,
        user_ids: list[int],
        feature_config: FeatureConfig = DEFAULT_FEATURE_CONFIG,
        seq_len: int = 256,
        stride: int | None = None,
        shuffle: bool = True,
    ):
        self.data_dir = Path(data_dir)
        self.user_ids = np.array(user_ids)
        self.cfg = feature_config
        self.seq_len = seq_len
        self.stride = stride or seq_len
        self.shuffle = shuffle

        # 预计算所有需要读取的列名
        self.required_columns = list(self.cfg.all_columns)

    def _get_worker_indices(self) -> np.ndarray:
        """计算当前 Worker 需要处理的 user_ids 索引。"""
        worker_info = get_worker_info()
        indices = np.arange(len(self.user_ids))

        if worker_info is not None:
            # 简单的分片逻辑：按 worker_id 间隔采样，或分块
            # 这里使用分块方式，让每个 worker 处理连续的用户块（对文件系统缓存可能更友好）
            per_worker = int(np.ceil(len(self.user_ids) / worker_info.num_workers))
            start = worker_info.id * per_worker
            end = min(start + per_worker, len(self.user_ids))
            indices = indices[start:end]

        return indices

    def __iter__(self) -> Iterator[dict[str, Tensor]]:
        # 1. 获取当前进程分配到的用户索引
        indices = self._get_worker_indices()

        # 2. 如果需要，打乱用户顺序
        if self.shuffle:
            np.random.shuffle(indices)

        local_user_ids = self.user_ids[indices]

        # 3. 逐用户处理
        for user_id in local_user_ids:
            # IO 阶段：加载数据
            data = load_user_data(
                data_dir=self.data_dir, user_id=user_id, columns=self.required_columns
            )
            if data is None:
                continue

            # 预处理阶段：构建特征矩阵 (针对单个用户)
            # 假设任意一列的长度即为总长度
            total_len = len(data[self.cfg.time_feature])
            if total_len <= 1:
                continue

            # 将特征列 Stack 起来 (N, Feat_Dim)
            numerical = np.stack(
                [data[f] for f in self.cfg.numerical_features], axis=1
            ).astype(np.float32)

            categorical = np.stack(
                [data[f] for f in self.cfg.categorical_features], axis=1
            ).astype(np.int64)

            ordinal_tgt = np.stack(
                [data[t] for t in self.cfg.ordinal_targets], axis=1
            ).astype(np.float32)

            # 其他列
            ts = data[self.cfg.time_feature].astype(np.float32)
            c_ids = data[self.cfg.card_id_feature].astype(np.int64)
            d_ids = data[self.cfg.deck_id_feature].astype(np.int64)
            dur_tgt = data[self.cfg.duration_target].astype(np.float32)

            # 4. 生成序列切片
            # 只需要计算 start 索引
            starts = range(0, total_len, self.stride)

            # 如果需要，可以打乱当前用户内部的序列顺序 (可选)
            # if self.shuffle: starts = np.random.permutation(list(starts))

            for start in starts:
                length = min(self.seq_len, total_len - start)
                if length <= 1:
                    continue

                end = start + length

                # 5. Yield Tensor
                yield {
                    "numerical_features": torch.from_numpy(numerical[start:end]),
                    "categorical_features": torch.from_numpy(categorical[start:end]),
                    "time_stamps": torch.from_numpy(ts[start:end]),
                    "card_ids": torch.from_numpy(c_ids[start:end]),
                    "deck_ids": torch.from_numpy(d_ids[start:end]),
                    "ordinal_targets": torch.from_numpy(ordinal_tgt[start:end]),
                    "duration_targets": torch.from_numpy(dur_tgt[start:end]),
                    "seq_len": torch.tensor(length, dtype=torch.long),
                }
