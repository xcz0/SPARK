"""Utility helpers for lightweight single-sequence prediction."""

from __future__ import annotations

from typing import Literal

import numpy as np
import torch
from torch import Tensor

from .models import SPARKModel
from .utils.masking import (
    create_card_mask,
    create_causal_mask,
    create_deck_mask,
    create_padding_mask,
    create_time_diff_matrix,
)


def _prepare_batch(
    data: dict[str, np.ndarray],
    numerical_features: list[str],
    categorical_features: list[str],
    time_feature: str = "day_offset",
    card_id_feature: str = "card_id",
    deck_id_feature: str = "deck_id",
) -> dict[str, Tensor]:
    """将 numpy 数组转换为模型输入的批次格式。

    Args:
        data: 包含特征数据的字典，每个值为 (seq_len,) 的 numpy 数组
        numerical_features: 数值特征列名列表
        categorical_features: 类别特征列名列表
        time_feature: 时间特征列名
        card_id_feature: 卡片 ID 列名
        deck_id_feature: 卡组 ID 列名

    Returns:
        模型输入字典，所有张量形状为 (1, seq_len, ...)
    """
    # 堆叠数值特征: (seq_len, num_features)
    numerical = np.stack([data[f] for f in numerical_features], axis=1).astype(
        np.float32
    )

    # 堆叠类别特征: (seq_len, num_features)
    categorical = np.stack([data[f] for f in categorical_features], axis=1).astype(
        np.int64
    )

    # 提取其他特征
    time_stamps = data[time_feature].astype(np.float32)
    card_ids = data[card_id_feature].astype(np.int64)
    deck_ids = data[deck_id_feature].astype(np.int64)

    seq_len = len(time_stamps)
    device = torch.device("cpu")

    # 转换为张量并添加批次维度
    numerical_tensor = torch.from_numpy(numerical).unsqueeze(0)  # (1, seq, num_feat)
    categorical_tensor = torch.from_numpy(categorical).unsqueeze(0)
    time_stamps_tensor = torch.from_numpy(time_stamps).unsqueeze(0)  # (1, seq)
    card_ids_tensor = torch.from_numpy(card_ids).unsqueeze(0)
    deck_ids_tensor = torch.from_numpy(deck_ids).unsqueeze(0)
    seq_lens = torch.tensor([seq_len], dtype=torch.long)

    # 创建掩码
    padding_mask = create_padding_mask(seq_lens, seq_len)
    causal_mask = create_causal_mask(seq_len, device=device)
    time_diff = create_time_diff_matrix(time_stamps_tensor)
    card_mask = create_card_mask(card_ids_tensor)
    deck_mask = create_deck_mask(deck_ids_tensor, card_ids_tensor)

    return {
        "numerical_features": numerical_tensor,
        "categorical_features": categorical_tensor,
        "time_stamps": time_stamps_tensor,
        "causal_mask": causal_mask,
        "card_mask": card_mask,
        "deck_mask": deck_mask,
        "time_diff": time_diff,
        "padding_mask": padding_mask,
        "card_ids": card_ids_tensor,
        "deck_ids": deck_ids_tensor,
    }


def _update_sequence_for_next_step(
    data: dict[str, np.ndarray],
    predicted_rating: int,
    next_day_offset: float,
    numerical_features: list[str],
    categorical_features: list[str],
    time_feature: str = "day_offset",
) -> dict[str, np.ndarray]:
    """根据预测结果更新序列，为下一步预测做准备。

    Args:
        data: 当前序列数据
        predicted_rating: 预测的评分 (1-4)
        next_day_offset: 下一次复习的时间偏移
        numerical_features: 数值特征列名
        categorical_features: 类别特征列名
        time_feature: 时间特征列名

    Returns:
        更新后的序列数据
    """
    new_data = {k: v.copy() for k, v in data.items()}

    # 获取最后一个时间点的信息
    last_idx = len(data[time_feature]) - 1
    last_day_offset = data[time_feature][last_idx]

    # 计算新复习的特征
    interval = next_day_offset - last_day_offset

    # 创建新的复习记录
    new_record: dict[str, np.ndarray] = {}

    # 时间特征
    new_record[time_feature] = np.array([next_day_offset], dtype=np.float32)

    # 数值特征：根据预测结果更新
    for feat in numerical_features:
        if feat == "interval":
            new_record[feat] = np.array([interval], dtype=np.float32)
        elif feat == "log_interval":
            new_record[feat] = np.array([np.log1p(max(0, interval))], dtype=np.float32)
        elif feat == "n_reviews":
            new_record[feat] = np.array([data[feat][last_idx] + 1], dtype=np.float32)
        elif feat == "n_success":
            # 评分 > 1 算成功
            success_delta = 1 if predicted_rating > 1 else 0
            new_record[feat] = np.array(
                [data[feat][last_idx] + success_delta], dtype=np.float32
            )
        elif feat == "n_failure":
            # 评分 == 1 算失败
            failure_delta = 1 if predicted_rating == 1 else 0
            new_record[feat] = np.array(
                [data[feat][last_idx] + failure_delta], dtype=np.float32
            )
        elif feat == "log_duration":
            # 耗时预测通常由模型输出，这里用 0 作为占位
            new_record[feat] = np.array([0.0], dtype=np.float32)
        else:
            # 其他数值特征保持最后一个值
            new_record[feat] = np.array([data[feat][last_idx]], dtype=np.float32)

    # 类别特征
    for feat in categorical_features:
        if feat == "prev_review_rating":
            # 上一次评分为预测评分
            new_record[feat] = np.array([predicted_rating], dtype=np.int64)
        elif feat == "state":
            # 根据评分更新状态: 1=Again(Learning), 2=Hard, 3=Good, 4=Easy
            # 简化处理：评分1保持Learning(1)，其他进入Review(2)
            if predicted_rating == 1:
                new_record[feat] = np.array([1], dtype=np.int64)  # Learning
            else:
                new_record[feat] = np.array([2], dtype=np.int64)  # Review
        elif feat == "is_first_review":
            new_record[feat] = np.array([0], dtype=np.int64)  # 不是首次复习
        else:
            # 其他类别特征保持最后一个值
            new_record[feat] = np.array([data[feat][last_idx]], dtype=np.int64)

    # ID 特征保持不变
    for id_feat in ["card_id", "deck_id", "user_id"]:
        if id_feat in data:
            new_record[id_feat] = np.array([data[id_feat][last_idx]], dtype=np.int64)

    # 将新记录追加到序列
    for key in new_data:
        if key in new_record:
            new_data[key] = np.concatenate([new_data[key], new_record[key]])

    return new_data


def predict(
    model: SPARKModel,
    data: dict[str, np.ndarray],
    numerical_features: list[str],
    categorical_features: list[str],
    options: dict | None = None,
) -> dict[str, np.ndarray]:
    """对单个复习序列进行预测。

    Args:
        model: SPARK 模型实例
        data: 复习序列数据，字典格式，每个值为 (seq_len,) 的 numpy 数组
            必须包含的键:
            - 所有 numerical_features 中的特征
            - 所有 categorical_features 中的特征
            - time_feature (默认 "day_offset")
            - card_id_feature (默认 "card_id")
            - deck_id_feature (默认 "deck_id")
        numerical_features: 数值特征列名列表
        categorical_features: 类别特征列名列表
        options: 可选参数字典
            - time_feature: 时间特征列名，默认 "day_offset"
            - card_id_feature: 卡片 ID 列名，默认 "card_id"
            - deck_id_feature: 卡组 ID 列名，默认 "deck_id"
            - steps: 预测步数，默认 1
            - intervals: 多步预测时每步的时间间隔列表 (天数)
            - output_type: 输出类型，"rating" | "expected_rating" | "correct_prob" | "all"
                默认 "all"

    Returns:
        预测结果字典，包含:
        - rating: (steps,) 离散评分预测
        - expected_rating: (steps,) 期望评分
        - correct_prob: (steps,) 回忆正确概率
        - duration: (steps,) 耗时预测 (log scale)
        - rating_probs: (steps, num_thresholds) 评分累积概率
    """
    opts = options or {}
    time_feature = opts.get("time_feature", "day_offset")
    card_id_feature = opts.get("card_id_feature", "card_id")
    deck_id_feature = opts.get("deck_id_feature", "deck_id")
    steps = opts.get("steps", 1)
    intervals = opts.get("intervals", None)
    output_type: Literal["rating", "expected_rating", "correct_prob", "all"] = opts.get(
        "output_type", "all"
    )

    # 设置默认间隔
    if intervals is None:
        intervals = [1.0] * steps
    elif len(intervals) < steps:
        # 扩展间隔列表
        intervals = list(intervals) + [intervals[-1]] * (steps - len(intervals))

    model.eval()
    device = next(model.parameters()).device

    # 存储结果
    results: dict[str, list] = {
        "rating": [],
        "expected_rating": [],
        "correct_prob": [],
        "duration": [],
        "rating_probs": [],
    }

    current_data = {k: v.copy() for k, v in data.items()}

    with torch.no_grad():
        for step in range(steps):
            # 准备批次
            batch = _prepare_batch(
                current_data,
                numerical_features,
                categorical_features,
                time_feature,
                card_id_feature,
                deck_id_feature,
            )

            # 移动到设备
            batch = {k: v.to(device) for k, v in batch.items()}

            # 模型前向传播
            outputs = model(
                numerical_features=batch["numerical_features"],
                categorical_features=batch["categorical_features"],
                time_stamps=batch["time_stamps"],
                causal_mask=batch["causal_mask"],
                card_mask=batch["card_mask"],
                deck_mask=batch["deck_mask"],
                time_diff=batch["time_diff"],
                padding_mask=batch["padding_mask"],
            )

            # 获取最后一个位置的预测（因果模型，最后位置预测下一个）
            last_rating_probs = outputs["rating_probs"][0, -1]  # (num_thresholds,)
            last_duration = outputs["duration_pred"][0, -1]  # scalar

            # 计算各种预测值
            rating = model.predict_rating(last_rating_probs.unsqueeze(0)).item()
            expected_rating = model.predict_expected_rating(
                last_rating_probs.unsqueeze(0)
            ).item()
            correct_prob = model.predict_correct(last_rating_probs.unsqueeze(0)).item()

            results["rating"].append(rating)
            results["expected_rating"].append(expected_rating)
            results["correct_prob"].append(correct_prob)
            results["duration"].append(last_duration.cpu().numpy())
            results["rating_probs"].append(last_rating_probs.cpu().numpy())

            # 如果还有下一步，更新序列
            if step < steps - 1:
                last_day = current_data[time_feature][-1]
                next_day = last_day + intervals[step]
                current_data = _update_sequence_for_next_step(
                    current_data,
                    int(rating),
                    next_day,
                    numerical_features,
                    categorical_features,
                    time_feature,
                )

    # 转换为 numpy 数组
    final_results = {
        "rating": np.array(results["rating"], dtype=np.int64),
        "expected_rating": np.array(results["expected_rating"], dtype=np.float32),
        "correct_prob": np.array(results["correct_prob"], dtype=np.float32),
        "duration": np.array(results["duration"], dtype=np.float32),
        "rating_probs": np.stack(results["rating_probs"], axis=0),
    }

    # 根据 output_type 过滤结果
    if output_type == "all":
        return final_results
    if output_type == "rating":
        return {"rating": final_results["rating"]}
    if output_type == "expected_rating":
        return {"expected_rating": final_results["expected_rating"]}
    if output_type == "correct_prob":
        return {"correct_prob": final_results["correct_prob"]}

    return final_results


def predict_batch(
    model: SPARKModel,
    sequences: list[dict[str, np.ndarray]],
    numerical_features: list[str],
    categorical_features: list[str],
    options: dict | None = None,
) -> list[dict[str, np.ndarray]]:
    """对多个复习序列进行批量预测。

    Args:
        model: SPARK 模型实例
        sequences: 复习序列列表，每个序列是一个字典
        numerical_features: 数值特征列名列表
        categorical_features: 类别特征列名列表
        options: 可选参数，同 predict 函数

    Returns:
        预测结果列表，每个元素对应一个输入序列的预测结果
    """
    return [
        predict(model, seq, numerical_features, categorical_features, options)
        for seq in sequences
    ]
