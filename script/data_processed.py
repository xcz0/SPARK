from pathlib import Path
import logging
from multiprocessing import Pool, cpu_count
from functools import partial
from dataclasses import dataclass

import pandas as pd
import numpy as np
from tqdm import tqdm

# 配置日志
logging.basicConfig(
    filename="summary_errors.log",
    level=logging.ERROR,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

RAW_DATA_DIR = Path("./data/raw/")
OUTPUT_DIR = Path("./data/processed/")


@dataclass
class ProcessResult:
    """处理结果数据类"""

    status: int  # 0=success, 1=skipped, 2=error
    message: str
    log_nums: int
    user_id: int
    user_stats: dict


def summarize_data(
    log_data: pd.DataFrame, card_data: pd.DataFrame | None, user_id: int
) -> pd.DataFrame:
    """
    Merges log data with card and deck information to provide a comprehensive summary.

    Parameters:
    log_data (pd.DataFrame): DataFrame containing review logs.
    card_data (pd.DataFrame | None): DataFrame containing card information, or None if not available.
    user_id (int): User ID for this data.

    Returns:
    pd.DataFrame: Merged DataFrame with log and card information.
    """
    log_data["user_id"] = user_id

    # 预计算分组对象，避免重复分组
    day_group = log_data.groupby("day_offset", sort=False)
    card_group = log_data.groupby("card_id", sort=False)

    # 对每个 day_offset 分组,累计 duration(不包含当前行)
    log_data["time_spent_today"] = day_group["duration"].cumsum().shift(1, fill_value=0)
    # 这是今天复习的第几张卡（同一 day_offset 内的顺序，从 1 开始）
    log_data["card_index_today"] = (day_group.cumcount() + 1).astype("int32")
    log_data["card_review_count"] = card_group.cumcount() + 1

    # 添加rating_level特征 (3维向量编码) - 使用向量化操作
    rating = log_data["rating"]
    log_data["rating_gt1"] = (rating > 1).astype("int8")
    log_data["rating_gt2"] = (rating > 2).astype("int8")
    log_data["rating_gt3"] = (rating > 3).astype("int8")

    # 添加is_first_review特征
    is_first_review = log_data["elapsed_seconds"] == -1
    log_data["is_first_review"] = is_first_review

    # 添加对数变换特征 - 批量处理
    duration = log_data["duration"]
    log_data["log_duration"] = np.log1p(duration)
    log_data["log_time_spent_today"] = np.log1p(log_data["time_spent_today"])

    # 处理 elapsed 相关的对数变换
    elapsed_seconds_clipped = log_data["elapsed_seconds"].clip(lower=0)
    elapsed_days_clipped = log_data["elapsed_days"].clip(lower=0)
    log_data["log_elapsed_seconds"] = np.where(
        is_first_review, np.inf, np.log1p(elapsed_seconds_clipped)
    )
    log_data["log_elapsed_days"] = np.where(
        is_first_review, np.inf, np.log1p(elapsed_days_clipped)
    )

    # 显式历史特征 - 该卡片上一次复习的评分和耗时
    log_data["last_rating_on_card"] = (
        card_group["rating"].shift(1).fillna(0).astype("int8")
    )
    log_data["log_last_duration_on_card"] = (
        card_group["log_duration"].shift(1).fillna(np.inf)
    )

    # Session 上下文特征 - 全局序列上一次复习的信息
    log_data["prev_review_rating"] = rating.shift(1).fillna(0).astype("int8")
    log_data["log_prev_review_duration"] = (
        log_data["log_duration"].shift(1).fillna(np.inf)
    )
    log_data["same_card_as_prev"] = log_data["card_id"] == log_data["card_id"].shift(1)

    # Merge log data with card data to get deck_id
    if card_data is None or card_data.empty:
        log_data["deck_id"] = 0
    else:
        log_data = log_data.merge(
            card_data[["card_id", "deck_id"]], on="card_id", how="left"
        )
        log_data["deck_id"] = log_data["deck_id"].fillna(0).astype("int32")

    log_data["is_correct"] = (log_data["rating"] > 1).astype("int8")

    return log_data


def merge_datasets(summaries_dir: Path, batch_size: int = 100) -> None:
    """
    合并处理后的数据文件，每batch_size个文件合并成一个文件

    Parameters:
    summaries_dir (Path): 包含处理后数据的目录
    batch_size (int): 每次合并的文件数量，默认100
    """
    parquet_files = sorted(
        summaries_dir.glob("user_id=*.parquet"),
        key=lambda x: int(x.stem.split("=")[1]),
    )

    if not parquet_files:
        print("未找到任何parquet文件，跳过合并步骤")
        return

    total_files = len(parquet_files)
    num_batches = (total_files + batch_size - 1) // batch_size

    print(
        f"\n开始合并数据集...\n总文件数: {total_files}, 批次大小: {batch_size}, 批次数: {num_batches}"
    )

    for batch_idx in tqdm(range(num_batches), desc="合并批次", unit="batch"):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, total_files)
        batch_files = parquet_files[start_idx:end_idx]

        # 使用列表推导式简化读取
        batch_dfs = [pd.read_parquet(f) for f in batch_files]
        batch_df = pd.concat(batch_dfs, ignore_index=True)

        output_file = summaries_dir / f"batch_{batch_idx:04d}.parquet"
        batch_df.to_parquet(
            output_file, index=False, engine="pyarrow", compression="snappy"
        )
        print(f"批次 {batch_idx + 1}: {len(batch_df)} 条记录 -> {output_file.name}")


def process_single_user(user_id: int, summaries_dir: Path) -> ProcessResult:
    """处理单个用户的数据"""
    output_file = summaries_dir / f"user_id={user_id}.parquet"

    try:
        log_path = RAW_DATA_DIR / "revlogs" / f"user_id={user_id}" / "data.parquet"
        log_data = pd.read_parquet(log_path)
        log_nums = len(log_data)

        # 尝试读取cards文件
        card_path = RAW_DATA_DIR / "cards" / f"user_id={user_id}" / "data.parquet"
        card_data = pd.read_parquet(card_path) if card_path.exists() else None

        summary_df = summarize_data(log_data, card_data, user_id)
        summary_df.to_parquet(
            output_file, index=False, engine="pyarrow", compression="snappy"
        )

        # 计算用户统计信息
        user_stats = {
            "user_id": user_id,
            "user_mean_rating": summary_df["rating"].mean(),
            "user_std_rating": summary_df["rating"].std(),
            "user_mean_log_duration": summary_df["log_duration"].mean(),
            "user_std_log_duration": summary_df["log_duration"].std(),
            "total_reviews": log_nums,
        }

        return ProcessResult(0, "", log_nums, user_id, user_stats)

    except Exception as e:
        logging.error(f"User {user_id} - Error: {e}")
        return ProcessResult(2, str(e), 0, user_id, {})


def compute_global_stats(summaries_dir: Path, user_range: range) -> dict | None:
    """
    计算全局统计量（使用增量方式避免内存溢出）

    Parameters:
    summaries_dir (Path): 数据目录
    user_range (range): 用户ID范围

    Returns:
    dict | None: 全局统计量字典，失败返回None
    """
    # 使用 Welford 算法进行在线计算均值和方差
    n_elapsed, mean_elapsed, m2_elapsed = 0, 0.0, 0.0
    n_duration, mean_duration, m2_duration = 0, 0.0, 0.0

    for user_id in tqdm(user_range, desc="计算全局统计量"):
        user_file = summaries_dir / f"user_id={user_id}.parquet"
        if not user_file.exists():
            continue

        try:
            user_df = pd.read_parquet(
                user_file, columns=["log_elapsed_seconds", "log_duration"]
            )

            # 处理 log_elapsed_seconds
            valid_elapsed = (
                user_df["log_elapsed_seconds"]
                .replace([np.inf, -np.inf], np.nan)
                .dropna()
                .values
            )
            for x in valid_elapsed:
                n_elapsed += 1
                delta = x - mean_elapsed
                mean_elapsed += delta / n_elapsed
                m2_elapsed += delta * (x - mean_elapsed)

            # 处理 log_duration
            valid_duration = (
                user_df["log_duration"]
                .replace([np.inf, -np.inf], np.nan)
                .dropna()
                .values
            )
            for x in valid_duration:
                n_duration += 1
                delta = x - mean_duration
                mean_duration += delta / n_duration
                m2_duration += delta * (x - mean_duration)

        except Exception as e:
            logging.error(f"读取用户 {user_id} 数据计算全局统计时出错: {e}")

    if n_elapsed < 2 or n_duration < 2:
        return None

    return {
        "mean_log_elapsed": mean_elapsed,
        "std_log_elapsed": np.sqrt(m2_elapsed / (n_elapsed - 1)),
        "mean_log_duration": mean_duration,
        "std_log_duration": np.sqrt(m2_duration / (n_duration - 1)),
    }


if __name__ == "__main__":
    summaries_dir = OUTPUT_DIR
    summaries_dir.mkdir(parents=True, exist_ok=True)

    # 使用多进程并行处理
    num_workers = max(1, cpu_count() - 1)
    process_func = partial(process_single_user, summaries_dir=summaries_dir)

    with Pool(num_workers) as pool:
        results: list[ProcessResult] = list(
            tqdm(
                pool.imap_unordered(process_func, range(1, 10001)),
                total=10000,
                desc="处理用户数据",
                unit="user",
            )
        )

    # 统计结果
    success_results = [r for r in results if r.status == 0]
    error_results = [r for r in results if r.status == 2]

    # 生成 user_stats.csv
    user_stats_data = [r.user_stats for r in success_results if r.user_stats]
    if user_stats_data:
        pd.DataFrame(user_stats_data).to_csv(
            summaries_dir / "user_stats.csv", index=False
        )
        print(f"已生成 user_stats.csv，包含 {len(user_stats_data)} 个用户的偏好统计")

    # 生成 meta.csv（使用增量算法计算全局统计量）
    print("\n计算全局统计量（使用前8000个用户数据）...")
    meta_dict = compute_global_stats(summaries_dir, range(1, 8001))
    if meta_dict:
        pd.DataFrame([meta_dict]).to_csv(summaries_dir / "meta.csv", index=False)
        print("已生成 meta.csv，包含全局统计量:")
        for key, value in meta_dict.items():
            print(f"  {key}: {value:.4f}")

    # 输出处理结果摘要
    print(
        f"\n处理完成: 成功 {len(success_results)} 个用户, 失败 {len(error_results)} 个用户"
    )
    if error_results:
        print("错误详情已记录到 summary_errors.log")
