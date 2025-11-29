from pathlib import Path
import logging
from multiprocessing import Pool, cpu_count
from functools import partial
import warnings

import pandas as pd
import numpy as np
from tqdm import tqdm

# 配置日志
logging.basicConfig(
    filename="summary_errors.log",
    level=logging.ERROR,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

RAW_DATA_DIR = Path("./data")
OUTPUT_DIR = Path("./processed/")


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
    # 添加user_id字段
    log_data["user_id"] = user_id

    # 对每个 day_offset 分组,累计 duration(不包含当前行)
    log_data["cumulative"] = (
        log_data.groupby("day_offset", sort=False)["duration"]
        .cumsum()
        .shift(1, fill_value=0)
    )
    log_data["review_count"] = log_data.groupby("card_id", sort=False).cumcount() + 1

    # 添加rating_level特征 (3维向量编码)
    log_data["rating_level_1"] = (log_data["rating"] > 1).astype("int8")
    log_data["rating_level_2"] = (log_data["rating"] > 2).astype("int8")
    log_data["rating_level_3"] = (log_data["rating"] > 3).astype("int8")

    # 添加is_first_review特征
    log_data["is_first_review"] = log_data["elapsed_seconds"] == -1

    # 添加对数变换特征
    log_data["log_duration"] = np.log1p(log_data["duration"])
    log_data["log_cumulative"] = np.log1p(log_data["cumulative"])
    log_data["log_elapsed_seconds"] = np.log1p(
        log_data["elapsed_seconds"].clip(lower=0)
    )
    log_data["log_elapsed_days"] = np.log1p(log_data["elapsed_days"].clip(lower=0))
    log_data.loc[
        log_data["is_first_review"], ["log_elapsed_seconds", "log_elapsed_days"]
    ] = np.inf

    # 显式历史特征 - 该卡片上一次复习的评分和耗时
    log_data["last_rating_on_card"] = (
        log_data.groupby("card_id", sort=False)["rating"]
        .shift(1)
        .fillna(0)
        .astype("int8")
    )
    log_data["last_log_duration_on_card"] = (
        log_data.groupby("card_id", sort=False)["log_duration"].shift(1).fillna(np.inf)
    )

    # Session 上下文特征 - 全局序列上一次复习的信息
    log_data["prev_rating"] = log_data["rating"].shift(1).fillna(0).astype("int8")
    log_data["prev_log_duration"] = log_data["log_duration"].shift(1).fillna(np.inf)
    log_data["is_same_card_as_prev"] = log_data["card_id"] == log_data["card_id"].shift(
        1
    )

    # Merge log data with card data to get deck_id (只选择需要的列)
    # 如果card_data为空,使用left join并将deck_id填充为0
    if card_data is None or len(card_data) == 0:
        log_data["deck_id"] = 0
        full_data = log_data
    else:
        full_data = log_data.merge(
            card_data[["card_id", "deck_id"]], on="card_id", how="left"
        )
        full_data["deck_id"] = full_data["deck_id"].fillna(0).astype("int32")

    full_data["is_correct"] = (full_data["rating"] > 1).astype("int8")

    return full_data


def merge_datasets(summaries_dir: Path, batch_size: int = 100) -> None:
    """
    合并处理后的数据文件，每batch_size个文件合并成一个文件

    Parameters:
    summaries_dir (Path): 包含处理后数据的目录
    batch_size (int): 每次合并的文件数量，默认100
    """
    # 获取所有user_id={id}.parquet文件
    parquet_files = sorted(
        summaries_dir.glob("user_id=*.parquet"), key=lambda x: int(x.stem.split("=")[1])
    )

    if not parquet_files:
        print("未找到任何parquet文件，跳过合并步骤")
        return

    total_files = len(parquet_files)
    num_batches = (total_files + batch_size - 1) // batch_size  # 向上取整

    print("\n开始合并数据集...")
    print(f"总文件数: {total_files}")
    print(f"批次大小: {batch_size}")
    print(f"将生成批次数: {num_batches}")

    # 分批合并
    for batch_idx in tqdm(range(num_batches), desc="读取批次", unit="file"):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, total_files)
        batch_files = parquet_files[start_idx:end_idx]

        print(
            f"\n正在合并批次 {batch_idx + 1}/{num_batches} (文件 {start_idx + 1}-{end_idx})..."
        )
        batch_dfs = []
        for file in batch_files:
            batch_dfs.append(pd.read_parquet(file))

        batch_df = pd.concat(batch_dfs, ignore_index=True)
        output_file = summaries_dir / f"batch_{batch_idx:04d}.parquet"
        batch_df.to_parquet(
            output_file,
            index=False,
            engine="pyarrow",
            compression="snappy",
        )
        print(
            f"批次 {batch_idx + 1} 已保存: {len(batch_df)} 条记录 -> {output_file.name}"
        )


def process_single_user(
    user_id: int, summaries_dir: Path
) -> tuple[int, str, int, int, dict]:
    """
    处理单个用户的数据

    Returns:
    tuple: (status, message, log_nums, user_id, user_stats_dict) - status: 0=success, 1=skipped, 2=error
    """
    try:
        output_file = summaries_dir / f"user_id={user_id}.parquet"
        # 检查输出文件是否已存在
        # if output_file.exists():
        #     读取已存在文件的行数用于统计
        #     existing_data = pd.read_parquet(output_file)
        #     return (1, "", len(existing_data), user_id, {})

        log_data = pd.read_parquet(
            RAW_DATA_DIR / "revlogs" / f"user_id={user_id}" / "data.parquet"
        )
        log_nums = len(log_data)

        # 尝试读取cards文件,如果不存在则设为None
        try:
            card_data = pd.read_parquet(
                RAW_DATA_DIR / "cards" / f"user_id={user_id}" / "data.parquet"
            )
        except FileNotFoundError:
            card_data = None

        # 捕获 RuntimeWarning: divide by zero encountered in log1p
        with warnings.catch_warnings(record=True):
            warnings.filterwarnings(
                "error", category=RuntimeWarning, message=".*divide by zero.*"
            )

            try:
                summary_df = summarize_data(log_data, card_data, user_id)
                summary_df.to_parquet(
                    output_file, index=False, engine="pyarrow", compression="snappy"
                )

                # 计算用户统计信息
                user_stats_dict = {
                    "user_id": user_id,
                    "user_mean_rating": summary_df["rating"].mean(),
                    "user_std_rating": summary_df["rating"].std(),
                    "user_mean_log_duration": summary_df["log_duration"].mean(),
                    "user_std_log_duration": summary_df["log_duration"].std(),
                    "total_reviews": log_nums,
                }

                return (0, "", log_nums, user_id, user_stats_dict)
            except RuntimeWarning:
                # 如果遇到 divide by zero 警告，跳过此用户并记录
                error_msg = "RuntimeWarning - divide by zero encountered in log1p"
                logging.error(f"User {user_id} - {error_msg}")
                return (2, error_msg, 0, user_id, {})

    except Exception as e:
        return (2, str(e), 0, user_id, {})


if __name__ == "__main__":
    summaries_dir = OUTPUT_DIR
    summaries_dir.mkdir(parents=True, exist_ok=True)

    success_count = 0
    error_count = 0
    skipped_count = 0
    user_stats_list = []  # 用于存储用户统计信息
    description_list = []  # 用于存储description信息

    # 使用多进程并行处理
    num_workers = max(1, cpu_count() - 1)  # 保留一个核心给系统

    with Pool(num_workers) as pool:
        # 创建部分函数以传递 summaries_dir
        process_func = partial(process_single_user, summaries_dir=summaries_dir)

        # 并行处理所有用户
        results = list(
            tqdm(
                pool.imap_unordered(process_func, range(1, 10001)),
                total=10000,
                desc="Processing users",
                unit="user",
            )
        )

    # 统计结果并收集用户信息
    for status, error_msg, log_nums, user_id, user_stats_dict in results:
        if status == 0:
            success_count += 1
            description_list.append({"user_id": user_id, "log_nums": log_nums})
            user_stats_list.append(user_stats_dict)
        elif status == 1:
            skipped_count += 1
            description_list.append({"user_id": user_id, "log_nums": log_nums})
        else:
            error_count += 1
            if error_msg:
                logging.error(f"User {user_id} - Error: {error_msg}")

    # 生成description.csv
    if description_list:
        description_df = pd.DataFrame(description_list)
        description_df.to_csv(summaries_dir / "description.csv", index=False)
        print(f"\n已生成 description.csv,包含 {len(description_list)} 个用户的统计信息")

    # 生成user_stats.csv
    if user_stats_list:
        user_stats_df = pd.DataFrame(user_stats_list)
        user_stats_df.to_csv(summaries_dir / "user_stats.csv", index=False)
        print(f"已生成 user_stats.csv,包含 {len(user_stats_list)} 个用户的偏好统计")

    # 生成meta.csv (仅使用前8000个用户数据计算全局统计量)
    print("\n计算全局统计量 (使用前8000个用户数据)...")
    global_log_elapsed = []
    global_log_duration = []

    for user_id in tqdm(range(1, 8001), desc="读取用户数据计算全局统计"):
        user_file = summaries_dir / f"user_id={user_id}.parquet"
        if user_file.exists():
            try:
                user_df = pd.read_parquet(
                    user_file, columns=["log_elapsed_seconds", "log_duration"]
                )
                # 过滤掉无穷大值
                valid_elapsed = (
                    user_df["log_elapsed_seconds"]
                    .replace([np.inf, -np.inf], np.nan)
                    .dropna()
                )
                valid_duration = (
                    user_df["log_duration"].replace([np.inf, -np.inf], np.nan).dropna()
                )
                global_log_elapsed.extend(valid_elapsed.tolist())
                global_log_duration.extend(valid_duration.tolist())
            except Exception as e:
                logging.error(f"读取用户 {user_id} 数据计算全局统计时出错: {e}")

    if global_log_elapsed and global_log_duration:
        meta_dict = {
            "mean_log_elapsed": np.mean(global_log_elapsed),
            "std_log_elapsed": np.std(global_log_elapsed),
            "mean_log_duration": np.mean(global_log_duration),
            "std_log_duration": np.std(global_log_duration),
        }
        meta_df = pd.DataFrame([meta_dict])
        meta_df.to_csv(summaries_dir / "meta.csv", index=False)
        print(f"已生成 meta.csv,包含全局统计量")
        print(f"  mean_log_elapsed: {meta_dict['mean_log_elapsed']:.4f}")
        print(f"  std_log_elapsed: {meta_dict['std_log_elapsed']:.4f}")
        print(f"  mean_log_duration: {meta_dict['mean_log_duration']:.4f}")
        print(f"  std_log_duration: {meta_dict['std_log_duration']:.4f}")

    print(
        f"\n处理完成: 成功 {success_count} 个用户, 跳过 {skipped_count} 个用户, 失败 {error_count} 个用户"
    )
    if error_count > 0:
        print("错误详情已记录到 summary_errors.log")

    # 合并数据集为训练集和验证集
    merge_datasets(summaries_dir)
