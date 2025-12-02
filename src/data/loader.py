"""数据加载工具函数。"""

from pathlib import Path

import pandas as pd


def load_user_data(data_dir: Path | str, user_id: int) -> pd.DataFrame:
    """加载单个用户的数据。

    Args:
        data_dir: 数据目录路径
        user_id: 用户 ID

    Returns:
        用户的复习记录 DataFrame
    """
    return pd.read_parquet(Path(data_dir) / f"user_id={user_id}.parquet")


def load_all_users_data(data_dir: Path | str) -> pd.DataFrame:
    """加载所有用户的数据。

    Args:
        data_dir: 数据目录路径

    Returns:
        所有用户的复习记录 DataFrame
    """
    data_dir = Path(data_dir)
    parquet_files = sorted(data_dir.glob("user_id=*.parquet"))
    return pd.concat((pd.read_parquet(f) for f in parquet_files), ignore_index=True)
