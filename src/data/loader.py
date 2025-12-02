"""数据加载工具函数。"""

from pathlib import Path

import pyarrow.parquet as pq
import numpy as np


def load_user_data(
    data_dir: Path | str, user_id: int, columns: list[str]
) -> dict[str, np.ndarray]:
    """
    直接通过 PyArrow 读取为 Numpy 数组。
    只读取需要的列，减少 IO。
    """
    file_path = Path(data_dir) / f"user_id={user_id}.parquet"

    table = pq.read_table(file_path, columns=columns)

    return {
        name: col.to_numpy() for name, col in zip(table.column_names, table.columns)
    }


def load_all_users_data(
    data_dir: Path | str, columns: list[str]
) -> dict[str, np.ndarray]:
    """加载所有用户的数据。

    Args:
        data_dir: 数据目录路径
        columns: 需要加载的列名列表

    Returns:
        包含所有用户数据的字典，键为列名，值为 numpy 数组
    """
    data_dir = Path(data_dir)
    parquet_files = list(data_dir.glob("*.parquet"))
    table = pq.read_table(parquet_files, columns=columns)

    return {
        name: col.to_numpy() for name, col in zip(table.column_names, table.columns)
    }
