"""SPARK 模型训练脚本 (Hydra 版本).

Usage:
    python script/train.py                          # 使用默认配置
    python script/train.py model=width128_depth4    # 切换模型配置
    python script/train.py trainer=debug            # 使用实验配置
    python script/train.py model.d_model=256        # 命令行覆盖参数
"""

from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
import torch
from loguru import logger
import hydra
from omegaconf import DictConfig, OmegaConf
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    RichModelSummary,
    RichProgressBar,
)
from lightning.pytorch.loggers import TensorBoardLogger

from spark.data.config import FeatureConfig
from spark.data.datamodule import ReviewDataModule
from spark.models.module import SPARKModule

# 在模块初始化时就加载环境变量，确保 Hydra 配置解析时能访问到
load_dotenv(Path(__file__).parent.parent / ".env", override=True)

# 开启 Tensor Cores 优化 (对于 3090/4090/A100 等显卡有显著加速)
torch.set_float32_matmul_precision("high")


def create_datamodule(cfg: DictConfig) -> ReviewDataModule:
    """从 Hydra 配置创建数据模块。"""
    data_cfg = cfg.data
    features_cfg = data_cfg.features

    feature_config = FeatureConfig(
        numerical_features=tuple(features_cfg.numerical_features),
        categorical_features=tuple(features_cfg.categorical_features),
        categorical_vocab_sizes=dict(features_cfg.categorical_vocab_sizes),
        time_feature=features_cfg.time_feature,
        card_id_feature=features_cfg.card_id_feature,
        deck_id_feature=features_cfg.deck_id_feature,
        ordinal_targets=tuple(features_cfg.ordinal_targets),
        duration_target=features_cfg.duration_target,
    )

    return ReviewDataModule(
        data_dir=data_cfg.processed_dir,
        seq_len=data_cfg.window_size,
        stride=data_cfg.stride,
        batch_size=data_cfg.batch_size,
        num_workers=data_cfg.num_workers,
        pin_memory=data_cfg.pin_memory,
        train_ratio=data_cfg.train_ratio,
        val_ratio=data_cfg.val_ratio,
        streaming=data_cfg.streaming,
        seed=data_cfg.seed,
        feature_config=feature_config,
    )


def create_model(cfg: DictConfig, data_module: ReviewDataModule) -> SPARKModule:
    """从 Hydra 配置创建模型。"""
    model_cfg = cfg.model

    return SPARKModule(
        d_model=model_cfg.d_model,
        n_heads=model_cfg.n_heads,
        depth=model_cfg.depth,
        dropout=model_cfg.dropout,
        learning_rate=cfg.optimizer.learning_rate,
        num_numerical_features=data_module.num_numerical_features,
        categorical_vocab_sizes=data_module.categorical_vocab_sizes,
        card_head_ratio=model_cfg.head_ratios.card,
        deck_head_ratio=model_cfg.head_ratios.deck,
    )


def create_callbacks(cfg: DictConfig) -> list:
    """创建训练回调。"""
    trainer_cfg = cfg.trainer
    checkpoint_cfg = trainer_cfg.checkpoint

    return [
        ModelCheckpoint(
            filename="spark-{epoch:02d}-{val/loss:.4f}",
            monitor=checkpoint_cfg.monitor,
            mode=checkpoint_cfg.mode,
            save_top_k=checkpoint_cfg.save_top_k,
            save_last=True,
        ),
        EarlyStopping(
            monitor=checkpoint_cfg.monitor,
            patience=trainer_cfg.patience,
            mode=checkpoint_cfg.mode,
        ),
        LearningRateMonitor(logging_interval="step"),
        RichProgressBar(),
        RichModelSummary(max_depth=2),
    ]


_PrecisionLiteral = Literal[
    "16-mixed", "bf16-mixed", "32", "64", "16-true", "bf16-true"
]
PRECISION_MAP: dict[int | str, _PrecisionLiteral] = {
    16: "16-mixed",
    "bf16": "bf16-mixed",
}


def get_precision_str(precision: int | str) -> _PrecisionLiteral:
    """转换精度配置为 Lightning 格式。"""
    if precision in PRECISION_MAP:
        return PRECISION_MAP[precision]
    # 默认返回 32 精度
    return "32"


def format_param_count(total_params: int) -> str:
    """格式化参数量显示。"""
    if total_params >= 1e6:
        return f"{total_params / 1e6:.1f}M"
    return f"{total_params / 1e3:.1f}k"


def find_checkpoint(root_dir: Path, exp_name: str, exp_version: str) -> str | None:
    """查找最近的检查点用于断点续训。"""
    last_ckpt_path = root_dir / exp_name / exp_version / "checkpoints" / "last.ckpt"
    if last_ckpt_path.exists():
        logger.info(f"检测到上次训练断点，将从 {last_ckpt_path} 恢复训练")
        return str(last_ckpt_path)
    return None


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """主训练函数。"""
    # 解析配置中的环境变量
    OmegaConf.resolve(cfg)

    logger.info(f"配置:\n{OmegaConf.to_yaml(cfg)}")

    seed_everything(cfg.experiment.seed, workers=True)

    trainer_cfg = cfg.trainer
    default_root_dir = Path(trainer_cfg.default_root_dir)
    default_root_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info(f"SPARK Training | Output: {default_root_dir}")
    logger.info("=" * 60)

    data_module = create_datamodule(cfg)
    data_module.setup("fit")

    model = create_model(cfg, data_module)

    total_params = sum(p.numel() for p in model.parameters())
    size_str = format_param_count(total_params)

    exp_name = cfg.experiment.name
    exp_version = f"{size_str}-d{cfg.model.d_model}-depth{cfg.model.depth}"

    logger.info(f"实验: {exp_name}/{exp_version}")
    logger.info(f"模型参数量: {size_str}")

    tb_logger = TensorBoardLogger(
        save_dir=default_root_dir,
        name=exp_name,
        version=exp_version,
    )

    # 优先检查 断点续训
    ckpt_path = None
    if cfg.trainer.get("resume", True):
        ckpt_path = find_checkpoint(default_root_dir, exp_name, exp_version)

    trainer = Trainer(
        default_root_dir=default_root_dir,
        max_epochs=trainer_cfg.max_epochs,
        precision=get_precision_str(trainer_cfg.precision),
        callbacks=create_callbacks(cfg),
        logger=tb_logger,
        gradient_clip_val=trainer_cfg.gradient_clip_val,
        accumulate_grad_batches=trainer_cfg.accumulate_grad_batches,
        val_check_interval=trainer_cfg.val_check_interval,
        log_every_n_steps=trainer_cfg.log_every_n_steps,
        deterministic=True,
        enable_model_summary=False,
    )

    logger.info("开始训练...")
    trainer.fit(model, datamodule=data_module, ckpt_path=ckpt_path)

    logger.info("运行测试...")
    trainer.test(model, datamodule=data_module, ckpt_path="best")

    final_path = default_root_dir / "final_model.ckpt"
    trainer.save_checkpoint(final_path)
    logger.info(f"训练流程结束，最终状态已保存: {final_path}")


if __name__ == "__main__":
    main()
