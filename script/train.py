"""SPARK 模型训练脚本。

Usage:
    python script/train.py [--config CONFIG_DIR] [--epochs N] [--batch-size N] [--lr LR]
                           [--data-dir DIR] [--output-dir DIR] [--resume CKPT]

Example:
    python script/train.py --epochs 20 --batch-size 64
"""

import argparse
import sys
from pathlib import Path

from lightning import Trainer, seed_everything
import yaml
from loguru import logger
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
)
from lightning.pytorch.loggers import TensorBoardLogger

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from spark.data.datamodule import ReviewDataModule
from spark.models.module import SPARKModule


def load_config(config_dir: Path) -> dict:
    """加载配置文件。

    Args:
        config_dir: 配置文件目录

    Returns:
        合并后的配置字典
    """
    config = {}

    config_files = ["data.yaml", "model.yaml", "trainer.yaml"]
    for filename in config_files:
        config_path = config_dir / filename
        if config_path.exists():
            with open(config_path) as f:
                file_config = yaml.safe_load(f)
                if file_config:
                    config.update(file_config)
            logger.info(f"加载配置: {config_path}")

    return config


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="SPARK 模型训练脚本",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # 配置路径
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs",
        help="配置文件目录",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="数据目录（覆盖配置文件）",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="输出目录（模型检查点、日志等）",
    )

    # 训练参数
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="训练轮数（覆盖配置文件）",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="批次大小（覆盖配置文件）",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="学习率",
    )

    # 训练控制
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="从检查点恢复训练",
    )

    parser.add_argument(
        "--devices",
        type=int,
        default=1,
        help="设备数量",
    )

    return parser.parse_args()


def main() -> None:
    """主训练流程。"""
    args = parse_args()

    # 加载配置
    config = load_config(args.config)
    data_config = config.get("data", {})
    model_config = config.get("model", {})
    trainer_config = config.get("trainer", {})

    # 设置随机种子
    seed = data_config.get("seed", 42)
    seed_everything(seed, workers=True)

    # 模型参数
    d_model = model_config.get("d_model", 128)
    n_heads = model_config.get("n_heads", 8)
    depth = model_config.get("depth", 4)
    dropout = model_config.get("dropout", 0.1)
    head_ratios = model_config.get("head_ratios", {})
    card_head_ratio = head_ratios.get("card", model_config.get("card_head_ratio", 0.5))
    deck_head_ratio = head_ratios.get("deck", model_config.get("deck_head_ratio", 0.25))
    global_head_ratio = head_ratios.get(
        "global", 1.0 - card_head_ratio - deck_head_ratio
    )
    max_epochs = args.epochs or trainer_config.get("max_epochs", 20)
    learning_rate = args.lr or 1e-4

    # 确定输出目录
    if args.output_dir:
        output_dir = args.output_dir
    elif "default_root_dir" in trainer_config:
        output_dir = Path(trainer_config["default_root_dir"])
    else:
        output_dir = PROJECT_ROOT / "outputs"

    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)

    # 构建数据模块覆盖参数
    data_overrides = {}
    if args.data_dir:
        data_overrides["data_dir"] = args.data_dir
    if args.batch_size:
        data_overrides["batch_size"] = args.batch_size

    # 初始化数据模块
    logger.info("初始化数据模块...")
    data_module = ReviewDataModule.from_config(
        args.config / "data.yaml",
        overrides=data_overrides,
    )

    logger.info("=" * 60)
    logger.info("SPARK 模型训练")
    logger.info("=" * 60)
    logger.info(f"数据目录: {data_module.data_dir}")
    logger.info(f"输出目录: {output_dir}")
    logger.info(f"批次大小: {data_module.hparams.batch_size}")
    logger.info(f"序列长度: {data_module.hparams.seq_len}")
    logger.info(f"模型维度: {d_model}")
    logger.info(f"注意力头数: {n_heads}")
    logger.info(f"Transformer 层数: {depth}")
    logger.info(
        f"注意力头比例: card={card_head_ratio:.2f} deck={deck_head_ratio:.2f} "
        f"global={global_head_ratio:.2f}"
    )
    logger.info(f"学习率: {learning_rate}")
    logger.info(f"训练轮数: {max_epochs}")
    logger.info("=" * 60)

    # 设置数据以获取特征信息
    data_module.setup("fit")

    # 初始化模型
    logger.info("初始化模型...")
    model = SPARKModule(
        d_model=d_model,
        n_heads=n_heads,
        depth=depth,
        dropout=dropout,
        num_numerical_features=data_module.num_numerical_features,
        categorical_vocab_sizes=data_module.categorical_vocab_sizes,
        card_head_ratio=card_head_ratio,
        deck_head_ratio=deck_head_ratio,
        learning_rate=learning_rate,
    )

    # 打印模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"模型参数量: {total_params:,} (可训练: {trainable_params:,})")

    # 配置回调
    callbacks = [
        ModelCheckpoint(
            dirpath=output_dir / "checkpoints",
            filename="spark-{epoch:02d}-{val/loss:.4f}",
            monitor="val/loss",
            mode="min",
            save_top_k=3,
            save_last=True,
        ),
        EarlyStopping(
            monitor="val/loss",
            patience=trainer_config.get("patience", 5),
            mode="min",
            verbose=True,
        ),
        LearningRateMonitor(logging_interval="step"),
        RichProgressBar(),
    ]

    # 配置日志器
    loggers = []

    # 使用 TensorBoard 记录日志
    tb_logger = TensorBoardLogger(
        save_dir=output_dir,
        name="lightning_logs",
    )
    loggers.append(tb_logger)
    logger.info("TensorBoard 日志已启用")

    # 初始化训练器
    trainer = Trainer(
        max_epochs=max_epochs,
        devices=args.devices,
        precision=str(trainer_config.get("precision", 32)),
        callbacks=callbacks,
        logger=loggers if loggers else None,
        default_root_dir=output_dir,
        gradient_clip_val=trainer_config.get("gradient_clip_val", 1.0),
        accumulate_grad_batches=trainer_config.get("accumulate_grad_batches", 1),
        val_check_interval=trainer_config.get("val_check_interval", 1.0),
        log_every_n_steps=10,
        deterministic=True,
    )

    # 开始训练
    logger.info("开始训练...")
    trainer.fit(
        model,
        datamodule=data_module,
        ckpt_path=args.resume,
    )

    # 测试
    logger.info("运行测试...")
    trainer.test(model, datamodule=data_module)

    # 保存最终模型
    final_model_path = output_dir / "final_model.ckpt"
    trainer.save_checkpoint(final_model_path)
    logger.info(f"最终模型已保存至: {final_model_path}")

    logger.info("训练完成！")


if __name__ == "__main__":
    main()
