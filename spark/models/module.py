"""PyTorch Lightning 模块封装。

将 SPARK 模型封装为 LightningModule，提供：
- 训练/验证/测试步骤
- 优化器配置
- 指标计算
"""

from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor
from lightning import LightningModule
from torchmetrics import MetricCollection, MeanAbsoluteError, MeanSquaredError
from torchmetrics.classification import MultilabelAccuracy, MultilabelAUROC, BinaryAUROC

from .architecture import ModelConfig, SPARKModel
from .losses import CombinedLoss
from ..utils.metrics import OrdinalAccuracy


class SPARKModule(LightningModule):
    """SPARK 模型的 Lightning 模块封装。"""

    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 8,
        depth: int = 4,
        d_ff: int | None = None,
        dropout: float = 0.1,
        num_numerical_features: int = 6,
        categorical_vocab_sizes: dict[str, int] | None = None,
        categorical_embed_dim: int = 16,
        num_rating_classes: int = 4,
        card_head_ratio: float = 0.5,
        deck_head_ratio: float = 0.25,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 1000,
        rating_loss_weight: float = 1.0,
        duration_loss_weight: float = 0.1,
    ):
        """初始化模块。

        Args:
            d_model: 模型维度
            n_heads: 注意力头数
            depth: Transformer 层数
            dropout: Dropout 概率
            num_numerical_features: 数值特征数量
            categorical_vocab_sizes: 类别特征词表大小
            categorical_embed_dim: 类别嵌入维度
            num_rating_classes: 评分类别数
            card_head_ratio: 卡片头占比
            deck_head_ratio: 卡组头占比
            learning_rate: 学习率
            weight_decay: 权重衰减
            warmup_steps: 预热步数
            rating_loss_weight: 评分损失权重
            duration_loss_weight: 耗时损失权重
        """
        super().__init__()
        self.save_hyperparameters()

        # 模型配置
        config = ModelConfig(
            d_model=d_model,
            n_heads=n_heads,
            depth=depth,
            d_ff=d_ff,
            dropout=dropout,
            num_numerical_features=num_numerical_features,
            categorical_vocab_sizes=categorical_vocab_sizes,
            categorical_embed_dim=categorical_embed_dim,
            num_rating_classes=num_rating_classes,
            card_head_ratio=card_head_ratio,
            deck_head_ratio=deck_head_ratio,
        )

        # 模型
        self.model = SPARKModel(config)

        # 损失函数
        self.loss_fn = CombinedLoss(
            rating_weight=rating_loss_weight,
            duration_weight=duration_loss_weight,
        )

        # 使用 MetricCollection 统一管理指标
        num_thresholds = num_rating_classes - 1
        self._init_metrics(num_thresholds)

    def _init_metrics(self, num_thresholds: int) -> None:
        """初始化评估指标。

        Args:
            num_thresholds: 有序分类的阈值数量
        """
        # 评分相关指标
        rating_metrics = MetricCollection(
            {
                "acc": MultilabelAccuracy(num_labels=num_thresholds),
                "auroc": MultilabelAUROC(num_labels=num_thresholds),
                "ordinal_acc": OrdinalAccuracy(),
            }
        )

        # 耗时相关指标，使用 MeanSquaredError(squared=False) 作为 RMSE
        duration_metrics = MetricCollection(
            {
                "mae": MeanAbsoluteError(),
                "rmse": MeanSquaredError(squared=False),
            }
        )

        # 为不同阶段克隆指标
        self.train_rating_metrics = rating_metrics.clone(prefix="train/rating_")
        self.val_rating_metrics = rating_metrics.clone(prefix="val/rating_")
        self.test_rating_metrics = rating_metrics.clone(prefix="test/rating_")

        self.train_duration_metrics = duration_metrics.clone(prefix="train/duration_")
        self.val_duration_metrics = duration_metrics.clone(prefix="val/duration_")
        self.test_duration_metrics = duration_metrics.clone(prefix="test/duration_")

        # 回忆（正确/错误）相关指标
        recall_metrics = MetricCollection(
            {
                "recall_auc": BinaryAUROC(),
            }
        )
        self.val_recall_metrics = recall_metrics.clone(prefix="val/")
        self.test_recall_metrics = recall_metrics.clone(prefix="test/")

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """前向传播。

        Args:
            batch: 数据批次

        Returns:
            模型输出
        """
        return self.model(
            numerical_features=batch["numerical_features"],
            categorical_features=batch["categorical_features"],
            time_stamps=batch["time_stamps"],
            causal_mask=batch["causal_mask"],
            card_mask=batch["card_mask"],
            deck_mask=batch["deck_mask"],
            time_diff=batch["time_diff"],
            padding_mask=batch["padding_mask"],
        )

    def _compute_step(
        self,
        batch: dict[str, Tensor],
        outputs: dict[str, Tensor],
        stage: str,
    ) -> dict[str, Tensor]:
        """计算损失和更新指标。

        Args:
            batch: 数据批次
            outputs: 模型输出
            stage: 阶段名称 ('train', 'val', 'test')

        Returns:
            损失字典
        """
        mask = batch["padding_mask"]

        # 计算损失
        losses = self.loss_fn(
            rating_probs=outputs["rating_probs"],
            ordinal_targets=batch["ordinal_targets"],
            duration_pred=outputs["duration_pred"],
            duration_target=batch["duration_targets"],
            mask=mask,
        )

        batch_size = batch["numerical_features"].size(0)

        # 记录损失
        self.log(
            f"{stage}/loss",
            losses["total"],
            prog_bar=True,
            sync_dist=True,
            batch_size=batch_size,
        )
        self.log(
            f"{stage}/rating_loss",
            losses["rating"],
            sync_dist=True,
            batch_size=batch_size,
        )
        self.log(
            f"{stage}/duration_loss",
            losses["duration"],
            sync_dist=True,
            batch_size=batch_size,
        )

        # 展平张量用于指标计算
        rating_probs_flat = outputs["rating_probs"][mask]
        ordinal_targets_flat = batch["ordinal_targets"][mask]
        duration_pred_flat = outputs["duration_pred"][mask]
        duration_target_flat = batch["duration_targets"][mask]

        # 获取对应阶段的指标集合
        rating_metrics = getattr(self, f"{stage}_rating_metrics")
        duration_metrics = getattr(self, f"{stage}_duration_metrics")

        targets_for_metrics = ordinal_targets_flat.int()

        # 更新指标状态
        rating_metrics.update(rating_probs_flat, targets_for_metrics)
        duration_metrics.update(duration_pred_flat, duration_target_flat)

        # 记录指标
        for name, metric in rating_metrics.items():
            self.log(name, metric, on_step=False, on_epoch=True, batch_size=batch_size)

        for name, metric in duration_metrics.items():
            self.log(name, metric, on_step=False, on_epoch=True, batch_size=batch_size)

        # 计算并记录 recall 相关指标 (仅验证和测试)
        if stage in ["val", "test"]:
            prob_correct = rating_probs_flat[:, 0]
            target_correct = ordinal_targets_flat[:, 0]

            # BCE Loss
            recall_bce = F.binary_cross_entropy(prob_correct, target_correct.float())
            self.log(
                f"{stage}/recall_bce",
                recall_bce,
                sync_dist=True,
                batch_size=batch_size,
            )

            # AUC Metric
            recall_metrics = getattr(self, f"{stage}_recall_metrics")
            recall_metrics.update(prob_correct, target_correct.int())

            for name, metric in recall_metrics.items():
                self.log(
                    name, metric, on_step=False, on_epoch=True, batch_size=batch_size
                )

        return losses

    def training_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        """训练步骤。"""
        outputs = self(batch)
        losses = self._compute_step(batch, outputs, "train")
        return losses["total"]

    def validation_step(self, batch: dict[str, Tensor], batch_idx: int) -> None:
        """验证步骤。"""
        outputs = self(batch)
        self._compute_step(batch, outputs, "val")

    def test_step(self, batch: dict[str, Tensor], batch_idx: int) -> None:
        """测试步骤。"""
        outputs = self(batch)
        self._compute_step(batch, outputs, "test")

    def predict_step(
        self, batch: dict[str, Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> dict[str, Tensor]:
        """预测步骤。"""
        return self(batch)

    def configure_optimizers(self) -> dict[str, Any]:
        """配置优化器和学习率调度器。"""
        hparams = self.hparams
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"],
        )

        # 使用带预热的余弦退火调度器
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=hparams["warmup_steps"],
            T_mult=2,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
