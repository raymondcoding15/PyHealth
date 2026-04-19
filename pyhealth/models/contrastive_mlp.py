from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from pyhealth.datasets import SampleDataset
from .base_model import BaseModel


class ContrastiveMLP(BaseModel):
    """MLP classifier with optional batch contrastive regularization.

    This model is intended for binary/multiclass tasks where structured features
    are represented as tensors. It combines a supervised task loss with an
    optional contrastive term over sample embeddings.

    Args:
        dataset: SampleDataset created by ``dataset.set_task`` or
            ``create_sample_dataset``.
        hidden_dim: Hidden representation dimension per feature branch.
        dropout: Dropout probability before classification.
        contrastive_weight: Weight for the contrastive regularizer. Set to 0.0 to
            disable.
        contrastive_margin: Margin used for negative pairs in contrastive loss.

    Examples:
        >>> from pyhealth.datasets import create_sample_dataset, get_dataloader
        >>> from pyhealth.models import ContrastiveMLP
        >>> samples = [
        ...     {"patient_id": "p1", "visit_id": "v1", "signal": [0.2, 0.3, 0.1], "label": 0},
        ...     {"patient_id": "p2", "visit_id": "v2", "signal": [0.9, 0.7, 0.8], "label": 1},
        ... ]
        >>> ds = create_sample_dataset(
        ...     samples=samples,
        ...     input_schema={"signal": "tensor"},
        ...     output_schema={"label": "binary"},
        ...     dataset_name="toy_contrastive",
        ... )
        >>> model = ContrastiveMLP(dataset=ds, hidden_dim=16)
        >>> batch = next(iter(get_dataloader(ds, batch_size=2, shuffle=False)))
        >>> out = model(**batch)
        >>> sorted(out.keys())
        ['contrastive_loss', 'embed', 'logit', 'loss', 'task_loss', 'y_prob', 'y_true']
    """

    def __init__(
        self,
        dataset: SampleDataset,
        hidden_dim: int = 64,
        dropout: float = 0.1,
        contrastive_weight: float = 0.1,
        contrastive_margin: float = 0.5,
    ) -> None:
        super().__init__(dataset)
        if len(self.label_keys) != 1:
            raise ValueError("ContrastiveMLP currently supports exactly one label key.")
        self.label_key = self.label_keys[0]
        self.hidden_dim = hidden_dim
        self.contrastive_weight = contrastive_weight
        self.contrastive_margin = contrastive_margin
        self.dropout = nn.Dropout(dropout)

        # LazyLinear avoids hardcoding feature dimensions and stays compatible
        # with arbitrary tensor widths from processors.
        self.feature_nets = nn.ModuleDict(
            {key: nn.Sequential(nn.LazyLinear(hidden_dim), nn.ReLU()) for key in self.feature_keys}
        )
        self.classifier = nn.LazyLinear(self.get_output_size())

    def _pool_feature(self, feature_key: str, feature: torch.Tensor | tuple[torch.Tensor, ...]) -> torch.Tensor:
        if isinstance(feature, torch.Tensor):
            feature = (feature,)
        schema = self.dataset.input_processors[feature_key].schema()
        value = feature[schema.index("value")] if "value" in schema else feature[0]
        value = value.to(self.device).float()
        # Support both [B, D] and [B, T, D] tensors.
        if value.dim() == 3:
            value = value.mean(dim=1)
        elif value.dim() != 2:
            raise ValueError(f"Unsupported feature tensor rank {value.dim()} for key {feature_key}.")
        return value

    def _contrastive_loss(self, embed: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        # Normalize and compute pairwise cosine similarity.
        z = F.normalize(embed, p=2, dim=-1)
        sim = z @ z.T
        y = y_true.view(-1)
        same = (y.unsqueeze(0) == y.unsqueeze(1)).float()
        eye = torch.eye(len(y), device=embed.device)
        same = same - eye * same
        diff = 1.0 - same - eye

        # Pull positives together, push negatives below margin.
        pos_loss = ((1.0 - sim) * same).sum() / (same.sum() + 1e-8)
        neg_loss = F.relu(sim - self.contrastive_margin) * diff
        neg_loss = neg_loss.sum() / (diff.sum() + 1e-8)
        return pos_loss + neg_loss

    def forward(self, **kwargs: torch.Tensor | tuple[torch.Tensor, ...]) -> Dict[str, torch.Tensor]:
        pooled: List[torch.Tensor] = []
        for feature_key in self.feature_keys:
            x = self._pool_feature(feature_key, kwargs[feature_key])
            pooled.append(self.feature_nets[feature_key](x))
        embed = torch.cat(pooled, dim=-1)
        embed = self.dropout(embed)

        logit = self.classifier(embed)
        y_true = kwargs[self.label_key].to(self.device)
        if self.mode == "binary":
            y_true = y_true.float()

        task_loss = self.get_loss_function()(logit, y_true)
        contrastive_loss = self._contrastive_loss(embed, y_true)
        loss = task_loss + self.contrastive_weight * contrastive_loss
        y_prob = self.prepare_y_prob(logit)
        return {
            "loss": loss,
            "task_loss": task_loss,
            "contrastive_loss": contrastive_loss,
            "y_prob": y_prob,
            "y_true": y_true,
            "logit": logit,
            "embed": embed,
        }

