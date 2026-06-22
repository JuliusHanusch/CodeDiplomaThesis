import torch
import torch.nn as nn
from typing import Optional
from transformers import PreTrainedModel

from .chronos import ChronosModel, ChronosConfig


class ChronosModelForTSER(ChronosModel):
    def __init__(
        self,
        config: ChronosConfig,
        model: PreTrainedModel,
        pooling: str = "mean",
        loss_type: str = "mae",
    ):
        super().__init__(config=config, model=model)

        self.pooling = pooling
        self.loss_type = loss_type

        hidden_size = getattr(model.config, "hidden_size", None) \
            or getattr(model.config, "d_model")

        self.regressor = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(0.1)

        if loss_type == "mse":
            self.criterion = nn.MSELoss()
        elif loss_type == "mae":
            self.criterion = nn.L1Loss()
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ):
        
        print("\n================ FORWARD PASS ================")
        print("input_ids:", input_ids.shape)
        print("attention_mask:", attention_mask.shape)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )

        hidden_states = outputs.hidden_states[-1]  # (B, T, H)
        print("hidden_states:", hidden_states.shape)

        if self.pooling == "mean":
            pooled = (
                hidden_states * attention_mask.unsqueeze(-1)
            ).sum(dim=1) / attention_mask.sum(dim=1, keepdim=True).clamp(min=1e-8)

            print("pooled:", pooled.shape)
            print("pooled sample:", pooled[0, :5] if pooled.ndim == 2 else pooled[:5])


        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")

        pooled = self.dropout(pooled)
        print("after dropout:", pooled.shape)
        logits = self.regressor(pooled).squeeze(-1)  # (B,)
        print("logits BEFORE squeeze:", self.regressor(pooled).shape)
        print("logits AFTER squeeze:", logits.shape)

        if logits.ndim > 1:
            print("⚠️ WARNING: logits still 2D/3D -> unexpected behavior")

        loss = None
        if labels is not None:
            labels = labels.float()
            loss = self.criterion(logits, labels)

        print("================ END FORWARD ================\n")

        return {
            "loss": loss,
            "logits": logits,
        }