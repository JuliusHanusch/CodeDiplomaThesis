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
        dropout_head: float = 0.1,
        loss_type: str = "mse",
    ):
        super().__init__(config=config, model=model)

        self.pooling = pooling
        self.loss_type = loss_type

        hidden_size = getattr(model.config, "hidden_size", None) \
            or getattr(model.config, "d_model")

        self.regressor = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(dropout_head)

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
        
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )

        hidden_states = outputs.hidden_states[-1]  # (B, T, H)

        if self.pooling == "mean":
            pooled = (
                hidden_states * attention_mask.unsqueeze(-1)
            ).sum(dim=1) / attention_mask.sum(dim=1, keepdim=True).clamp(min=1e-8)


        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")

        pooled = self.dropout(pooled)
        logits = self.regressor(pooled).squeeze(-1)  # (B,)

        loss = None
        if labels is not None:
            labels = labels.float()
            loss = self.criterion(logits, labels)


        return {
            "loss": loss,
            "logits": logits,
        }



class ChronosModelForTSERMultivariate(ChronosModel):
    def __init__(
        self,
        config: ChronosConfig,
        model: PreTrainedModel,
        n_channels: int,
        pooling: str = "mean",
        dropout_head: float = 0.1,
        loss_type: str = "mse",
    ):
        super().__init__(
            config=config,
            model=model
        )

        self.pooling = pooling
        self.loss_type = loss_type
        self.dropout_head = dropout_head
        self.n_channels = n_channels


        hidden_size = getattr(
            model.config,
            "hidden_size",
            None
        ) or getattr(
            model.config,
            "d_model"
        )

        self.hidden_size = hidden_size

        self.fusion = nn.Sequential(
            nn.Linear(
                n_channels * hidden_size,
                hidden_size
            ),
            nn.ReLU(),
            nn.Dropout(dropout_head),
            nn.Linear(
                hidden_size,
                1
            ),
        )

        if loss_type == "mse":
            self.criterion = nn.MSELoss()
        elif loss_type == "mae":
            self.criterion = nn.L1Loss()
        else:
            raise ValueError(
                f"Unknown loss_type: {loss_type}"
            )

    def encode_covariate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )

        hidden_states = outputs.hidden_states[-1]

        if self.pooling == "mean":
            pooled = (
                hidden_states
                * attention_mask.unsqueeze(-1)
            ).sum(dim=1) / attention_mask.sum(
                dim=1,
                keepdim=True
            ).clamp(min=1e-8)

        else:
            raise ValueError(
                f"Unknown pooling: {self.pooling}"
            )

        return pooled

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ):

        batch_size, n_channels, seq_len = input_ids.shape

        if n_channels != self.n_channels:
            raise ValueError(
                f"Expected {self.n_channels} channels, "
                f"but received {n_channels}"
            )

        representations = []

        for channel in range(n_channels):

            representation = self.encode_covariate(
                input_ids[:, channel, :],
                attention_mask[:, channel, :],
            )

            representations.append(
                representation
            )

        combined = torch.cat(
            representations,
            dim=-1
        )

        logits = self.fusion(
            combined
        ).squeeze(-1)

        loss = None

        if labels is not None:
            labels = labels.float()

            loss = self.criterion(
                logits,
                labels
            )

        return {
            "loss": loss,
            "logits": logits,
        }