import torch
import torch.nn as nn
from typing import Optional
from transformers import PreTrainedModel

from .chronos import ChronosModel, ChronosConfig


class ChronosModelForAnomalyDetection(ChronosModel):
    def __init__(
        self,
        config: ChronosConfig,
        model: PreTrainedModel,
    ):
        super().__init__(config=config, model=model)

        hidden_size = (
            getattr(model.config, "hidden_size", None)
            or getattr(model.config, "d_model")
        )

        self.dropout = nn.Dropout(0.1)

        # token-level classifier
        self.classifier = nn.Linear(hidden_size, 1)

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

        hidden = outputs.hidden_states[-1]      # (B,T,H)

        logits = self.classifier(
            self.dropout(hidden)
        ).squeeze(-1)                           # (B,T)

        loss = None

        if labels is not None:
            labels = labels.float()

            loss_fct = nn.BCEWithLogitsLoss(
                reduction="none"
            )

            loss_per_token = loss_fct(
                logits,
                labels,
            )

            loss = (
                loss_per_token * attention_mask
            ).sum() / attention_mask.sum()

        return {
            "loss": loss,
            "logits": logits,
        }