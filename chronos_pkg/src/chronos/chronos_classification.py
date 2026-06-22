import torch
import torch.nn as nn
from typing import Optional
from transformers import PreTrainedModel

from .chronos import ChronosModel, ChronosConfig


class ChronosModelForClassification(ChronosModel):
    def __init__(
        self,
        config: ChronosConfig,
        model: PreTrainedModel,
        num_labels: int = 2,
        pooling: str = "mean",
    ):
        super().__init__(config=config, model=model)

        self.num_labels = num_labels
        self.pooling = pooling

        hidden_size = getattr(model.config, "hidden_size", None) \
            or getattr(model.config, "d_model")

        self.classifier = nn.Linear(hidden_size, num_labels)
        self.dropout = nn.Dropout(0.1)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ):
        print("\n================ FORWARD PASS ================")
        print("input_ids:", input_ids.shape, input_ids.dtype)
        print("attention_mask:", attention_mask.shape, attention_mask.dtype)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )

        print("model type:", type(self.model))
        print("has MLM head:", hasattr(self.model, "cls"))

        hidden_states = outputs.hidden_states[-1]  # (B, T, H)

        print("hidden_states:", hidden_states.shape)

        if self.pooling == "mean":
            pooled = (hidden_states * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(1, keepdim=True)
        #if cls available
        elif self.pooling == "cls":
            pooled = hidden_states[:, 0]
        else:
            raise ValueError("Unknown pooling")
        
        print("pooled:", pooled.shape)

        logits = self.classifier(self.dropout(pooled))

        print("logits:", logits.shape)

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)

        print("loss:", loss)
        print("===========================================\n")

        return {"loss": loss, "logits": logits}