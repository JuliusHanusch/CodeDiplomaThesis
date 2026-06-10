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

        hidden = outputs.hidden_states[-1]  # (B, T, H)
        hidden = self.dropout(hidden)

        logits = self.classifier(hidden).squeeze(-1)  # (B, T)
        probs = torch.sigmoid(logits)

        loss = None

        if labels is not None:

            labels = labels.float()

            # --------------------------------------------------
            # 🔥 DEBUG BLOCK (THIS IS WHAT YOU ASKED FOR)
            # --------------------------------------------------
            with torch.no_grad():

                mask = attention_mask.bool()

                pos_mask = (labels == 1) & mask
                neg_mask = (labels == 0) & mask

                print("\n[DEBUG FORWARD] ---------------------")
                print("pos tokens:", pos_mask.sum().item())
                print("neg tokens:", neg_mask.sum().item())
                print("mask tokens:", mask.sum().item())

                if pos_mask.sum() > 0:
                    print("pos logits mean:", logits[pos_mask].mean().item())
                    print("pos probs mean:", probs[pos_mask].mean().item())

                if neg_mask.sum() > 0:
                    print("neg logits mean:", logits[neg_mask].mean().item())
                    print("neg probs mean:", probs[neg_mask].mean().item())

                print("global logits mean:", logits.mean().item())
                print("label ratio:", labels[mask].mean().item())

            # --------------------------------------------------
            # LOSS (UNCHANGED)
            # --------------------------------------------------

            pos = labels.sum()
            neg = labels.numel() - pos
            pos_weight = neg / (pos + 1e-8)
            pos_weight = torch.clamp(pos_weight, 1.0, 50.0)

            pos_weight_tensor = torch.tensor(
                [pos_weight],
                device=logits.device,
                dtype=logits.dtype,
            )

            loss_fct = nn.BCEWithLogitsLoss(
                pos_weight=pos_weight_tensor,
                reduction="none"
            )

            loss_per_token = loss_fct(logits, labels)

            loss = (loss_per_token * attention_mask).sum() / (
                attention_mask.sum() + 1e-8
            )

        return {
            "loss": loss,
            "logits": logits,
        }