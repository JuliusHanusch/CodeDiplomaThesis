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
        # print(type(outputs))
        # print(outputs.keys())
        # print(outputs["logits"].mean(), outputs["logits"].std())

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
                print("\n[ANOMALY SIGNAL DEBUG] ---------------------")
                print("logits mean:", logits.mean().item())
                print("logits std :", logits.std().item())
                print("logits min :", logits.min().item())
                print("logits max :", logits.max().item())

                print("probs mean :", probs.mean().item())
                print("probs std  :", probs.std().item())
                print("probs min  :", probs.min().item())
                print("probs max  :", probs.max().item())

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

            # --------------------------------------------------
            # LOSS (UNCHANGED)
            # --------------------------------------------------

            pos = labels.sum()
            neg = labels.numel() - pos
            pos_weight = neg / (pos + 1e-8)
            pos_weight = torch.clamp(pos_weight, 1.0, 5.0)

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


# import torch
# import torch.nn as nn
# from typing import Optional
# from transformers import PreTrainedModel

# from .chronos import ChronosModel, ChronosConfig


# class ChronosModelForSpanAnomalyDetection(ChronosModel):
#     def __init__(
#         self,
#         config: ChronosConfig,
#         model: PreTrainedModel,
#     ):
#         super().__init__(config=config, model=model)

#         hidden_size = (
#             getattr(model.config, "hidden_size", None)
#             or getattr(model.config, "d_model")
#         )

#         self.dropout = nn.Dropout(0.1)

#         # sequence-level anomaly prediction
#         self.anomaly_head = nn.Linear(hidden_size, 1)

#         # token-level start/end prediction
#         self.start_head = nn.Linear(hidden_size, 1)
#         self.end_head = nn.Linear(hidden_size, 1)

#         self.bce_loss = nn.BCEWithLogitsLoss()
#         self.ce_loss = nn.CrossEntropyLoss(ignore_index=-100)

#     def forward(
#         self,
#         input_ids: torch.Tensor,
#         attention_mask: torch.Tensor,
#         has_anomaly: Optional[torch.Tensor] = None,
#         start_positions: Optional[torch.Tensor] = None,
#         end_positions: Optional[torch.Tensor] = None,
#     ):

#         outputs = self.model(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#             output_hidden_states=True,
#             return_dict=True,
#         )

#         hidden = outputs.hidden_states[-1]  # (B, T, H)
#         hidden = self.dropout(hidden)

#         # --------------------------------------------------
#         # Sequence representation
#         # --------------------------------------------------

#         mask = attention_mask.unsqueeze(-1)

#         pooled = (hidden * mask).sum(dim=1) / (
#             mask.sum(dim=1).clamp(min=1)
#         )

#         # --------------------------------------------------
#         # Heads
#         # --------------------------------------------------

#         anomaly_logit = self.anomaly_head(pooled).squeeze(-1)  # (B)

#         start_logits = self.start_head(hidden).squeeze(-1)  # (B,T)
#         end_logits = self.end_head(hidden).squeeze(-1)      # (B,T)

#         # prevent selecting padding tokens
#         start_logits = start_logits.masked_fill(
#             attention_mask == 0,
#             -1e9,
#         )

#         end_logits = end_logits.masked_fill(
#             attention_mask == 0,
#             -1e9,
#         )

#         loss = None

#         if (
#             has_anomaly is not None
#             and start_positions is not None
#             and end_positions is not None
#         ):

#             has_anomaly = has_anomaly.float()

#             # --------------------------------------------------
#             # Presence loss
#             # --------------------------------------------------

#             anomaly_loss = self.bce_loss(
#                 anomaly_logit,
#                 has_anomaly,
#             )

#             # --------------------------------------------------
#             # Start / End losses
#             # --------------------------------------------------

#             start_loss = self.ce_loss(
#                 start_logits,
#                 start_positions,
#             )

#             end_loss = self.ce_loss(
#                 end_logits,
#                 end_positions,
#             )

#             loss = anomaly_loss + start_loss + end_loss

#             # --------------------------------------------------
#             # Debug
#             # --------------------------------------------------

#             with torch.no_grad():

#                 anomaly_prob = torch.sigmoid(anomaly_logit)

#                 pred_start = start_logits.argmax(dim=-1)
#                 pred_end = end_logits.argmax(dim=-1)

#                 print("\n[SPAN DEBUG] ---------------------")
#                 print(
#                     "has_anomaly mean:",
#                     has_anomaly.mean().item(),
#                 )
#                 print(
#                     "pred anomaly prob mean:",
#                     anomaly_prob.mean().item(),
#                 )

#                 valid = has_anomaly.bool()

#                 if valid.any():

#                     print(
#                         "true start:",
#                         start_positions[valid][:5].tolist(),
#                     )

#                     print(
#                         "pred start:",
#                         pred_start[valid][:5].tolist(),
#                     )

#                     print(
#                         "true end:",
#                         end_positions[valid][:5].tolist(),
#                     )

#                     print(
#                         "pred end:",
#                         pred_end[valid][:5].tolist(),
#                     )

#         return {
#             "loss": loss,
#             "anomaly_logit": anomaly_logit,
#             "start_logits": start_logits,
#             "end_logits": end_logits,
#         }