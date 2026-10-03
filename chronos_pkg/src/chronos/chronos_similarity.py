import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from transformers import PreTrainedModel

from .chronos import ChronosModel, ChronosConfig


class ChronosModelForSimilarity(ChronosModel):
    """
    Chronos-BERT model for time series similarity learning.

    Produces embeddings instead of class predictions.

    Training objective:
        - same class -> embeddings close
        - different class -> embeddings far apart

    """

    def __init__(
        self,
        config: ChronosConfig,
        model: PreTrainedModel,
        embedding_dim: int = 128,
        dropout_head: float = 0.1,
        pooling: str = "mean",
        temperature: float = 0.1,
    ):
        super().__init__(config=config, model=model)

        self.pooling = pooling
        self.temperature = temperature

        hidden_size = getattr(model.config, "hidden_size", None) \
            or getattr(model.config, "d_model")

        self.dropout = nn.Dropout(dropout_head)

        # projection head
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout_head),
            nn.Linear(hidden_size, embedding_dim),
        )


    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ):
        """
        Convert time series into embeddings.
        """

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )

        hidden_states = outputs.hidden_states[-1]
        # (batch, tokens, hidden)


        if self.pooling == "mean":

            pooled = (
                hidden_states *
                attention_mask.unsqueeze(-1)
            ).sum(1) / attention_mask.sum(
                1,
                keepdim=True
            )


        elif self.pooling == "cls":

            pooled = hidden_states[:, 0]


        else:
            raise ValueError(
                f"Unknown pooling {self.pooling}"
            )


        embedding = self.projection(
            self.dropout(pooled)
        )


        # important for cosine similarity
        embedding = F.normalize(
            embedding,
            dim=-1
        )

        return embedding



    def forward(
        self,
        input_ids_1: torch.Tensor,
        attention_mask_1: torch.Tensor,
        input_ids_2: torch.Tensor,
        attention_mask_2: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ):

        """
        labels:
            1  -> similar
            0  -> different
        """

        z1 = self.encode(
            input_ids_1,
            attention_mask_1
        )

        z2 = self.encode(
            input_ids_2,
            attention_mask_2
        )


        loss = None

        if labels is not None:

            # cosine similarity
            similarity = F.cosine_similarity(
                z1,
                z2
            )

            # convert:
            # similar -> high cosine
            # different -> low cosine

            target = labels.float()

            loss = F.binary_cross_entropy_with_logits(
                similarity / self.temperature,
                target
            )


        return {
            "loss": loss,
            "embeddings_1": z1,
            "embeddings_2": z2,
        }