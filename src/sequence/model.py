from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm


class BaseSequenceRetriever(nn.Module):
    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int,
        pooling_method: str = "mean",
        dropout: float = 0.2,
        mask_pooling: bool = True,
        item_embedding: nn.Embedding = None,
    ):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.pooling_method = pooling_method.lower()
        self.mask_pooling = mask_pooling

        if item_embedding is None:
            self.item_embedding = nn.Embedding(
                num_items + 1,  # extra index for unknown/padding
                embedding_dim,
                padding_idx=num_items,
            )
        else:
            self.item_embedding = item_embedding

        self.user_embedding = nn.Embedding(num_users, embedding_dim)

        if self.pooling_method == "gru":
            self.gru = nn.GRU(embedding_dim, embedding_dim, batch_first=True)
        elif self.pooling_method == "mean":
            self.gru = None
        else:
            raise ValueError("Invalid pooling_method. Choose 'gru' or 'mean'.")

        self.dropout = nn.Dropout(p=dropout)

    def pool_sequence(
        self, seq_embeds: torch.Tensor, mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Pools a sequence of embeddings.
        If self.mask_pooling is True and a mask is provided, applies masked pooling.
        Otherwise, performs standard pooling over all tokens.

        Args:
            seq_embeds: Tensor of shape (batch_size, seq_length, embedding_dim)
            mask: Boolean tensor of shape (batch_size, seq_length) where True indicates a valid token.
        Returns:
            A tensor of shape (batch_size, embedding_dim) representing the pooled sequence.
        """
        if self.mask_pooling and mask is not None:
            if self.pooling_method == "gru":
                # Compute sequence lengths from the mask.
                lengths = mask.sum(dim=1).clamp(min=1)
                packed_seq = nn.utils.rnn.pack_padded_sequence(
                    seq_embeds, lengths.cpu(), batch_first=True, enforce_sorted=False
                )
                _, hidden_state = self.gru(packed_seq)
                return hidden_state.squeeze(0)
            elif self.pooling_method == "mean":
                # Mask the embeddings and compute a weighted average.
                mask_float = mask.unsqueeze(-1).float()
                sum_embeds = (seq_embeds * mask_float).sum(dim=1)
                count = mask_float.sum(dim=1).clamp(min=1)
                return sum_embeds / count
            else:
                raise ValueError("Invalid pooling_method. Choose 'gru' or 'mean'.")
        else:
            # Unmasked pooling (ignoring the mask).
            if self.pooling_method == "gru":
                output, hidden_state = self.gru(seq_embeds)
                return hidden_state.squeeze(0)
            elif self.pooling_method == "mean":
                return torch.mean(seq_embeds, dim=1)
            else:
                raise ValueError("Invalid pooling_method. Choose 'gru' or 'mean'.")

    def replace_neg_one_with_padding(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Replaces all -1 values in the tensor with the padding index.
        """
        padding_idx_tensor = torch.tensor(
            self.item_embedding.padding_idx, device=tensor.device
        )
        return torch.where(tensor == -1, padding_idx_tensor, tensor)


class SequenceRetrieverFactory:
    # Registry to hold model implementations.
    MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {}

    @staticmethod
    def register_retriever(name: str, params: list = None, required: list = None):
        """
        Static method decorator to register a new retriever model with the factory.

        Args:
            name (str): Unique name for the model type.
            params (list, optional): List of allowed parameter names.
            required (list, optional): List of required parameter names.
        """

        def decorator(cls):
            SequenceRetrieverFactory.MODEL_REGISTRY[name] = {
                "class": cls,
                "params": params or [],
                "required": required or [],
            }
            return cls

        return decorator

    @staticmethod
    def create_retriever(model_type: str, **kwargs):
        config = SequenceRetrieverFactory.MODEL_REGISTRY.get(model_type)
        if not config:
            raise ValueError(f"Unknown model type: {model_type}")

        # Validate required parameters.
        missing = [p for p in config["required"] if p not in kwargs]
        if missing:
            raise ValueError(f"Missing required parameters for {model_type}: {missing}")

        # Filter kwargs to only include parameters relevant for the model.
        allowed_keys = config["params"]
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in allowed_keys}
        return config["class"](**filtered_kwargs)


@SequenceRetrieverFactory.register_retriever(
    "TwoTowerSequenceRetriever",
    params=[
        "num_users",
        "num_items",
        "embedding_dim",
        "pooling_method",
        "dropout",
        "mask_pooling",
    ],
    required=["num_users", "num_items", "embedding_dim"],
)
class TwoTowerSequenceRetriever(BaseSequenceRetriever):
    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int,
        pooling_method: str = "mean",
        dropout: float = 0.2,
        mask_pooling: bool = True,
    ):
        super().__init__(
            num_users, num_items, embedding_dim, pooling_method, dropout, mask_pooling
        )
        # Query tower: combines user embedding with sequence representation.
        self.query_fc = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.Dropout(dropout),
        )
        # Candidate tower: transforms raw item embeddings.
        self.candidate_fc = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        user_ids: torch.Tensor,
        item_seq: torch.Tensor,
        candidate_items: torch.Tensor,
    ) -> torch.Tensor:
        query_embedding = self.get_query_embedding(user_ids, item_seq)
        candidate_embedding = self.get_candidate_embedding(candidate_items)
        # Normalize embeddings.
        query_embedding = F.normalize(query_embedding, p=2, dim=1)
        if candidate_embedding.dim() == 2:
            candidate_embedding = F.normalize(candidate_embedding, p=2, dim=1)
            cos_sim = torch.sum(query_embedding * candidate_embedding, dim=1)
        elif candidate_embedding.dim() == 3:
            candidate_embedding = F.normalize(candidate_embedding, p=2, dim=2)
            cos_sim = torch.sum(
                query_embedding.unsqueeze(1) * candidate_embedding, dim=2
            )
        else:
            raise ValueError("Candidate embedding must be either 2D or 3D.")
        # Scale cosine similarity from [-1, 1] to [0, 1].
        return (cos_sim + 1) / 2

    def get_query_embedding(
        self, user_ids: torch.Tensor, item_seq: torch.Tensor
    ) -> torch.Tensor:
        # Replace -1 values with the designated padding index.
        item_seq = self.replace_neg_one_with_padding(item_seq)
        # Create a mask: True for valid tokens, False for padding.
        mask = item_seq != self.item_embedding.padding_idx
        seq_embeds = self.item_embedding(item_seq)
        # Pool the sequence; the method will decide whether to use the mask based on self.mask_pooling.
        seq_rep = self.pool_sequence(seq_embeds, mask)
        user_embed = self.user_embedding(user_ids)
        combined = torch.cat([user_embed, seq_rep], dim=1)
        return self.query_fc(combined)

    def get_candidate_embedding(self, candidate_items: torch.Tensor) -> torch.Tensor:
        candidate_embedding = self.item_embedding(candidate_items)
        if candidate_embedding.dim() == 2:
            candidate_embedding = self.candidate_fc(candidate_embedding)
        elif candidate_embedding.dim() == 3:
            batch_size, num_candidates, _ = candidate_embedding.shape
            candidate_embedding = candidate_embedding.view(-1, self.embedding_dim)
            candidate_embedding = self.candidate_fc(candidate_embedding)
            candidate_embedding = candidate_embedding.view(
                batch_size, num_candidates, self.embedding_dim
            )
        else:
            raise ValueError("Candidate embedding must be either 2D or 3D.")
        return candidate_embedding

    def predict(
        self, user: torch.Tensor, item_sequence: torch.Tensor, target_item: torch.Tensor
    ) -> torch.Tensor:
        return self.forward(user, item_sequence, target_item)

    def recommend(
        self,
        users: torch.Tensor,
        item_sequences: torch.Tensor,
        k: int,
        batch_size: int = 128,
    ) -> Dict[str, Any]:
        self.eval()
        device = users.device
        all_items = torch.arange(self.item_embedding.num_embeddings, device=device)
        # Precompute candidate embeddings.
        candidate_embeddings = self.get_candidate_embedding(all_items)

        user_indices = []
        recommendations = []
        scores = []

        with torch.no_grad():
            total_users = users.size(0)
            for i in tqdm(
                range(0, total_users, batch_size), desc="Generating recommendations"
            ):
                user_batch = users[i : i + batch_size]
                item_seq_batch = item_sequences[i : i + batch_size]
                query_embedding = self.get_query_embedding(user_batch, item_seq_batch)
                batch_scores = torch.matmul(query_embedding, candidate_embeddings.t())

                topk_scores, topk_indices = torch.topk(batch_scores, k, dim=1)
                for j in range(user_batch.size(0)):
                    user_indices.extend([int(user_batch[j].item())] * k)
                    recommendations.extend(topk_indices[j].cpu().tolist())
                    scores.extend(topk_scores[j].cpu().tolist())

        return {
            "user_indice": user_indices,
            "recommendation": recommendations,
            "score": scores,
        }
