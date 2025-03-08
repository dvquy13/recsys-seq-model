from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm


class SequenceRatingPrediction(nn.Module):
    """
    A PyTorch neural network model for predicting user-item interaction ratings based on sequences of previous items
    and a target item. This model uses user and item embeddings, and performs rating predictions using fully connected layers.

    Args:
        num_users (int): The number of unique users.
        num_items (int): The number of unique items.
        embedding_dim (int): The size of the embedding dimension for both user and item embeddings.
        pooling_method (str, optional): The pooling method to use for sequence encoding. Options are 'gru' or 'mean'.
                                        Defaults to 'gru'.
        item_embedding (torch.nn.Embedding, optional): Pretrained item embeddings. Defaults to None.
        dropout (float, optional): The dropout probability applied to the fully connected layers. Defaults to 0.2.

    Attributes:
        num_items (int): Number of unique items.
        num_users (int): Number of unique users.
        item_embedding (nn.Embedding): Embedding layer for items, including a padding index for unknown items.
        user_embedding (nn.Embedding): Embedding layer for users.
        gru (nn.GRU, optional): GRU layer to process item sequences (if pooling_method is 'gru').
        pooling_method (str): The selected pooling method.
        fc_rating (nn.Sequential): Fully connected layers for predicting the rating from concatenated embeddings.
        relu (nn.ReLU): ReLU activation function.
        dropout (nn.Dropout): Dropout layer applied after activation.
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int,
        pooling_method: str = "mean",
        item_embedding: nn.Embedding = None,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.num_items = num_items
        self.num_users = num_users
        self.pooling_method = pooling_method.lower()

        self.item_embedding = item_embedding
        if self.item_embedding is None:
            # Item embedding (Add 1 to num_items for the unknown item (-1 padding))
            self.item_embedding = nn.Embedding(
                num_items + 1,  # One additional index for unknown/padding item
                embedding_dim,
                padding_idx=num_items,  # The additional index for the unknown item
            )

        # User embedding
        self.user_embedding = nn.Embedding(num_users, embedding_dim)

        if self.pooling_method == "gru":
            # GRU layer to process item sequences
            self.gru = nn.GRU(
                input_size=embedding_dim, hidden_size=embedding_dim, batch_first=True
            )
        elif self.pooling_method == "mean":
            # No additional layer is needed for mean pooling.
            self.gru = None
        else:
            raise ValueError("Invalid pooling_method. Choose 'gru' or 'mean'.")

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)

        # Fully connected layer to map concatenated embeddings to rating prediction
        self.fc_rating = nn.Sequential(
            nn.Linear(embedding_dim * 3, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            self.relu,
            self.dropout,
            nn.Linear(embedding_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, user_ids, input_seq, target_item):
        """
        Forward pass to predict the rating.

        Args:
            user_ids (torch.Tensor): Batch of user IDs.
            input_seq (torch.Tensor): Batch of item sequences.
            target_item (torch.Tensor): Batch of target items to predict the rating for.

        Returns:
            torch.Tensor: Predicted rating for each user-item pair.
        """
        # Replace -1 in input_seq and target_item with num_items (padding_idx)
        padding_idx_tensor = torch.tensor(
            self.item_embedding.padding_idx, device=input_seq.device
        )
        input_seq = torch.where(input_seq == -1, padding_idx_tensor, input_seq)
        target_item = torch.where(target_item == -1, padding_idx_tensor, target_item)

        # Embed input sequence
        embedded_seq = self.item_embedding(
            input_seq
        )  # Shape: [batch_size, seq_len, embedding_dim]

        if self.pooling_method == "gru":
            # Process with GRU and use the final hidden state as the sequence representation
            _, hidden_state = self.gru(
                embedded_seq
            )  # hidden_state: [1, batch_size, embedding_dim]
            pooled_seq = hidden_state.squeeze(0)  # [batch_size, embedding_dim]
        elif self.pooling_method == "mean":
            # Mean pooling over the sequence dimension
            pooled_seq = torch.mean(embedded_seq, dim=1)  # [batch_size, embedding_dim]
        else:
            raise ValueError("Invalid pooling_method. Choose 'gru' or 'mean'.")

        # Embed the target item
        embedded_target = self.item_embedding(target_item)

        # Embed the user IDs
        user_embeddings = self.user_embedding(user_ids)

        # Concatenate the pooled sequence with the target item and user embeddings
        combined_embedding = torch.cat(
            (pooled_seq, embedded_target, user_embeddings), dim=1
        )

        # Predict the rating
        output_ratings = self.fc_rating(combined_embedding)

        return output_ratings

    def predict(self, user, item_sequence, target_item):
        """
        Predict the rating for a specific user and item sequence.

        Args:
            user (torch.Tensor): User ID.
            item_sequence (torch.Tensor): Sequence of previously interacted items.
            target_item (torch.Tensor): The target item to predict the rating for.

        Returns:
            torch.Tensor: Predicted rating after applying Sigmoid activation.
        """
        return self.forward(user, item_sequence, target_item)

    def recommend(
        self,
        users: torch.Tensor,
        item_sequences: torch.Tensor,
        k: int,
        batch_size: int = 128,
    ) -> Dict[str, Any]:
        """
        Generate top-k recommendations for a batch of users based on their item sequences.

        Args:
            users (torch.Tensor): Tensor containing user IDs.
            item_sequences (torch.Tensor): Tensor containing sequences of previously interacted items.
            k (int): Number of recommendations to generate for each user.
            batch_size (int, optional): Batch size for processing users. Defaults to 128.

        Returns:
            Dict[str, Any]: Dictionary containing recommended items and scores:
                'user_indice': List of user indices.
                'recommendation': List of recommended item indices.
                'score': List of predicted interaction scores.
        """
        self.eval()
        all_items = torch.arange(
            self.item_embedding.num_embeddings, device=users.device
        )

        user_indices = []
        recommendations = []
        scores = []

        with torch.no_grad():
            total_users = users.size(0)
            for i in tqdm(
                range(0, total_users, batch_size), desc="Generating recommendations"
            ):
                user_batch = users[i : i + batch_size]
                item_sequence_batch = item_sequences[i : i + batch_size]

                # Expand user_batch to match all items
                user_batch_expanded = (
                    user_batch.unsqueeze(1).expand(-1, len(all_items)).reshape(-1)
                )
                items_batch = (
                    all_items.unsqueeze(0).expand(len(user_batch), -1).reshape(-1)
                )
                item_sequences_batch = item_sequence_batch.unsqueeze(1).repeat(
                    1, len(all_items), 1
                )
                item_sequences_batch = item_sequences_batch.view(
                    -1, item_sequence_batch.size(-1)
                )

                # Predict scores for the batch
                batch_scores = self.predict(
                    user_batch_expanded, item_sequences_batch, items_batch
                ).view(len(user_batch), -1)

                # Get top k items for each user in the batch
                topk_scores, topk_indices = torch.topk(batch_scores, k, dim=1)
                topk_items = all_items[topk_indices]

                # Collect recommendations
                user_indices.extend(user_batch.repeat_interleave(k).cpu().tolist())
                recommendations.extend(topk_items.cpu().flatten().tolist())
                scores.extend(topk_scores.cpu().flatten().tolist())

        return {
            "user_indice": user_indices,
            "recommendation": recommendations,
            "score": scores,
        }


class TwoTowerSequenceModel(nn.Module):
    """
    Two-tower model architecture that separates the user/sequence processing (query tower)
    from the candidate item embedding (candidate tower). The score between a query and a candidate
    is computed as the dot product between their respective embeddings.

    After training, candidate embeddings can be pre-computed and persisted so that at inference,
    the model only needs to compute the query embedding and then perform a nearest neighbor search.

    Args:
        num_users (int): Number of unique users.
        num_items (int): Number of unique items.
        embedding_dim (int): Embedding dimension for both user and item embeddings.
        pooling_method (str, optional): Method for pooling the sequence representation; either "gru" or "mean". Defaults to "mean".
        dropout (float, optional): Dropout probability. Defaults to 0.2.
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int,
        pooling_method: str = "mean",
        dropout: float = 0.2,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.pooling_method = pooling_method.lower()

        # Shared item embedding used by both towers (with an extra index for unknown/padding)
        self.item_embedding = nn.Embedding(
            num_items + 1,  # extra index for padding/unknown items
            embedding_dim,
            padding_idx=num_items,
        )
        # User embedding (used only in the query tower)
        self.user_embedding = nn.Embedding(num_users, embedding_dim)

        # For sequence encoding in the query tower
        if self.pooling_method == "gru":
            self.gru = nn.GRU(
                input_size=embedding_dim,
                hidden_size=embedding_dim,
                batch_first=True,
            )
        elif self.pooling_method == "mean":
            self.gru = None
        else:
            raise ValueError("Invalid pooling_method. Choose 'gru' or 'mean'.")

        # Query Tower: Combines the user embedding with a representation of the interaction sequence.
        # The input to this tower is the concatenation of the user embedding and the pooled sequence representation.
        self.query_fc = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Candidate Tower: Applies a transformation to the raw item embedding.
        self.candidate_fc = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        user_ids: torch.Tensor,
        item_seq: torch.Tensor,
        candidate_items: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass that computes cosine similarity between query and candidate embeddings.
        The cosine similarity is scaled to [0, 1] so it can be used with BCELoss.
        """
        query_embedding = self.get_query_embedding(user_ids, item_seq)
        candidate_embedding = self.get_candidate_embedding(candidate_items)

        # Normalize embeddings to unit vectors
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

        # Scale cosine similarity from [-1, 1] to [0, 1]
        scores = (cos_sim + 1) / 2
        return scores

    def get_query_embedding(
        self, user_ids: torch.Tensor, item_seq: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes the query embedding from the user ID and their interaction sequence.

        Args:
            user_ids (torch.Tensor): Tensor of shape [batch_size] with user IDs.
            item_seq (torch.Tensor): Tensor of shape [batch_size, seq_len] with item IDs in the interaction sequence.

        Returns:
            torch.Tensor: Query embedding of shape [batch_size, embedding_dim].
        """
        # Replace -1 values (if any) with the padding index
        padding_idx_tensor = torch.tensor(
            self.item_embedding.padding_idx, device=item_seq.device
        )
        item_seq = torch.where(item_seq == -1, padding_idx_tensor, item_seq)

        # Get item embeddings for the sequence: shape [batch_size, seq_len, embedding_dim]
        seq_embeds = self.item_embedding(item_seq)

        # Pool the sequence embeddings
        if self.pooling_method == "gru":
            # Use GRU's final hidden state as the sequence representation
            _, hidden_state = self.gru(
                seq_embeds
            )  # hidden_state: [1, batch_size, embedding_dim]
            seq_rep = hidden_state.squeeze(0)  # shape: [batch_size, embedding_dim]
        elif self.pooling_method == "mean":
            seq_rep = torch.mean(seq_embeds, dim=1)
        else:
            raise ValueError("Invalid pooling_method. Choose 'gru' or 'mean'.")

        # Get user embeddings: shape [batch_size, embedding_dim]
        user_embed = self.user_embedding(user_ids)

        # Concatenate the user embedding with the pooled sequence representation
        combined = torch.cat(
            [user_embed, seq_rep], dim=1
        )  # shape: [batch_size, embedding_dim * 2]
        query_embedding = self.query_fc(combined)  # shape: [batch_size, embedding_dim]

        return query_embedding

    def get_candidate_embedding(self, candidate_items: torch.Tensor) -> torch.Tensor:
        """
        Computes the candidate embedding for given candidate item IDs.
        This is useful for precomputing and persisting candidate embeddings.

        Args:
            candidate_items (torch.Tensor): Tensor of candidate item IDs. Can be of shape [num_candidates]
                                            or [batch_size, num_candidates].

        Returns:
            torch.Tensor: Candidate embedding(s) of shape [num_candidates, embedding_dim] or [batch_size, num_candidates, embedding_dim].
        """
        candidate_embedding = self.item_embedding(candidate_items)
        # Transform the raw candidate embedding through the candidate tower
        if candidate_embedding.dim() == 2:
            candidate_embedding = self.candidate_fc(candidate_embedding)
        elif candidate_embedding.dim() == 3:
            # For a 3D tensor, reshape to combine the first two dimensions, apply transformation, then reshape back.
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
        """
        Predict the similarity score for a given user, item sequence, and target item.
        This is analogous to the original predict method and computes the dot product similarity.

        Args:
            user (torch.Tensor): Tensor of user IDs.
            item_sequence (torch.Tensor): Tensor of shape [batch_size, seq_len] representing the item interaction sequence.
            target_item (torch.Tensor): Tensor of target item IDs (can be a single candidate per user).

        Returns:
            torch.Tensor: Predicted similarity score(s).
        """
        return (
            self.forward(user, item_sequence, target_item) * 2 - 1
        )  # Scale to [-1, 1] so that to avoid the min prediction value is 0.5 cause most of the time the forward output is 0

    def recommend(
        self,
        users: torch.Tensor,
        item_sequences: torch.Tensor,
        k: int,
        batch_size: int = 128,
    ) -> Dict[str, Any]:
        """
        Generate top-k recommendations for a batch of users based on their interaction sequences.
        This method computes the query embedding for each user, then performs a dot product with
        pre-computed candidate embeddings for all items.

        Args:
            users (torch.Tensor): Tensor of shape [num_users] with user IDs.
            item_sequences (torch.Tensor): Tensor of shape [num_users, seq_len] with user interaction sequences.
            k (int): Number of recommendations to generate for each user.
            batch_size (int, optional): Batch size for processing users. Defaults to 128.

        Returns:
            Dict[str, Any]: Dictionary containing:
                - "user_indice": List of user IDs (repeated per recommendation).
                - "recommendation": List of recommended item IDs.
                - "score": List of similarity scores.
        """
        self.eval()
        device = users.device
        # Create tensor of all candidate item IDs.
        all_items = torch.arange(self.item_embedding.num_embeddings, device=device)
        # Precompute candidate embeddings for all items.
        candidate_embeddings = self.get_candidate_embedding(
            all_items
        )  # Shape: [num_candidates, embedding_dim]

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
                # Compute query embedding for the current batch.
                query_embedding = self.get_query_embedding(
                    user_batch, item_seq_batch
                )  # [batch_size, embedding_dim]
                # Compute dot product similarity with all candidate embeddings.
                # candidate_embeddings: [num_candidates, embedding_dim] => use transpose for dot product.
                batch_scores = torch.matmul(
                    query_embedding, candidate_embeddings.t()
                )  # [batch_size, num_candidates]

                # Get top-k items for each user in the batch.
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
