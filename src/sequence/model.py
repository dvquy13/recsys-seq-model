from typing import Any, Dict

import torch
import torch.nn as nn
from tqdm.auto import tqdm


class SequenceRatingPrediction(nn.Module):
    """
    A PyTorch neural network model for predicting user-item interaction ratings based on sequences of previous items
    and a target item. This model uses user and item embeddings, and performs rating predictions using fully connected layers.

    Args:
        num_users (int): The number of unique users.
        num_items (int): The number of unique items.
        embedding_dim (int): The size of the embedding dimension for both user and item embeddings.
        item_embedding (torch.nn.Embedding): pretrained item embeddings. Defaults to None.
        dropout (float, optional): The dropout probability applied to the fully connected layers. Defaults to 0.2.

    Attributes:
        num_items (int): Number of unique items.
        num_users (int): Number of unique users.
        item_embedding (nn.Embedding): Embedding layer for items, including a padding index for unknown items.
        user_embedding (nn.Embedding): Embedding layer for users.
        fc_rating (nn.Sequential): Fully connected layers for predicting the rating from concatenated embeddings.
        relu (nn.ReLU): ReLU activation function.
        dropout (nn.Dropout): Dropout layer applied after activation.
    """

    def __init__(
        self,
        num_users,
        num_items,
        embedding_dim,
        item_embedding=None,
        dropout=0.2,
    ):
        super().__init__()

        self.num_items = num_items
        self.num_users = num_users

        self.item_embedding = item_embedding
        if item_embedding is None:
            # Item embedding (Add 1 to num_items for the unknown item (-1 padding))
            self.item_embedding = nn.Embedding(
                num_items + 1,  # One additional index for unknown/padding item
                embedding_dim,
                padding_idx=num_items,  # The additional index for the unknown item
            )

        # User embedding
        self.user_embedding = nn.Embedding(num_users, embedding_dim)

        # GRU layer to process item sequences
        self.gru = nn.GRU(
            input_size=embedding_dim, hidden_size=embedding_dim, batch_first=True
        )

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
        padding_idx_tensor = torch.tensor(self.item_embedding.padding_idx)
        input_seq = torch.where(input_seq == -1, padding_idx_tensor, input_seq)
        target_item = torch.where(target_item == -1, padding_idx_tensor, target_item)

        # Embed input sequence
        embedded_seq = self.item_embedding(
            input_seq
        )  # Shape: [batch_size, seq_len, embedding_dim]

        # GRU processing: output the hidden states and the final hidden state
        _, hidden_state = self.gru(
            embedded_seq
        )  # hidden_state: [1, batch_size, embedding_dim]
        gru_output = hidden_state.squeeze(
            0
        )  # Remove the sequence dimension -> [batch_size, embedding_dim]

        # Embed the target item
        embedded_target = self.item_embedding(target_item)

        # Embed the user IDs
        user_embeddings = self.user_embedding(user_ids)

        # Concatenate the GRU output with the target item and user embeddings
        combined_embedding = torch.cat(
            (gru_output, embedded_target, user_embeddings), dim=1
        )

        # Project combined embedding to rating prediction
        output_ratings = self.fc_rating(combined_embedding)

        return output_ratings

    def predict(self, user, item_sequence, target_item):
        """
        Predict the rating for a specific user and item sequence using the forward method
        and applying a Sigmoid function to the output.

        Args:
            user (torch.Tensor): User ID.
            item_sequence (torch.Tensor): Sequence of previously interacted items.
            target_item (torch.Tensor): The target item to predict the rating for.

        Returns:
            torch.Tensor: Predicted rating after applying Sigmoid activation.
        """
        output_rating = self.forward(user, item_sequence, target_item)
        return output_rating

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
