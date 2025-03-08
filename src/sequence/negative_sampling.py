import numpy as np
import pandas as pd
from tqdm.auto import tqdm


def generate_negative_samples(
    df,
    user_col="user_indice",
    item_col="item_indice",
    label_col="rating",
    neg_label=0,
    seed=None,
):
    """
    Generate negative samples for a user-item interaction DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing user-item interactions.
        user_col (str): Column name representing users.
        item_col (str): Column name representing items.
        label_col (str): Column name for the interaction label (e.g., rating).
        neg_label (int): Label to assign to negative samples (default is 0).
        seed (int, optional): Seed for random number generator to ensure reproducibility.

    Returns:
        pd.DataFrame: DataFrame containing generated negative samples.
    """

    # Set the random seed if provided for reproducibility.
    if seed is not None:
        np.random.seed(seed)

    # Calculate item popularity based on how frequently they appear in the DataFrame.
    item_popularity = df[item_col].value_counts()

    # Define all unique items from the DataFrame.
    items = item_popularity.index.values
    all_items_set = set(items)

    # Create a dictionary mapping each user to the set of items they interacted with.
    user_item_dict = df.groupby(user_col)[item_col].apply(set).to_dict()

    # Prepare popularity values for sampling probabilities.
    popularity = item_popularity.values.astype(np.float64)

    # Calculate item sampling probabilities proportional to their popularity.
    total_popularity = popularity.sum()
    if total_popularity == 0:
        # Handle edge case where no items have popularity by using uniform distribution.
        sampling_probs = np.ones(len(items)) / len(items)
    else:
        sampling_probs = popularity / total_popularity

    # Create a mapping from item to index to quickly access item-related data.
    item_to_index = {item: idx for idx, item in enumerate(items)}

    # Initialize a list to store negative samples for each user.
    negative_samples = []

    # Initialize the progress bar to track the process of generating negative samples for each user.
    total_users = len(user_item_dict)
    progress_bar = tqdm(
        user_item_dict.items(), total=total_users, desc="Generating Negative Samples"
    )

    # Iterate through each user and their positive interactions (items they interacted with).
    for user, pos_items in progress_bar:
        num_pos = len(pos_items)  # Number of positive interactions.

        # Identify items not interacted with by the user (negative candidates).
        negative_candidates = all_items_set - pos_items
        num_neg_candidates = len(negative_candidates)

        if num_neg_candidates == 0:
            # If the user interacted with all items, skip this user.
            continue

        # The number of negative samples to generate equals the number of positive interactions, or fewer if there aren't enough candidates.
        num_neg = min(num_pos, num_neg_candidates)

        # Convert the set of negative candidates to a list for indexing.
        negative_candidates_list = list(negative_candidates)

        # Obtain the indices and probabilities for the negative candidates.
        candidate_indices = [item_to_index[item] for item in negative_candidates_list]
        candidate_probs = sampling_probs[candidate_indices]
        candidate_probs /= candidate_probs.sum()  # Normalize probabilities.

        # Sample negative items for the user based on their probabilities.
        sampled_items = np.random.choice(
            negative_candidates_list, size=num_neg, replace=False, p=candidate_probs
        )

        # Append the negative samples (user, item) pairs to the list.
        negative_samples.extend([(user, item) for item in sampled_items])

    # Convert the list of negative samples to a DataFrame.
    df_negative = pd.DataFrame(negative_samples, columns=[user_col, item_col])
    # Assign the label for negative samples.
    df_negative[label_col] = neg_label

    return df_negative


def add_features_to_neg_df(pos_df, neg_df, user_col, timestamp_col, feature_cols=[]):
    """
    Add features from positive samples to negative samples DataFrame.

    Args:
        pos_df (pd.DataFrame): DataFrame with positive samples containing features.
        neg_df (pd.DataFrame): DataFrame with negative samples.
        user_col (str): Column name representing users.
        timestamp_col (str): Column name for the timestamp of interactions.
        feature_cols (list): List of feature column names to transfer from positive to negative samples.

    Returns:
        pd.DataFrame: Negative samples DataFrame with added features.
    """

    # Create a pseudo timestamp column for negative samples, incrementing by 1 for each user.
    neg_df = neg_df.assign(
        timestamp_pseudo=lambda df: df.groupby(user_col).cumcount() + 1
    )

    # Merge negative samples with corresponding positive samples based on user and pseudo timestamp.
    neg_df = pd.merge(
        neg_df,
        pos_df.assign(
            timestamp_pseudo=lambda df: df.groupby([user_col])[timestamp_col].rank(
                method="first"
            )
        )[[user_col, timestamp_col, "timestamp_pseudo", *feature_cols]],
        how="left",
        on=[user_col, "timestamp_pseudo"],
    ).drop(columns=["timestamp_pseudo"])

    return neg_df
