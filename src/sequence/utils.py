import numpy as np


def generate_item_sequences(
    df,
    user_col,
    item_col,
    timestamp_col,
    sequence_length,
    padding=True,
    padding_value=-1,
):
    """
    Generates a column 'item_sequence' containing lists of previous item indices for each user.

    Parameters:
    - df: DataFrame containing the data
    - user_col: The name of the user column
    - item_col: The name of the item column
    - timestamp_col: The name of the timestamp column
    - sequence_length: The maximum length of the item sequence to keep
    - padding: whether to pad the item sequence with `padding_value` if it's shorter than sequence_length
    - padding_value: value used for padding

    Returns:
    - DataFrame with an additional column 'item_sequence'
    """

    def get_item_sequence(sub_df):
        sequences = []
        for i in range(len(sub_df)):
            prev_df = sub_df.loc[
                lambda df: df[timestamp_col].lt(sub_df[timestamp_col].iloc[i])
            ]
            # Get item indices up to the current row (excluding the current row)
            sequence = prev_df[item_col].tolist()[-sequence_length:]
            if padding:
                padding_needed = sequence_length - len(sequence)
                sequence = np.pad(
                    sequence,
                    (padding_needed, 0),  # Add padding at the beginning
                    "constant",
                    constant_values=padding_value,
                )
            sequences.append(sequence)
        return sequences

    agg_df = df.sort_values([user_col, timestamp_col])
    item_sequences = agg_df.groupby(user_col, group_keys=True).apply(get_item_sequence)
    item_sequences_flatten = (
        item_sequences.to_frame("item_sequence").reset_index().explode("item_sequence")
    )
    agg_df["item_sequence"] = (
        # Need to use .values to avoid auto index mapping between agg_df and item_sequences_flatten which causes miss join
        item_sequences_flatten["item_sequence"].fillna("").apply(list).values
    )

    return agg_df
