import numpy as np
import torch

import mlflow
from src.id_mapper import IDMapper


class SequenceRetrieverInferenceWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model):
        self.model = model
        self.model.eval()

    def load_context(self, context):
        """
        This load_context method is automatically called when later we load the model.
        """
        json_path = context.artifacts["idm"]
        self.idm = IDMapper().load(json_path)

    def predict(self, context, model_input, params=None):
        """
        Args:
            context: The context object in mlflow.pyfunc often contains pointers to artifacts that are logged
                     alongside the model during training (like feature encoders, embeddings, etc.)
            model_input: Expected to contain keys 'user_ids', 'candidate_items', and 'item_seq'.
        """
        if not isinstance(model_input, dict):
            # This is to work around the issue where MLflow automatically convert dict input into Dataframe
            # Ref: https://github.com/mlflow/mlflow/issues/11930
            model_input = model_input.to_dict(orient="records")[0]
        infer_output = self.infer(model_input).tolist()
        return {**model_input, "scores": infer_output}

    def infer(self, input_dict: dict):
        """
        Refactored infer method that accepts a dictionary of inputs.
        Expects the input dictionary to have keys:
            - "user_ids_raw": list of raw user IDs.
            - "item_seq_raw": list of lists, each containing raw item IDs for the sequence.
            - "candidate_items_raw": list of raw candidate item IDs.
        """
        sequence_length = 10
        padding_value = -1

        # Map raw IDs to indices using the IDMapper.
        raw_user_ids = input_dict["user_ids_raw"]
        raw_item_sequences = input_dict["item_seq_raw"]
        raw_item_ids = input_dict["candidate_items_raw"]

        user_indices = [self.idm.get_user_index(id_) for id_ in raw_user_ids]
        candidate_items = [self.idm.get_item_index(id_) for id_ in raw_item_ids]

        item_sequences = []
        for seq in raw_item_sequences:
            # Map each item in the sequence.
            seq_mapped = [self.idm.get_item_index(id_) for id_ in seq]
            padding_needed = sequence_length - len(seq_mapped)
            seq_padded = np.pad(
                seq_mapped,
                (padding_needed, 0),
                mode="constant",
                constant_values=padding_value,
            )
            item_sequences.append(seq_padded)

        # Convert to tensors.
        user_indices = torch.tensor(user_indices)
        item_sequences = torch.tensor(item_sequences)
        candidate_items = torch.tensor(candidate_items)

        # Pass a dictionary input to the model's predict method.
        output = self.model.predict(
            {
                "user_ids": user_indices,
                "item_seq": item_sequences,
                "candidate_items": candidate_items,
            }
        )
        return output.view(len(user_indices)).detach().numpy()
