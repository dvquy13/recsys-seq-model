import numpy as np
import torch

import mlflow
from src.id_mapper import IDMapper


class SequenceRatingPredictionInferenceWrapper(mlflow.pyfunc.PythonModel):
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
            context: The context object in mlflow.pyfunc often contains pointers to artifacts that are logged alongside the model during training (like feature encoders, embeddings, etc.)
        """
        sequence_length = 10
        padding_value = -1

        if not isinstance(model_input, dict):
            # This is to work around the issue where MLflow automatically convert dict input into Dataframe
            # Ref: https://github.com/mlflow/mlflow/issues/11930
            model_input = model_input.to_dict(orient="records")[0]
        user_indices = [self.idm.get_user_index(id_) for id_ in model_input["user_ids"]]
        item_indices = [self.idm.get_item_index(id_) for id_ in model_input["item_ids"]]
        item_sequences = []
        for item_sequence in model_input["item_sequences"]:
            item_sequence = [self.idm.get_item_index(id_) for id_ in item_sequence]
            padding_needed = sequence_length - len(item_sequence)
            item_sequence = np.pad(
                item_sequence,
                (padding_needed, 0),
                "constant",
                constant_values=padding_value,
            )
            item_sequences.append(item_sequence)
        infer_output = self.infer(user_indices, item_sequences, item_indices).tolist()
        return {
            **model_input,
            "scores": infer_output,
        }

    def infer(self, user_indices, item_sequences, item_indices):
        user_indices = torch.tensor(user_indices)
        item_sequences = torch.tensor(item_sequences)
        item_indices = torch.tensor(item_indices)
        output = self.model.predict(user_indices, item_sequences, item_indices)
        return output.view(len(user_indices)).detach().numpy()
