from abc import ABC, abstractmethod
from typing import List

import numpy as np
from qdrant_client import QdrantClient
from tqdm.auto import tqdm


class VectorStore(ABC):
    @abstractmethod
    def get_vector_by_ids(self, ids: List[int], chunk_size: int = 100) -> np.ndarray:
        """Retrieve vectors for a list of IDs."""
        pass

    @abstractmethod
    def get_neighbors_by_ids(self, ids: List[int], limit: int = 5):
        """Retrieve neighbor vectors given an ID list."""
        pass


class QdrantVectorStore(VectorStore):
    def __init__(self, qdrant_url: str, qdrant_collection_name: str):
        self.qdrant_client = QdrantClient(url=qdrant_url)
        self.qdrant_collection_name = qdrant_collection_name

        if not self.qdrant_client.collection_exists(qdrant_collection_name):
            raise Exception(
                f"Required Qdrant collection {qdrant_collection_name} does not exist"
            )

    def get_vector_by_ids(self, ids: List[int], chunk_size: int = 100) -> np.ndarray:
        records = []
        for i in tqdm(range(0, len(ids), chunk_size)):
            _ids = ids[i : i + chunk_size]
            _records = self.qdrant_client.retrieve(
                collection_name=self.qdrant_collection_name, ids=_ids, with_vectors=True
            )
            records.extend(_records)
        # Handle case where duplicated ids are sent to Qdrant then it returns back only the set of vectors
        if len(records) != len(ids):
            mapper = {record.id: record.vector for record in records}
            return np.array([mapper[id] for id in ids])
        return np.array([record.vector for record in records])

    def get_neighbors_by_ids(self, ids: List[int], limit: int = 5):
        vector = self.get_vector_by_ids(ids)[0]
        neighbors = self.qdrant_client.search(
            collection_name=self.qdrant_collection_name,
            query_vector=vector,
            limit=limit,
        )
        return neighbors


class VectorStoreFactory:
    @staticmethod
    def create_vectorstore(store_type: str, **kwargs) -> VectorStore:
        if store_type.lower() == "qdrant":
            return QdrantVectorStore(**kwargs)
        else:
            raise ValueError(f"Unknown vector store type: {store_type}")
