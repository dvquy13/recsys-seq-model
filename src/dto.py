from typing import List, Optional

from pydantic import BaseModel


class RetrieveContext(BaseModel):
    user_ids_raw: Optional[List[str]] = []
    item_seq_raw: Optional[List[List[str]]] = [[]]
    candidate_items_raw: Optional[List[str]] = []
