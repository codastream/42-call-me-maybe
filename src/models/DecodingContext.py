from pydantic import BaseModel, ConfigDict
from llm_sdk import Small_LLM_Model  # type: ignore[attr-defined]

from src.utils.Trie import TrieNode
from src.models.TypeDef import TypeDef
from src.matcher.AutomatonController import AutomatonController


class DecodingContext(BaseModel):
    """Global context

    Args:
        BaseModel (pydantic base model)
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    model: Small_LLM_Model
    tokenid_to_bytes: dict[int, bytes]
    tokenbytes_to_id: dict[bytes, int]
    trie_root: TrieNode
    value_buckets: dict[TypeDef, set[int]]
    controller: AutomatonController
    current_prompt: str
    available_fun: str
