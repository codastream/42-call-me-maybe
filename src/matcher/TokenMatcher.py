from abc import ABC, abstractmethod

from src.models.TypeDef import TypeDef
from src.utils.Trie import TrieNode
from src.config import get_logger


log = get_logger("match")


class TokenMatcher(ABC):

    @abstractmethod
    def prefilter_candidates(self, current_buf: bytes, token_id_to_bytes: dict[int, bytes], trie_root: TrieNode,
                             value_buckets: dict[TypeDef, set[int]]) -> list[int]:
        """Filter token still acceptable given current buffer"""

    @abstractmethod
    def evaluate(self, buf: bytes) -> bool:
        """Return True if buf is valid for this state"""

    @abstractmethod
    def is_complete(self, buf: bytes) -> bool:
        """Return True if state does not require content"""

    @abstractmethod
    def display_name(self) -> str:
        """Display matcher name"""

    @abstractmethod
    def display_state(self) -> str:
        """Display matcher state"""

    def commit(self, buf: bytes) -> None:
        """Define buffer as the valid. Should be called for the chosen token only"""
        return None

    @abstractmethod
    def leftover_bytes(self, buf: bytes) -> bytes:
        """Return bytes not corresponding any more to the matcher"""
