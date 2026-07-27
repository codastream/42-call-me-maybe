from src.models.TypeDef import TypeDef
from src.utils.Trie import TrieNode
from src.matcher.TokenMatcher import TokenMatcher


class StaticSequenceMatcher(TokenMatcher):
    """Check validity against one expected static byte sequence"""

    def __init__(self, target: bytes):
        self.target = target

    def prefilter_candidates(self, current_buf: bytes, token_id_to_bytes: dict[int, bytes], trie_root: TrieNode,
                             value_buckets: dict[TypeDef, list[int]]) -> list[int]:
        remaining = self.target[len(current_buf):]
        return trie_root.get_token_ids_for_remaining(remaining)

    def evaluate(self, buf: bytes) -> bool:
        return self.target.startswith(buf)

    def is_complete(self, buf: bytes) -> bool:
        return buf.startswith(self.target)

    def commit(self, buf: bytes) -> None:
        pass

    def display_name(self) -> str:
        return "Static Sequence Matcher"

    def display_state(self) -> str:
        return f"target -> {self.target.decode()}"

    def leftover_bytes(self, buf: bytes) -> bytes:
        if buf.startswith(self.target):
            return buf[len(self.target):]
        return b""
