from src.models.TypeDef import TypeDef
from src.utils.Trie import TrieNode
from src.matcher.TokenMatcher import TokenMatcher


class ChoiceMatcher(TokenMatcher):
    """Check validity among many possible states"""

    def __init__(self, acceptable_targets: list[bytes]):
        self.acceptable_targets = acceptable_targets
        self.matched_target: bytes | None = None

    def prefilter_candidates(self, current_buf: bytes, token_id_to_bytes: dict[int, bytes], trie_root: TrieNode,
                             value_buckets: dict[TypeDef, list[int]]) -> list[int]:
        ids: set[int] = set()
        for target in self.acceptable_targets:
            if target.startswith(current_buf):
                remaining = target[len(current_buf):]
                ids.update(trie_root.get_token_ids_for_remaining(remaining))
        return list(ids)

    def evaluate(self, buf: bytes) -> bool:
        """Return True if at least one target starts with buffer"""
        return any(t.startswith(buf) for t in self.acceptable_targets)

    def is_complete(self, buf: bytes) -> bool:
        """Return True if one target matches exactly the buffer"""
        return any(buf.startswith(t) for t in self.acceptable_targets)

    def commit(self, buf: bytes) -> None:
        """Find matching"""
        for t in self.acceptable_targets:
            if buf.startswith(t):
                self.matched_target = t
                break

    def leftover_bytes(self, buf: bytes) -> bytes:
        if self.matched_target and buf.startswith(self.matched_target):
            return buf[len(self.matched_target):]
        return b""

    def display_name(self) -> str:
        return "Choice Matcher"

    def display_state(self) -> str:
        state = "targets:\n"
        for t in self.acceptable_targets:
            if t == self.matched_target:
                state += f"{t.decode()} [selected]\n"
            else:
                state += f"{t.decode()}\n"
        return state
