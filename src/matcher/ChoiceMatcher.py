from src.models.TypeDef import TypeDef
from src.utils.Trie import TrieNode
from src.matcher.TokenMatcher import TokenMatcher


class ChoiceMatcher(TokenMatcher):
    """Check validity among many possible states"""

    def __init__(self, acceptable_targets: list[bytes]):
        """Initializer"""
        self.acceptable_targets = acceptable_targets
        self.matched_target: bytes | None = None

    def prefilter_candidates(self, current_buf: bytes, token_id_to_bytes: dict[int, bytes], trie_root: TrieNode,
                             value_buckets: dict[TypeDef, set[int]]) -> set[int]:
        """Return ids of tokens completing any of its targets AND matching current buffer

        Args:
            current_buf (bytes): current buffer
            token_id_to_bytes (dict[int, bytes]): mapping of token id to their byte equivalent
            trie_root (TrieNode): Trie for model vocabulary
            value_buckets (dict[TypeDef, set[int]]): buckets of prefiltered token ids by value type

        Returns:
            set[int]: filtered token ids
        """
        ids: set[int] = set()
        for target in self.acceptable_targets:
            if target.startswith(current_buf):
                remaining = target[len(current_buf):]
                ids.update(trie_root.get_token_ids_for_remaining(trie_root, remaining))
        return ids

    def evaluate(self, buf: bytes) -> bool:
        """Return True if at least one target starts with bytes passed as argument

        Args:
            buf (bytes): buffer

        Returns:
            bool: True if any target starts with buffer
        """
        return any(t.startswith(buf) for t in self.acceptable_targets)

    def is_complete(self, buf: bytes) -> bool:
        """Return True if at least one target starts with all bytes passed as argument

        Note:
            cannot check for exact match as buffer can also contain next matcher elements

        Args:
            buf (bytes): buffer

        Returns:
            bool: True if any target  starts with all bytes passed as argument
        """
        return any(buf.startswith(t) for t in self.acceptable_targets)

    def commit(self, buf: bytes) -> None:
        """Determine matched target

        Args:
            buf (bytes): buffer
        """
        for t in self.acceptable_targets:
            if buf.startswith(t):
                self.matched_target = t
                break

    def leftover_bytes(self, buf: bytes) -> bytes:
        """Return bytes from buffer not matching current target

        Args:
            buf (bytes): buffer

        Returns:
            bytes: bytes from buffer not matching current target
        """
        if self.matched_target and buf.startswith(self.matched_target):
            return buf[len(self.matched_target):]
        return b""

    def display_name(self) -> str:
        """Return human readable name of matcher

        Returns:
            str: name
        """
        return "Choice Matcher"

    def display_state(self) -> str:
        """Return human readable state of matcher with current matched starget highlighted

        Returns:
            str: list of targets
        """
        state = "targets:\n"
        for t in self.acceptable_targets:
            if t == self.matched_target:
                state += f"{t.decode()} [selected]\n"
            else:
                state += f"{t.decode()}\n"
        return state
