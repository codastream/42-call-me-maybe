from src.models.TypeDef import TypeDef
from src.utils.Trie import TrieNode
from src.matcher.TokenMatcher import TokenMatcher


class StaticSequenceMatcher(TokenMatcher):
    """Check validity against one expected static byte sequence"""

    def __init__(self, target: bytes):
        self.target = target

    def prefilter_candidates(self, current_buf: bytes, token_id_to_bytes: dict[int, bytes], trie_root: TrieNode,
                             value_buckets: dict[TypeDef, set[int]]) -> set[int]:
        """Return ids of tokens completing its targets AND matching current buffer

        Args:
            current_buf (bytes): current buffer
            token_id_to_bytes (dict[int, bytes]): mapping of token id to their byte equivalent
            trie_root (TrieNode): Trie for model vocabulary
            value_buckets (dict[TypeDef, set[int]]): buckets of prefiltered token ids by value type

        Returns:
            set[int]: matching token ids
        """
        remaining = self.target[len(current_buf):]
        return set(trie_root.get_token_ids_for_remaining(trie_root, remaining))

    def evaluate(self, buf: bytes) -> bool:
        """Return True if target starts with bytes passed as argument

        Args:
            buf (bytes): buffer

        Returns:
            bool: True if target starts with buffer
        """
        return self.target.startswith(buf)

    def is_complete(self, buf: bytes) -> bool:
        """Return True if target starts with all bytes passed as argument

        Note:
            cannot check for exact match as buffer can also contain next matcher elements

        Args:
            buf (bytes): buffer

        Returns:
            bool: True if target  starts with all bytes passed as argument
        """
        return buf.startswith(self.target)

    def commit(self, buf: bytes) -> None:
        pass

    def leftover_bytes(self, buf: bytes) -> bytes:
        """Return bytes from buffer not matching target

        Args:
            buf (bytes): buffer

        Returns:
            bytes: bytes from buffer not matching target
        """
        if buf.startswith(self.target):
            return buf[len(self.target):]
        return b""

    def display_name(self) -> str:
        """Return human readable name of matcher

        Returns:
            str: name
        """
        return "Static Sequence Matcher"

    def display_state(self) -> str:
        """Return human readable state of matcher with its target

        Returns:
            str: target
        """
        return f"target -> {self.target.decode()}"
