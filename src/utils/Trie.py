from __future__ import annotations
from typing import Optional


class TrieNode:
    """Hierarchical representation of vocabulary ordered by byte

    Attributes:
        children (dict[int, TrieNode]): Mapping from byte values to child nodes.
        token_ids (list[int]): List of token IDs that end precisely at this node.
    """

    def __init__(self) -> None:
        """Initialize children mapped with byte as a key and token ids for this node"""
        self.children: dict[int, "TrieNode"] = {}
        self.token_ids: list[int] = []

    @staticmethod
    def build_vocab_trie(token_to_bytes: dict[int, bytes]) -> TrieNode:
        """Parse vocabulary once and place each token in a node

        Args:
            token_to_bytes (dict[int, bytes]): mapping of token ids to bytes

        Returns:
            TrieNode: root node
        """
        root = TrieNode()
        for t_id, t_b in token_to_bytes.items():
            node = root
            for byte in t_b:
                node = node.children.setdefault(byte, TrieNode())
            node.token_ids.append(t_id)
        return root

    @staticmethod
    def get_token_ids_for_remaining(root: TrieNode, remaining: bytes) -> list[int]:
        """Collect token ids for all valid prefixes

        Args:
            root (TrieNode): root
            remaining (bytes): expected byte sequence to match against

        Returns:
            list[int]: list of token ids matching start or all of remaining bytes
        """
        node = root
        valids: list[int] = []
        for byte in remaining:
            valids.extend(node.token_ids)
            if byte not in node.children:
                break
            node = node.children[byte]
        else:
            valids.extend(node.token_ids)
        return valids

    def find_node(self, prefix: bytes) -> Optional[TrieNode]:
        """Return last node corresponding to the prefix sequence of bytes

        Args:
            prefix (bytes): prefix

        Returns:
            Optional[TrieNode]: node if the prefix can be expressed with vocabulary, None otherwise
        """
        node = self
        for byte in prefix:
            if byte not in node.children:
                return None
            node = node.children[byte]
        return node

    def get_all_subtree_token_ids(self) -> list[int]:
        """Recursively collectis token ids of node and all children nodes

        Returns:
            list[int]: token ids from the subtree
        """
        results: list[int] = list(self.token_ids)
        stack = list(self.children.values())
        while stack:
            curr = stack.pop()
            results.extend(curr.token_ids)
            stack.extend(curr.children.values())
        return results
