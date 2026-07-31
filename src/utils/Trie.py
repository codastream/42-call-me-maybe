from __future__ import annotations
from typing import Optional


class TrieNode:
    """Hierarchical representation of vocabulary ordered by byte"""

    def __init__(self) -> None:
        """Initialize children mapped with byte as a key and token ids for this node"""
        self.children: dict[int, "TrieNode"] = {}
        self.token_ids: list[int] = []

    @staticmethod
    def build_vocab_trie(token_to_bytes: dict[int, bytes]) -> TrieNode:
        """Parse vocabulary once and place each token in a node"""
        root = TrieNode()
        for t_id, t_b in token_to_bytes.items():
            node = root
            for byte in t_b:
                node = node.children.setdefault(byte, TrieNode())
            node.token_ids.append(t_id)
        return root

    def get_token_ids_for_remaining(root: TrieNode, remaining: bytes) -> list[int]:
        """Return All token ids which are a prefix of remaining"""
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
        node = self
        for byte in prefix:
            if byte not in node.children:
                return None
            node = node.children[byte]
        return node

    def get_all_subtree_token_ids(self) -> list[int]:
        results: list[int] = list(self.token_ids)
        stack = list(self.children.values())
        while stack:
            curr = stack.pop()
            results.extend(curr.token_ids)
            stack.extend(curr.children.values())
        return results
