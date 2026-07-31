import json
import re
from typing import Tuple

from src.models.TypeDef import TypeDef


def _build_base_printable_mappings() -> Tuple[dict[int, str], dict[str, int]]:
    """Return a table associating byte with visible unicode character

    Maintain a list of bytes and chars that are zipped into a dict
    First, list bytes already having a printable value
    Then generate a guaranted printable value (starting with Unicode 256 = Ā) for remaining ones

    Returns:
        Tuple[dict[int, str], dict[str, int]]: mappings of byte to its representaton and reverse mapping
    """

    printables = list(range(ord("!"), ord("~") + 1)) + \
        list(range(ord("¡"), ord("¬") + 1)) + \
        list(range(ord("®"), ord("ÿ") + 1))
    chars = printables[:]
    n = 0
    for b in range(256):
        if b not in printables:
            printables.append(b)
            chars.append(256 + n)
            n += 1

    byte_to_printable = {b: chr(c) for b, c in zip(printables, chars)}
    printable_to_byte = {chr(c): b for b, c in zip(printables, chars)}
    return byte_to_printable, printable_to_byte


BYTE_TO_PRINTABLE: dict[int, str] = {}

PRINTABLE_TO_BYTE: dict[str, int] = {}

BYTE_TO_PRINTABLE, PRINTABLE_TO_BYTE = _build_base_printable_mappings()


def bytes_to_str(buf: bytes) -> str | None:
    """Return decoded bytes to UTF-8 string

    Args:
        buf (bytes): buffer

    Returns:
        str | None: decoded bytes or None when a decode error occurred
    """
    try:
        return buf.decode('utf-8', errors='surrogateescape')
    except Exception:
        return None

# ==================
# VOCABULARY CACHES
# ==================


_PARTIAL_NUM = re.compile(r'^\s*[-+]?[0-9]*\.?[0-9]*([eE][-+]?[0-9]*)?$')

PATTERNS = {
    TypeDef.INTEGER:    re.compile(r'^\s*[-+]?\d+$'),
    TypeDef.FLOAT: re.compile(r'^\s*[-+]?\d+\.?\d*([eE][-+]?\d*)?$'),
    TypeDef.NUMBER:     _PARTIAL_NUM,
    TypeDef.BOOLEAN:    re.compile(r'^\s*(true|false|t|tr|tru|f|fa|fal|fals)$'),
    TypeDef.NULL:       re.compile(r'^\s*(null|n|nu|nul)$'),
}


def build_value_buckets(
        dic_id_to_bytes: dict[int, bytes]
) -> dict[TypeDef, set[int]]:
    """Precomputes acceptable token ids for JSON types

    Args:
        dic_id_to_bytes (dict[int, bytes]): map of token ids to bytes

    Returns:
        dict[TypeDef, list[int]]: buckets
    """
    buckets: dict[TypeDef, set[int]] = {t: set() for t in TypeDef}

    for t_id, t_b in dic_id_to_bytes.items():
        s = bytes_to_str(t_b)
        if not s:
            continue
        for type_def, pattern in PATTERNS.items():
            if pattern.match(s):
                buckets[type_def].add(t_id)
    return buckets


def extract_and_cache_vocabulary(vocab_file_path: str) -> Tuple[dict[int, bytes], dict[int, str], dict[bytes, int]]:
    """Extract vocabulary and build mappings to be cached


    Args:
        vocab_file_path (str): vocabulary file path

    Raises:
      ValueError when json cannot be decoded
      Exception in case of unknown error

    Returns:
        Tuple[dict[int, bytes], dict[int, str], dict[bytes, int]]: mappings
    """
    try:
        with open(vocab_file_path, "r", encoding="utf-8") as vocab_file:
            raw_vocab = json.load(vocab_file)
    except json.JSONDecodeError:
        raise ValueError("Error: could not decode model vocabulary file")
    except Exception as e:
        raise Exception(f"Unexpected error while extracting vocabulary: {e}")

    dic_id_to_bytes: dict[int, bytes] = {}
    dic_id_to_print: dict[int, str] = {}
    dic_bytes_to_id: dict[bytes, int] = {}
    for token_str, token_id in raw_vocab.items():
        t_id = int(token_id)
        try:
            raw_bytes = bytes(PRINTABLE_TO_BYTE[char] for char in token_str)
        except KeyError:
            raw_bytes = token_str.encode("utf-8", errors="surrogateescape")
        dic_id_to_bytes[t_id] = raw_bytes
        dic_bytes_to_id[raw_bytes] = t_id
        dic_id_to_print[t_id] = token_str

    return dic_id_to_bytes, dic_id_to_print, dic_bytes_to_id


def convert_token_str_to_bytes(token_str: str) -> bytes:
    """Convert a vocab token to real bytes

    Args:
        token_str (str): human_readable string

    Returns:
        bytes: bytes that can be decoded by model
    """
    return bytes(PRINTABLE_TO_BYTE[c] for c in token_str)


__all__ = ["BYTE_TO_PRINTABLE", "PRINTABLE_TO_BYTE", "extract_and_cache_vocabulary"]
