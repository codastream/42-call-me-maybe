import json
from typing import Tuple


def _build_base_printable_mappings() -> Tuple[dict[int, str], dict[str, int]]:
    """Return a table associating byte with visible unicode character

    Maintain a list of bytes and chars that are zipped into a dict
    First, list bytes already having a printable value
    Then generate a guaranted printable value (starting with Unicode 256 = Ā) for remaining ones
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
    """Return decoded bytes to UTF-8 string"""
    try:
        return buf.decode('utf-8', errors='surrogateescape').strip()
    except Exception:
        return None

# ==================
# VOCABULARY CACHES
# ==================


def extract_and_cache_vocabulary(vocab_file_path: str) -> Tuple[dict[int, bytes], dict[int, str]]:
    """Extract vocabulary

    Raises:
      ValueError when json cannot be decoded
      RuntimeError in case of unknown error
    """
    try:
        with open(vocab_file_path, "r", encoding="utf-8") as vocab_file:
            raw_vocab = json.load(vocab_file)
    except json.JSONDecodeError:
        raise ValueError("Error: could not decode model vocabulary file")
    except Exception as e:
        raise Exception(f"Unexpected error while extracting vocabulary: {e}")

    vocab_raw_bytes: dict[int, bytes] = {}
    vocab_print: dict[int, str] = {}
    for token_str, token_id in raw_vocab.items():
        t_id = int(token_id)
        try:
            raw_bytes = bytes(PRINTABLE_TO_BYTE[char] for char in token_str)
        except KeyError:
            raw_bytes = token_str.encode("utf-8", errors="surrogateescape")
        vocab_raw_bytes[t_id] = raw_bytes
        vocab_print[t_id] = token_str

    return vocab_raw_bytes, vocab_print


def convert_token_str_to_bytes(token_str: str) -> bytes:
    """Convert a vocab token to real bytes"""
    return bytes(PRINTABLE_TO_BYTE[c] for c in token_str)


__all__ = ["BYTE_TO_PRINTABLE", "PRINTABLE_TO_BYTE", "extract_and_cache_vocabulary"]
