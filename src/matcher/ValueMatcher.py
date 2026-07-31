import re

from src.models.TypeDef import TypeDef
from src.config import get_logger
from src.utils.convert import bytes_to_str
from src.utils.Trie import TrieNode
from src.matcher.TokenMatcher import TokenMatcher


log = get_logger("match")


class ValueMatcher(TokenMatcher):
    """Check that the buffer maintains type consistency"""

    PARTIAL_NUM_PATTERNS = {
        TypeDef.INTEGER:    re.compile(r'^[-+]?[0-9]*$'),
        TypeDef.FLOAT:      re.compile(r'^[-+]?[0-9]*\.?[0-9]*$'),
        TypeDef.NUMBER:     re.compile(r'^[-+]?[0-9]*\.?[0-9]*([eE][-+]?[0-9]*)?$')
    }

    FULL_NUM_PATTERNS = {
        TypeDef.INTEGER:    re.compile(r'^[-+]?[0-9]+$'),
        TypeDef.FLOAT:      re.compile(r'^[-+]?[0-9]+\.[0-9]+$'),
        TypeDef.NUMBER:     re.compile(r'^[-+]?[0-9]+(\.?[0-9]+)?([eE][-+]?[0-9]+)?$')
    }

    EXTRACT_NUM_PATTERNS = {
        TypeDef.INTEGER:    re.compile(r'^[-+]?[0-9]+'),
        TypeDef.FLOAT:      re.compile(r'^[-+]?[0-9]+\.[0-9]+'),
        TypeDef.NUMBER:     re.compile(r'^[-+]?[0-9]+\.?[0-9]*([eE][-+]?[0-9]+)?')
    }

    @staticmethod
    def _find_string_end(s: str) -> int:
        """Return index of closing unescaped quote after pos 0, or -1"""
        if not s or s[0] != '"':
            return -1
        i = 1
        while i < len(s):
            if s[i] == '"':
                j = i - 1
                num_backslashes = 0
                while j >= 0 and s[j] == '\\':
                    num_backslashes += 1
                    j -= 1
                if num_backslashes % 2 == 0:
                    return i
            i += 1
        return -1

    @staticmethod
    def _count_unescaped_quotes(s: str) -> int:
        """Count quotes not preceded by an odd number of backslashes"""
        count = 0
        i = 0
        while i < len(s):
            if s[i] == '"':
                j = i - 1
                num_backslashes = 0
                while j >= 0 and s[j] == '\\':
                    num_backslashes += 1
                    j -= 1
                if num_backslashes % 2 == 0:
                    count += 1
            i += 1
        return count

    def __init__(self, type_def: TypeDef, value_buckets: dict[TypeDef, set[int]]):
        """Initializer with type"""
        self.type = type_def
        self.allowed_bucket: set[int] = value_buckets.get(type_def, set())

    def prefilter_candidates(
            self,
            current_buf: bytes,
            token_id_to_bytes: dict[int, bytes],
            trie_root: TrieNode,
            value_buckets: dict[TypeDef, set[int]]
    ) -> set[int]:
        """Filter the vocabulary tokens to iterate around"""
        if not current_buf:
            log.debug("start of ValueMatcher, returning all allowed tokens for its type")
            return self.allowed_bucket

        if self.is_complete(current_buf):
            return self.allowed_bucket

        if self.type == TypeDef.STRING or self.type in (TypeDef.INTEGER, TypeDef.FLOAT, TypeDef.NUMBER):
            return set(token_id_to_bytes.keys())

        return self.allowed_bucket

    def evaluate(self, buf: bytes) -> bool:
        """Return True if buffer maintains type consistency for a non final value"""
        s = bytes_to_str(buf)
        if s is None:
            return False
        stripped = s.lstrip()
        if not stripped:
            return True

        if self.type == TypeDef.INTEGER or self.type == TypeDef.FLOAT or self.type == TypeDef.NUMBER:
            return bool(self.PARTIAL_NUM_PATTERNS[self.type].match(stripped))

        if self.type == TypeDef.BOOLEAN:
            return "true".startswith(stripped) or "false".startswith(stripped)

        if self.type == TypeDef.STRING:
            if stripped[0] != '"':
                print(f"[DEBUG ValueMatcher STRING] rejected for not starting with '\"': {s!r}")
                return False
            if stripped == '"':
                return True
            return self._find_string_end(stripped) == -1

        return False

    def is_unambiguous_terminal(self, buf: bytes) -> bool:
        """Return True if is_complete and type implies a recognizable end of value delimiter"""
        if self.type == TypeDef.NUMBER or self.type == TypeDef.STRING:
            return False
        return self.is_complete(buf)

    def is_complete(self, buf: bytes) -> bool:
        """Return True if buffer maintains type consistency for a final value"""
        s = bytes_to_str(buf)
        # log.debug(f"is_complete: type={self.type} buf={buf!r} s={s!r} not_s={not s} len={len(s) if s else 'N/A'}")
        if not s:
            return False
        stripped = s.lstrip()
        if not stripped:
            return False

        if self.type == TypeDef.INTEGER or self.type == TypeDef.FLOAT or self.type == TypeDef.NUMBER:
            if bool(self.FULL_NUM_PATTERNS[self.type].match(stripped)):
                return True
            match = self.EXTRACT_NUM_PATTERNS[self.type].match(stripped)
            return match is not None and len(match.group(0)) > 0

        if self.type == TypeDef.BOOLEAN:
            return stripped in ("true", "false")

        if self.type == TypeDef.STRING:
            if stripped[0] != '"':
                return False
            return self._count_unescaped_quotes(stripped) >= 2
        return False

    def display_name(self) -> str:
        """Get name"""
        return "Value Matcher"

    def display_state(self) -> str:
        """Get state"""
        return f"type : {self.type.name}"

    def leftover_bytes(self, buf: bytes) -> bytes:
        """Return bytes not corresponding any more to the matcher"""

        s = bytes_to_str(buf)
        if not s:
            return b""

        if self.type == TypeDef.BOOLEAN:
            for keyword in (b"false", b"true"):
                if keyword in buf:
                    idx = buf.find(keyword)
                    return buf[idx + len(keyword):]
            return b""

        if self.type == TypeDef.INTEGER or self.type == TypeDef.FLOAT or self.type == TypeDef.NUMBER:
            stripped = s.lstrip()
            match = self.EXTRACT_NUM_PATTERNS[self.type].match(stripped)
            if not match:
                return b""
            matched_text = match.group(0)
            leading_space_len = len(s) - len(stripped)
            consumed_len = len((s[:leading_space_len] + matched_text).encode(errors="surrogateescape"))
            return buf[consumed_len:]

        if self.type == TypeDef.STRING:
            end = self._find_string_end(s)
            if end == -1:
                return b""
            matched_text = s[:end + 1]
            matched_bytes = matched_text.encode(errors="surrogateescape")
            return buf[len(matched_bytes):]

        return b""
