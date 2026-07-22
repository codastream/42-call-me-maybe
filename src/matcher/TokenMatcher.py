import re
from abc import ABC, abstractmethod

from src.models.TypeDef import TypeDef
from src.utils.convert import bytes_to_str
from src.config import get_logger


log = get_logger("match")


class TokenMatcher(ABC):

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


class StaticSequenceMatcher(TokenMatcher):
    """Check validity against one expected static byte sequence"""

    def __init__(self, target: bytes):
        self.target = target

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


class ChoiceMatcher(TokenMatcher):
    """Check validity among many possible states"""

    def __init__(self, acceptable_targets: list[bytes]):
        self.acceptable_targets = acceptable_targets
        self.matched_target: bytes | None = None

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


class ValueMatcher(TokenMatcher):
    """Check that the buffer maintains type consistency"""

    _PARTIAL_NUM_RE = re.compile(r'^[-+]?[0-9]*?$')
    _FULL_NUM_RE = re.compile(r'^[-+]?[0-9]+$')
    _EXTRACT_NUM_RE = re.compile(r'^[-+]?[0-9]+')
    # _PARTIAL_NUM_RE   = re.compile(r'^[-+]?[0-9]*\.?[0-9]*([eE][-+]?[0-9]*)?$')
    # _FULL_NUM_RE      = re.compile(r'^[-+]?[0-9]+\.?[0-9]*([eE][-+]?[0-9]+)?$')

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

    def __init__(self, type: TypeDef):
        """Initializer with type"""
        self.type = type

    def evaluate(self, buf: bytes) -> bool:
        """Return True if buffer maintains type consistency for a non final value"""
        s = bytes_to_str(buf)
        if s is None:
            return False
        if not s:
            return True

        if self.type == TypeDef.NUMBER:
            return bool(self._PARTIAL_NUM_RE.match(s))

        if self.type == TypeDef.BOOLEAN:
            return "true".startswith(s) or "false".startswith(s)

        if self.type == TypeDef.STRING:
            if not s.startswith('"'):
                return False
            if len(s) == 1:
                return True
            end = self._find_string_end(s)
            if end == -1:
                return True
            return bool(s[end + 1:] == "")

        return False

    def is_unambiguous_terminal(self, buf: bytes) -> bool:
        """Return True if is_complete and type implies a recognizable end of value delimiter"""
        if self.type == TypeDef.NUMBER:
            return False
        return self.is_complete(buf)

    def is_complete(self, buf: bytes) -> bool:
        """Return True if buffer maintains type consistency for a final value"""
        s = bytes_to_str(buf)
        log.debug(f"is_complete: type={self.type} buf={buf!r} s={s!r} not_s={not s} len={len(s) if s else 'N/A'}")
        if not s:
            return False

        if self.type == TypeDef.NUMBER:
            return bool(self._FULL_NUM_RE.match(s))

        if self.type == TypeDef.BOOLEAN:
            return s in ("true", "false")

        if self.type == TypeDef.STRING:
            end = self._find_string_end(s)
            if len(s) < 3:
                return False
            if end == -1 or end == 1:
                return False
            nb_quotes = self._count_unescaped_quotes(s)
            if nb_quotes < 2:
                return False
            return True
        return False

    def display_name(self) -> str:
        """Get name"""
        return "Value Matcher"

    def display_state(self) -> str:
        """Get state"""
        return f"type : {self.type.name}"

    def leftover_bytes(self, buf: bytes) -> bytes:
        """Return bytes belonging to next matcher"""

        s = bytes_to_str(buf)
        if not s:
            return b""

        if self.type == TypeDef.BOOLEAN:
            false_b = "false".encode(errors="surrogateescape")
            false_idx = buf.rfind(false_b)
            if false_idx != -1:
                return buf[false_idx + len(false_b):]
            true_b = "true".encode(errors="surrogateescape")
            true_idx = buf.rfind(true_b)
            if true_idx != -1:
                return buf[true_idx + len(true_b):]
            return b""

        elif self.type == TypeDef.NUMBER:
            match = self._EXTRACT_NUM_RE.match(s)
            if not match:
                return b""
            end_idx = match.end()
            matched_bytes = s[:end_idx].encode(errors="surrogateescape")
            return buf[len(matched_bytes):]

        elif self.type == TypeDef.STRING:
            end = self._find_string_end(s)
            matched_bytes = s[:end + 1].encode(errors="surrogateescape")
            start_idx = buf.find(matched_bytes)
            if start_idx == -1:
                return b""

            cut_pos = start_idx + len(matched_bytes)

            return buf[cut_pos:]
        else:
            return b""
