import re
from abc import ABC, abstractmethod

from src.models.TypeDef import TypeDef
from src.utils import bytes_to_str


class TokenMatcher(ABC):

    @abstractmethod
    def evaluate(self, buf: bytes) -> bool:
        """Return True if buf is valid for this state"""

    @abstractmethod
    def is_complete(self, buf: bytes) -> bool:
        """Return True if state does not require content"""

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
        return buf == self.target


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
        return buf in self.acceptable_targets

    def commit(self, buf: bytes) -> None:
        self.matched_target = buf


class ValueMatcher(TokenMatcher):
    """Check that the buffer maintains type consistency"""

    _PARTIAL_NUM_RE = re.compile(r'^[-+]?[0-9]*\.?[0-9]*([eE][-+]?[0-9]*)?$')
    _FULL_NUM_RE = re.compile(r'^[-+]?[0-9]+\.?[0-9]*([eE][-+]?[0-9]+)?$')

    def __init__(self, type: TypeDef):
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
            quotes = len(re.findall(r'(?<!\\)"', s))
            if quotes <= 1:
                return True
            if quotes == 2:
                return bool(s.endswith('"'))
            return False

        return False

    def is_unambiguous_terminal(self, buf: bytes) -> bool:
        """Return True if is_complete and type implies a recognizable end of value delimiter"""
        if self.type == TypeDef.NUMBER:
            return False
        return self.is_complete(buf)

    def is_complete(self, buf: bytes) -> bool:
        """Return True if buffer maintains type consistency for a final value"""
        s = bytes_to_str(buf)
        if not s:
            return False

        if self.type == TypeDef.NUMBER:
            return bool(self._FULL_NUM_RE.match(s))

        if self.type == TypeDef.BOOLEAN:
            return s in ("true", "false")

        if self.type == TypeDef.STRING:
            return s.startswith('"') and s.endswith('"') and len(s) >= 2

        return False
