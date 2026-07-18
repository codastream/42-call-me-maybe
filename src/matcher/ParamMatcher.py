from typing import Any
import logging

from src.matcher.TokenMatcher import TokenMatcher, ValueMatcher
from src.matcher.MatcherState import MatcherState
from src.models import FunctionDefinition


class ParamMatcher(TokenMatcher):
    """Stay on stack till last param"""

    log = logging.getLogger('matcher_logger')

    def __init__(self, function_def: FunctionDefinition, close_suffix: bytes = b'}}') -> None:
        self.fun_def: FunctionDefinition = function_def
        self.evaluated_params: set[str] = set()
        self.sub_state = MatcherState.EXPECT_PARAM_KEY
        self.current_key: str | None = None
        self.all_keys = self.fun_def.parameters.keys()
        self.close_suffix = close_suffix

    def _allowed_keys(self) -> list[bytes]:
        prefix = b", " if self.evaluated_params else b""
        return [prefix + f'"{k}": '.encode('utf-8') for k in self.all_keys if k not in self.evaluated_params]

    def _value_matcher(self) -> ValueMatcher:
        return ValueMatcher(self.fun_def.parameters[self.current_key].type)

    def _remaining_key_targets(self, exclude_current: bool = True) -> list[bytes]:
        """Return remaining keys on format ', "key":' """
        keys = [k for k in self.all_keys
                if k not in self.evaluated_params and (not exclude_current or k != self.current_key)]
        return [b', ' + f'"{k}": '.encode('utf-8') for k in keys]

    def _next_targets(self) -> list[bytes]:
        """Return next keys"""
        remaining = self._remaining_key_targets()
        return remaining if remaining else [self.close_suffix]

    def _get_next_key_sequence(self, buf: bytes) -> bytes | None:
        """Look for a cut point between value and next

        next can be empty or the start of a ', "key": ' sequence
        Return None if no valid key can be extracted
        Return next if buffer contains a valid sequence for a remaining key
        """

        vm = self._value_matcher()
        targets = self._next_targets()

        for i in range(len(buf), 0, -1):
            endval, next = buf[:i], buf[i:]
            if not vm.is_complete(endval):
                continue

            if next != b"" and any(t.startswith(next) for t in targets):
                return next
        return None

    def evaluate(self, buf: bytes) -> Any | bool:
        if self.sub_state == MatcherState.EXPECT_PARAM_KEY:
            return any(k.startswith(buf) for k in self._allowed_keys())
        vm = self._value_matcher()
        if vm.evaluate(buf):
            return True
        return self._get_next_key_sequence(buf) is not None

    def segment_complete(self, buf: bytes) -> Any | bool:
        """Return True when a key of val segment is complete"""
        if self.sub_state == MatcherState.EXPECT_PARAM_KEY:
            return bool(buf in self._allowed_keys())
        vm = self._value_matcher()
        if not vm.is_complete(buf) and self._get_next_key_sequence(buf) is None:
            return False
        if vm.is_unambiguous_terminal(buf):
            return True
        return self._get_next_key_sequence(buf) is not None

    @property
    def is_done(self) -> bool:
        """Return true if all params have been evaluated"""
        return len(self.evaluated_params) == len(self.all_keys)

    def is_complete(self, buf: bytes) -> bool:
        """Return True if the matcher has reviewed all expected keys"""
        if self.sub_state != MatcherState.EXPECT_PARAM_VAL:
            return False

        remaining_keys = len(self.all_keys) - len(self.evaluated_params) - 1
        if remaining_keys > 0:
            return False

        vm = self._value_matcher()
        for i in range(len(buf), 0, -1):
            endval, next_seq = buf[:i], buf[i:]
            if vm.is_unambiguous_terminal(endval) and next_seq == b"":
                return True
        return False

    def commit(self, buf: bytes) -> bytes:
        """Return not consumed bytes to be reinjected as start of next segment"""
        if self.sub_state == MatcherState.EXPECT_PARAM_KEY:
            if buf in self._allowed_keys():
                self.current_key = buf.decode('utf-8', errors='surrogateescape').split('"')[1]
                self.sub_state = MatcherState.EXPECT_PARAM_VAL
            return b""

        vm = self._value_matcher()
        if vm.is_complete(buf) and getattr(vm.type, "name", "") != "NUMBER":
            if self.current_key:
                self.evaluated_params.add(self.current_key)
            self.current_key = None
            self.sub_state = MatcherState.EXPECT_PARAM_KEY
            return b""

        next_key_start = self._get_next_key_sequence(buf)
        if next_key_start is not None:
            if self.current_key:
                self.evaluated_params.add(self.current_key)
            self.current_key = None
            self.sub_state = MatcherState.EXPECT_PARAM_KEY
            return next_key_start

        return b""
