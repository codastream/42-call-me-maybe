
from src.matcher.MatcherState import MatcherState
from src.matcher.ParamMatcher import ParamMatcher
from src.matcher.TokenMatcher import TokenMatcher, StaticSequenceMatcher, ChoiceMatcher
from src.models.FunctionDefinition import FunctionDefinition
from src.utils.convert import convert_token_str_to_bytes

import logging


class JSONSchemaMatcher:
    PROMPT_PREFIX: bytes = b'{"prompt": "'
    NAME_PREFIX: bytes = b'", "name": "'
    PARAM_PREFIX: bytes = b'", "parameters": {'
    CLOSE_SUFFIX: bytes = b'}}'

    log = logging.getLogger('matcher_logger')

    def __init__(self, fun_defs: list[FunctionDefinition], initial_prompt: bytes):

        self.initial_prompt_b: bytes = initial_prompt
        self.state: MatcherState = MatcherState.START
        self.selected_function: FunctionDefinition = None
        self.fun_defs: list[FunctionDefinition] = fun_defs
        self.current_buffer_b: bytes = b""

        valid_fun_names_b = [f.name.encode('utf-8') for f in fun_defs]

        self.pipeline: list[TokenMatcher] = [ChoiceMatcher(valid_fun_names_b)]
        self.current_matcher_idx = 0

    @property
    def is_finished(self) -> bool:
        return self.current_matcher_idx >= len(self.pipeline)

    @property
    def _current_matcher(self) -> TokenMatcher | None:
        if self.is_finished:
            return None
        return self.pipeline[self.current_matcher_idx]

    @property
    def state_label(self) -> str:
        if self.is_finished:
            return "FINISHED"
        return type(self._current_matcher).__name__

    def _process_buffer(self) -> None:
        """Advance pipeline as long as buffer satisfies matcher"""
        while not self.is_finished:
            matcher = self._current_matcher
            buf = self.current_buffer_b
            if matcher and matcher.is_complete(buf):
                matcher.commit(buf)
                self._advance_pipeline(matcher, buf)
                continue
            if isinstance(matcher, ParamMatcher) and matcher and matcher.segment_complete(buf):
                leftover = matcher.commit(buf)
                if matcher.is_done:
                    self.current_matcher_idx += 1
                    self.current_buffer_b = leftover if leftover else b""
                    continue
                self.current_buffer_b = leftover if leftover else b""
                return
            return

    def evaluate_token(self, token_str: str) -> bool:
        """Simulate token insertion and return True if valide

        Evaluate byte sequence against current state
        """

        matcher = self._current_matcher
        if matcher is None:
            return False
        token_bytes = convert_token_str_to_bytes(token_str)
        test_buf = self.current_buffer_b + token_bytes
        return bool(matcher.evaluate(test_buf))

    def consume_token(self, token_str: str) -> None:
        """Apply token and update automata accordingly"""

        matcher = self._current_matcher
        if matcher is None:
            return
        self.current_buffer_b += token_str.encode('utf-8', errors='surrogateescape')
        self._process_buffer()

    def _advance_pipeline(self, completedMatcher: TokenMatcher, final_buf: bytes) -> None:
        """Dynamically adjust stack : can add ParamMatcher once function is selected"""

        self.current_buffer_b = b""

        if isinstance(completedMatcher, ChoiceMatcher):
            fun_name = final_buf.decode('utf-8', errors='surrogateescape')
            self.selected_function = next((f for f in self.fun_defs if f.name == fun_name))
            next_matchers: list[TokenMatcher] = [StaticSequenceMatcher(self.PARAM_PREFIX)]
            if self.selected_function and self.selected_function.parameters:
                next_matchers.append(ParamMatcher(self.selected_function, close_suffix=self.CLOSE_SUFFIX))
            next_matchers.append(StaticSequenceMatcher(self.CLOSE_SUFFIX))

            self.pipeline[self.current_matcher_idx + 1:self.current_matcher_idx + 1] = next_matchers

        self.current_matcher_idx += 1
