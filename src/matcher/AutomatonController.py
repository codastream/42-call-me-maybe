
from src.matcher.TokenMatcher import TokenMatcher
from src.matcher.StaticSequenceMatcher import StaticSequenceMatcher
from src.matcher.ChoiceMatcher import ChoiceMatcher
from src.matcher.ValueMatcher import ValueMatcher
from src.matcher.AutomatonDef import AState, AUTOMATON
from src.models.FunctionDefinition import FunctionDefinition
from src.models.TypeDef import TypeDef
from src.utils.Trie import TrieNode
from src.config import get_logger


class AutomatonController:
    """In charge of evaluating tokens and managing state transitions"""
    PROMPT_PREFIX: bytes = b'{"prompt": "'
    NAME_PREFIX: bytes = b'", "name": "'
    PARAM_PREFIX: bytes = b'", "parameters": {'
    CLOSE_SUFFIX: bytes = b'}}'

    def __init__(self, fun_defs: list[FunctionDefinition], initial_prompt: bytes):
        """Initialize controller"""

        self.state = AState.FUN_NAME_VAL
        self.log = get_logger('match')
        self.fun_defs: list[FunctionDefinition] = fun_defs
        self.selected_function: FunctionDefinition | None = None
        self.evaluated_params: set[str] = set()
        self.selected_param_key: str | None = None
        self.pipeline: list[TokenMatcher] = []
        self.current_buffer_b: bytes = b""
        self._push_next()

    @property
    def state_label(self) -> str:
        """Return human readable label"""
        return str(AUTOMATON[self.state].label)

    @property
    def is_finished(self) -> bool:
        """Return True if automaton reached FINISH state"""
        return bool(self.state == AState.FINISH)

    @property
    def _top(self) -> TokenMatcher | None:
        """Return first matcher of the stack"""
        return self.pipeline[0] if self.pipeline else None

    def _remaining_keys(self) -> list[str]:
        """Return remaing param keys to evaluate"""
        if self.selected_function:
            return [k for k in self.selected_function.parameters if k not in self.evaluated_params]
        return []

    def _get_next_possible_key_matchers(self) -> list[bytes]:
        """Return next possible keys"""
        targets = []
        for k in self._remaining_keys():
            if k not in self.evaluated_params:
                targets.extend([
                    f'"{k}": '.encode(),
                    f' "{k}": '.encode(),
                    f', "{k}": '.encode(),
                    f',"{k}": '.encode(),
                ])
        return targets

    def _build_matcher_for_state(self, state: AState) -> TokenMatcher | None:
        """return a matcher according to state"""
        next_matcher: TokenMatcher | None = None
        match self.state:
            case AState.FUN_NAME_VAL:
                names = [f.name.encode() for f in self.fun_defs]
                next_matcher = ChoiceMatcher(names)
            case AState.PARAMS_OBJ_KEY:
                next_matcher = StaticSequenceMatcher(b'", "parameters": {')
            case AState.PARAM_KEY:
                targets = self._get_next_possible_key_matchers()
                next_matcher = ChoiceMatcher(targets)
            case AState.PARAM_VAL:
                if self.selected_function and self.selected_param_key:
                    p_type = self.selected_function.parameters[self.selected_param_key].type
                    next_matcher = ValueMatcher(p_type)
            case AState.CLOSE:
                next_matcher = ChoiceMatcher([b'}}', b' }}', b'}', b' }'])
        return next_matcher

    def _push_next(self) -> None:
        """Decide which matcher should go on stack"""
        next_matcher = self._build_matcher_for_state(self.state)
        if next_matcher is not None:
            self.pipeline.insert(0, next_matcher)

    def _advance_state(self, leftover: bytes = b"") -> None:
        """Transition following a fixed or dynamic order"""
        if not self.state:
            return
        if self.state == AState.PARAM_VAL:
            if self.selected_param_key:
                self.evaluated_params.add(self.selected_param_key)
            self.selected_param_key = None
            self.state = AState.PARAM_KEY if self._remaining_keys() else AState.CLOSE
        else:
            next_state = AUTOMATON[self.state].next
            if next_state is None:
                return
            self.state = next_state

        leftover_stripped = leftover.strip(b' \t\r\n')
        if self.state == AState.CLOSE:
            if leftover_stripped in (b'}}', b' }}'):
                self.log.debug("Close state matcher satisfied with leftover")
                self.state = AState.FINISH
                self.current_buffer_b = b""
                return

        self.current_buffer_b = leftover
        if self.state != AState.FINISH:
            self._push_next()

    def _get_next_matcher_after_value(self) -> TokenMatcher:
        """Peek next possible matcher"""
        next_targets = [k for k in self._remaining_keys() if k != self.selected_param_key]
        if next_targets:
            targets = self._get_next_possible_key_matchers()
            return ChoiceMatcher(targets)
        else:
            return ChoiceMatcher([b'}}', b' }}', b'}', b' }'])

    def _get_next_matcher(self) -> TokenMatcher | None:
        """Peek dynamical transition if ValueMatcher else return next"""
        if self.state == AState.PARAM_VAL:
            return self._get_next_matcher_after_value()
        next_state = AUTOMATON[self.state].next
        if next_state is None:
            return None
        return self._build_matcher_for_state(next_state)

    def get_current_parameter_type(self) -> TypeDef | None:
        """Return current evaluated type if current matcher is ValueMatcher"""
        if (self.state == AState.PARAM_VAL and self.selected_function and self.selected_param_key):
            return self.selected_function.parameters[self.selected_param_key].type
        return None

    def evaluate_tokens(self, tokenid_to_bytes: dict[int, bytes], trie_root: TrieNode,
                        value_buckets: dict[TypeDef, list[int]]) -> list[int]:
        """Prefilter and evaluate token against current state"""
        top = self._top
        if top is None:
            return []

        candidates_ids = top.prefilter_candidates(
            self.current_buffer_b,
            token_id_to_bytes=tokenid_to_bytes,
            trie_root=trie_root,
            value_buckets=value_buckets
        )

        valid_t_ids = []
        for t_id in candidates_ids:
            t_b = tokenid_to_bytes[t_id]
            if self.evaluate_token_bytes(t_b):
                valid_t_ids.append(t_id)
        return valid_t_ids

    def evaluate_token_bytes(self, token_b: bytes) -> bool:
        """Return current matcher evaluation of token bytes"""

        top = self._top
        if top is None:
            return False

        combined_buf = self.current_buffer_b + token_b
        if top.evaluate(combined_buf):
            return True

        if top.is_complete(self.current_buffer_b):
            leftover = top.leftover_bytes(combined_buf)
            target = leftover if leftover else token_b
            next_matcher = self._get_next_matcher()
            if next_matcher and next_matcher.evaluate(target):
                self.log.debug('evaluate bytes - ')
                return True

        if top.is_complete(combined_buf):
            leftover = top.leftover_bytes(combined_buf)
            if leftover:
                next_matcher = self._get_next_matcher()
                if next_matcher and next_matcher.evaluate(leftover):
                    return True

        return False

    def consume_token_bytes(self, token_b: bytes) -> None:
        """Add token bytes to buffer"""

        pending_bytes = token_b

        while not self.is_finished:
            top = self._top
            if top is None:
                return

            combined_buf = self.current_buffer_b + pending_bytes
            self.log.debug(f"""consume :\tstate = {self.state_label}
            \tself.current_buffer_b = {self.current_buffer_b!r}
            \tpending_bytes = {pending_bytes!r}
            """)
            # ex cases: buf b'3' + token b'4'
            if top.evaluate(combined_buf):
                self.current_buffer_b = combined_buf
                top.commit(self.current_buffer_b)
                if self.state == AState.FUN_NAME_VAL:
                    fun_name = self.current_buffer_b.decode(errors='surrogateescape')
                    matched = next((f for f in self.fun_defs if f.name == fun_name), None)
                    if matched is not None:
                        self.selected_function = matched
                        self.log.debug(f"consume : selected function = {self.selected_function}")
                    else:
                        if top.is_complete(self.current_buffer_b):
                            raise ValueError(f"generated function name {fun_name!r} matches no known function")
                elif self.state == AState.PARAM_KEY:
                    parts = [p for p in self.current_buffer_b.decode(
                        errors="surrogateescape").split('"') if p.strip() and p.strip() != ',']
                    self.selected_param_key = parts[0] if parts else None
                    self.log.debug(f"consume:\tselected param = {self.selected_param_key}")

                if top.is_complete(self.current_buffer_b) and not isinstance(top, ValueMatcher):
                    leftover = top.leftover_bytes(self.current_buffer_b)
                    self.pipeline.pop(0)
                    self._advance_state(leftover=leftover)
                    if leftover:
                        pending_bytes = b""
                        continue
                break

            # case buf '34' + token '}' - way to detect end of ValueMatcher
            elif top.is_complete(self.current_buffer_b):
                self.current_buffer_b = combined_buf
                top.commit(self.current_buffer_b)
                self.log.debug(f"consume:\t{top.__class__.__name__} completed. Advancing state")

                self.pipeline.pop(0)
                self._advance_state(leftover=b"")
                continue

            # case buf '3' + token '4}' - another way to detect end of ValueMatcher
            elif top.is_complete(combined_buf):
                leftover = top.leftover_bytes(combined_buf)
                consumed_len = len(leftover) if leftover else len(combined_buf)
                consumed_part = combined_buf[:consumed_len]
                self.current_buffer_b = consumed_part
                top.commit(self.current_buffer_b)
                self.log.debug(
                    f"""consume:\t{top.__class__.__name__} completed.
                    consumed: {consumed_part!r}
                    leftover: {leftover!r}.
                    Advancing state""")
                self.pipeline.pop(0)
                self._advance_state(leftover=leftover)
                pending_bytes = b""
                continue

            # should not happen
            else:
                self.log.error(f"consume:\tunable to consume {pending_bytes!r} in state {self.state_label}")
                break
