

from src.matcher.TokenMatcher import TokenMatcher
from src.matcher.StaticSequenceMatcher import StaticSequenceMatcher
from src.matcher.ChoiceMatcher import ChoiceMatcher
from src.matcher.ValueMatcher import ValueMatcher
from src.matcher.AutomatonDef import AState, AUTOMATON
from src.models.FunctionDefinition import FunctionDefinition
from src.models.TypeDef import TypeDef
from src.utils.Trie import TrieNode
from src.utils.Profiler import profiler
from src.config import get_logger


log = get_logger()


class AutomatonController:
    """Evaluate tokens and manage state transitions

    Attributes:
        PROMPT_PREFIX (bytes): Prefix for function call prompt.
        NAME_PREFIX (bytes): Separator for function name.
        PARAM_PREFIX (bytes): Separator for function parameters.
        CLOSE_SUFFIX (bytes): Closing suffix for JSON function call.
    """

    PROMPT_PREFIX: bytes = b'{"prompt": "'
    NAME_PREFIX: bytes = b'", "name": "'
    PARAM_PREFIX: bytes = b'", "parameters": {'
    CLOSE_SUFFIX: bytes = b'}}'

    def __init__(self, fun_defs: list[FunctionDefinition], value_buckets: dict[TypeDef, set[int]]):
        """Initialize controller

        Args:
            fun_defs (list[FunctionDefinition]): List of valid function definitions.
            value_buckets (dict[TypeDef, set[int]]): Token IDs pre-categorized by type.
        """

        self.state = AState.FUN_NAME_VAL
        self.log = get_logger('match')
        self.fun_defs: list[FunctionDefinition] = fun_defs
        self.selected_function: FunctionDefinition | None = None
        self.evaluated_params: set[str] = set()
        self.selected_param_key: str | None = None
        self.pipeline: list[TokenMatcher] = []
        self.current_buffer_b: bytes = b""
        self.value_buckets: dict[TypeDef, set[int]] = value_buckets
        self._push_next()

    @property
    def state_label(self) -> str:
        """Returns a human-readable label of the current automaton state.

        Returns:
            str: Name/label of the current state.
        """
        return str(AUTOMATON[self.state].label)

    @property
    def is_finished(self) -> bool:
        """Checks if the generation cycle has completed.

        Returns:
            bool: True if the automaton reached the FINISH state, False otherwise.
        """
        return bool(self.state == AState.FINISH)

    @property
    def _top(self) -> TokenMatcher | None:
        """Gets the active TokenMatcher at the top of the stack.

        Returns:
            TokenMatcher | None: The current matcher, or None if the stack is empty.
        """
        return self.pipeline[0] if self.pipeline else None

    def _remaining_keys(self) -> list[str]:
        """Gets the list of parameter keys that still need to be evaluated.

        Returns:
            list[str]: Unprocessed parameter names for the selected function.
        """
        if self.selected_function:
            return [
                k for k in self.selected_function.parameters
                if k not in self.evaluated_params
                and k != self.selected_param_key
            ]
        return []

    def _get_next_possible_key_matchers(self) -> list[bytes]:
        """Return next possible keys

        Returns:
            list[bytes]: list of possible targets, including variants with extra spaces
        """
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
        """Instantiates an appropriate TokenMatcher for a given state.

        Args:
            state (AState): Automaton state to build a matcher for.

        Returns:
            TokenMatcher | None: Concrete matcher corresponding to the state, or None.
        """
        matcher: TokenMatcher | None = None
        match state:
            case AState.FUN_NAME_VAL:
                names = [f.name.encode() for f in self.fun_defs]
                matcher = ChoiceMatcher(names)
            case AState.EMPTY_PARAMS_AND_CLOSE:
                matcher = StaticSequenceMatcher(b'", "parameters": {}}')
            case AState.PARAMS_OBJ_KEY:
                matcher = StaticSequenceMatcher(b'", "parameters": {')
            case AState.PARAM_KEY:
                targets = self._get_next_possible_key_matchers()
                matcher = ChoiceMatcher(targets)
            case AState.PARAM_VAL:
                if self.selected_function and self.selected_param_key:
                    p_type = self.selected_function.parameters[self.selected_param_key].type
                    matcher = ValueMatcher(p_type, self.value_buckets)
            case AState.CLOSE:
                matcher = ChoiceMatcher([b'}}', b' }}'])
        return matcher

    def _push_next(self) -> None:
        """Build and push next token matcher onto the stack"""
        next_matcher = self._build_matcher_for_state(self.state)
        if next_matcher is not None:
            self.pipeline.insert(0, next_matcher)

    def _advance_state(self, leftover: bytes = b"") -> None:
        """Transition following a fixed or dynamic order

        Note:
            - Dynamic transitions are from FUN_NAME_VAL and PARAM_VAL
            - If state is CLOSE and leftover bytes satisfies this state,
            automatically advance to FINISH
        """
        if not self.state:
            return
        if self.state == AState.PARAM_VAL:
            if self.selected_param_key:
                self.evaluated_params.add(self.selected_param_key)
            self.selected_param_key = None
            self.state = AState.PARAM_KEY if self._remaining_keys() else AState.CLOSE
        elif self.state == AState.FUN_NAME_VAL:
            if self.selected_function and self.selected_function.parameters:
                self.state = AState.PARAMS_OBJ_KEY
            else:
                self.state = AState.EMPTY_PARAMS_AND_CLOSE
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
            self.log.debug(
                f"""advanced to :\t{self.state.name}""")
        self.current_buffer_b = leftover
        if self.state != AState.FINISH:
            self._push_next()

    def _get_next_matcher_after_value(self) -> TokenMatcher | None:
        """Peek next possible matcher for dynamic transition from PARAM_VAL

        Returns:
            TokenMatcher: ChoiceMatcher for PARAM_KEY state if there is still one parameter left
            TokenMatcher: StaticSequenceMatcher for CLOSE state if no parameter left
        """
        if len(self._remaining_keys()) > 0:
            return self._build_matcher_for_state(AState.PARAM_KEY)
        else:
            return self._build_matcher_for_state(AState.CLOSE)

    def _get_next_matcher_after_fun_name(self) -> TokenMatcher | None:
        """Peek next possible matcher for dynamic transition from function name

        Returns:
            TokenMatcher: StaticSequenceMatcher till either first param key or end of json
        """
        if self.selected_function and not self.selected_function.parameters:
            return self._build_matcher_for_state(AState.EMPTY_PARAMS_AND_CLOSE)
        else:
            return self._build_matcher_for_state(AState.PARAMS_OBJ_KEY)

    def _get_next_matcher(self) -> TokenMatcher | None:
        """Peeks the next expected TokenMatcher across dynamic or static transitions.

        Returns:
            TokenMatcher | None: The anticipated TokenMatcher for the subsequent state, or None.
        """
        if self.state == AState.PARAM_VAL:
            return self._get_next_matcher_after_value()
        next_state = AUTOMATON[self.state].next
        if self.state == AState.FUN_NAME_VAL:
            return self._get_next_matcher_after_fun_name()
        if next_state is None:
            return None
        return self._build_matcher_for_state(next_state)

    def get_current_parameter_type(self) -> TypeDef | None:
        """Returns the type definition of the parameter currently being evaluated.

        Returns:
            TypeDef | None: Expected type definition if evaluating a parameter value, None otherwise.
        """
        if (self.state == AState.PARAM_VAL and self.selected_function and self.selected_param_key):
            return self.selected_function.parameters[self.selected_param_key].type
        return None

    @profiler.decorate("AutomatonController.evaluate_tokens")
    def evaluate_tokens(self, tokenid_to_bytes: dict[int, bytes], trie_root: TrieNode,
                        value_buckets: dict[TypeDef, set[int]]) -> list[int]:
        """Prefilter and evaluate token against current state

        Args:
            tokenid_to_bytes (dict[int, bytes]): mapping of token id to bytes
            trie_root (TrieNode): Trie for model vocabulary
            value_buckets (dict[TypeDef, set[int]]): filtered token ids by value type

        Note:
            enable transition from PARAM_VAL by authorizing tokens prefixed by `,` or `}`
            according to presence of remaining keys

        Returns:
            list[int]: eligible tokens for state
        """
        log.debug("[blue]evaluate_tokens[/blue]")
        top = self._top
        if top is None:
            return []

        try:
            with profiler.track("evaluate_tokens#prefilter_candidates"):
                candidates_ids = top.prefilter_candidates(
                    self.current_buffer_b,
                    token_id_to_bytes=tokenid_to_bytes,
                    trie_root=trie_root,
                    value_buckets=value_buckets
                )
                has_remaining_keys: bool = len(self._remaining_keys()) > 0
                if isinstance(top, ValueMatcher):
                    log.debug(f"len of prefiltered for value matcher is {len(candidates_ids)}")
                    for t_id, t_b in tokenid_to_bytes.items():
                        stripped_b = t_b.strip()
                        if has_remaining_keys and stripped_b.startswith(b','):
                            candidates_ids.add(t_id)
                        elif stripped_b.startswith(b'}'):
                            candidates_ids.add(t_id)

            valid_t_ids = []
            with profiler.track("evaluate_tokens#evaluate_token_bytes_loop"):
                for t_id in candidates_ids:
                    t_b = tokenid_to_bytes[t_id]
                    if self.evaluate_token_bytes(t_b):
                        valid_t_ids.append(t_id)

        except Exception as e:
            log.exception(e)

        return valid_t_ids

    def evaluate_token_bytes(self, token_b: bytes) -> bool:
        """Return current matcher evaluation of token bytes

        Args:
            token_b (bytes): token bytes

        Note:
            authorize tokens covering current state and next one

        Returns:
            bool: True if token is valid for current state (and potentially next)
        """

        top = self._top
        if top is None:
            return False

        combined_buf = self.current_buffer_b + token_b
        combined_buf = getattr(top, "_normalize", lambda b: b)(combined_buf)
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

    @profiler.decorate("AutomatonController.consume_token_bytes")
    def consume_token_bytes(self, token_b: bytes) -> None:
        """Consumes a token's bytes, updates internal buffers, and advances state if complete.

        Args:
            token_b (bytes): Accepted token byte payload to append and process.

        Raises:
            ValueError: If the generated function name does not match any known function.
        """

        pending_bytes = token_b
        log.debug("[blue]consume_token_bytes[/blue]")

        while not self.is_finished:
            top = self._top
            if top is None:
                return

            combined_buf = self.current_buffer_b + pending_bytes
            combined_buf = getattr(top, "_normalize", lambda b: b)(combined_buf)
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

            # case buf '3' + token '4}' - another way to detect end of ValueMatcher
            elif top.is_complete(combined_buf):
                leftover = top.leftover_bytes(combined_buf)
                consumed_len = len(leftover) if leftover else len(combined_buf)
                consumed_part = combined_buf[:consumed_len]
                self.current_buffer_b = consumed_part
                top.commit(self.current_buffer_b)
                self.log.debug(
                    f"""consume:\t{top.__class__.__name__} completed.
                    Committing current buffer + part of pending bytes
                    consumed: {consumed_part!r}
                    Advancing state
                    leftover transferred as buffer of next state : {leftover!r}.
                    """)
                self.pipeline.pop(0)
                self._advance_state(leftover=leftover)
                pending_bytes = b""
                continue

            # case buf '34' + token '}' - way to detect end of ValueMatcher
            elif top.is_complete(self.current_buffer_b):
                # self.current_buffer_b = combined_buf
                top.commit(self.current_buffer_b)
                self.log.debug(
                    f"""consume:\t{top.__class__.__name__} completed.
                    Committing current buffer only. Ignoring pending bytes.
                    consumed: {self.current_buffer_b!r}
                    Advancing state
                    pending transferred as buffer of next state : {pending_bytes!r}.
                    """)
                self.pipeline.pop(0)
                self._advance_state(leftover=pending_bytes)
                continue

            # should not happen
            else:
                self.log.error(f"consume:\tunable to consume {pending_bytes!r} in state {self.state_label}")
                break
