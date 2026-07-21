
# from src.matcher.MatcherState import MatcherState
from src.matcher.TokenMatcher import TokenMatcher, StaticSequenceMatcher, ChoiceMatcher, ValueMatcher
from src.matcher.AutomatonDef import AState, AUTOMATON
from src.models.FunctionDefinition import FunctionDefinition
from src.models.TypeDef import TypeDef

import logging


class AutomatonController:
    PROMPT_PREFIX: bytes = b'{"prompt": "'
    NAME_PREFIX: bytes = b'", "name": "'
    PARAM_PREFIX: bytes = b'", "parameters": {'
    CLOSE_SUFFIX: bytes = b'}}'

    log = logging.getLogger('match')

    def __init__(self, fun_defs: list[FunctionDefinition], initial_prompt: bytes):

        self.state = AState.FUN_NAME_VAL
        self.fun_defs: list[FunctionDefinition] = fun_defs
        self.selected_function: FunctionDefinition = None
        self.evaluated_params: set[str] = set()
        self.selected_param_key: str | None = None
        self.pipeline: list[TokenMatcher] = []
        self.current_buffer_b: bytes = b""
        self._push_next() 

    @property
    def state_label(self) -> str:
        return AUTOMATON[self.state].label

    @property
    def is_finished(self) -> bool:
        return self.state == AState.FINISH

    @property
    def _top(self) -> TokenMatcher | None:
        return self.pipeline[0] if self.pipeline else None


    def _remaining_keys(self) -> list[str]:
        return [k for k in self.selected_function.parameters if k not in self.evaluated_params]

    def _push_next(self) -> None:
        """Decide which matcher should go on stack"""
        match self.state:
            case AState.FUN_NAME_VAL:
                names = [f.name.encode() for f in self.fun_defs]
                self.pipeline.insert(0, ChoiceMatcher(names))
            case AState.PARAMS_OBJ_KEY:
                self.pipeline.insert(0, StaticSequenceMatcher(b'", "parameters": {'))
            case AState.PARAM_KEY:
                prefix = b', ' if self.evaluated_params else b''
                targets = [prefix + f'"{k}": '.encode() for k in self._remaining_keys()]
                self.pipeline.insert(0, ChoiceMatcher(targets))
            case AState.PARAM_VAL:
                p_type = self.selected_function.parameters[self.selected_param_key].type
                self.pipeline.insert(0, ValueMatcher(p_type))
            case AState.CLOSE:
                self.pipeline.insert(0, StaticSequenceMatcher(b'}}'))

    def _advance_state(self) -> None:
        """Transition following a fixed or dynamic order"""
        if self.state == AState.PARAM_VAL:
            self.evaluated_params.add(self.selected_param_key)
            self.selected_param_key = None
            self.state = AState.PARAM_KEY if self._remaining_keys() else AState.CLOSE
        else:
            self.state = AUTOMATON[self.state].next
        if self.state != AState.FINISH:
            self._push_next()

    def _get_next_matcher_after_values(self) -> TokenMatcher:
        if self._remaining_keys():
            targets = [f', "{k}": '.encode() for k in self._remaining_keys()]
            return ChoiceMatcher(targets)
        else:
            return StaticSequenceMatcher(b"}}")

    def _quick_prefilter_at_state_change(self, token_b: bytes) -> bool:
        top = self._top

        if not isinstance(top, ValueMatcher):
            return True
        if self.current_buffer_b:
            return True

        if top.type == TypeDef.STRING:
          if not self.current_buffer_b:
              return token_b.startswith(b'"')
          return True
        
        if top.type == TypeDef.NUMBER:
            if not self.current_buffer_b:
                return token_b[:1] in (b'0', b'1', b'2', b'3',b'4',b'5',b'6',b'7',b'8',b'9',b'-',b'+',b'.')
            return True
        
        if top.type == TypeDef.BOOLEAN:
            if not self.current_buffer_b:
                return token_b[:1] in (b't', b'f')
            return True
        return True


    def evaluate_tokens(self, tokenid_to_bytes: dict[int, bytes]) -> list[int]:
        """Prefilter and evaluate token against current state"""
        top = self._top
        if top is None:
            return

        valid_t_ids = []
        for (t_id, t_b) in tokenid_to_bytes.items():
          # if not self._quick_prefilter_at_state_change(t_b):
          #   continue
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

        # lookahead
        if isinstance(top, ValueMatcher) and top.is_unambiguous_terminal(combined_buf):
            leftover = top.leftover_bytes(combined_buf)
            # self.log.debug(f"evaluate_token_bytes for ValueMatcher leftover={leftover!r}")
            if leftover:
                next_matcher = self._get_next_matcher_after_values()
                if next_matcher and next_matcher.evaluate(leftover):
                    return True
        
        return False

    def consume_token_bytes(self, token_b: bytes) -> None:
        """Add token bytes to buffer"""
        log = logging.getLogger('match')

        self.current_buffer_b += token_b
        
        while not self.is_finished:
          top = self._top
          if top is None:
              return

          if not top.is_complete(self.current_buffer_b):
              break

          top.commit(self.current_buffer_b)

          log.debug(f"consume : top={top.display_name()} buf={self.current_buffer_b!r} is_complete=True")

          leftover = b""
          if hasattr(top, "leftover_bytes"):
              leftover = top.leftover_bytes(self.current_buffer_b)
              log.debug(f"consume : leftover = {leftover}")
          
          if self.state == AState.FUN_NAME_VAL:
              fun_name = self.current_buffer_b.decode(errors='surrogateescape')
              self.selected_function = next(f for f in self.fun_defs if f.name == fun_name)
          elif self.state == AState.PARAM_KEY:
              self.selected_param_key = self.current_buffer_b.decode().split('"')[1]
              log.debug(f"consume : selected param = {self.selected_param_key}")
          self.pipeline.pop(0)
          self.current_buffer_b = leftover
          self._advance_state()
