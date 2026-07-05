from src.MatcherState import MatcherState
from enum import Enum, auto
from src.models.FunctionDefinition import FunctionDefinition
from rich import print as rprint

class MatchType(Enum):
  NO_MATCH = auto()
  PARTIAL_MATCH = auto()
  COMPLETE_MATCH = auto()

def bytes_to_unicode() -> dict[int, str]:
  """Table associating byte with visible unicode character"""

  bs =  list(range(ord("!"), ord("~") + 1)) + \
        list(range(ord("¡"), ord("¬") + 1)) + \
        list(range(ord("®"), ord("ÿ") + 1))
  cs = bs[:]
  n = 0
  for b in range(256):
    if b not in bs:
      bs.append(b)
      cs.append(256 + n)
      n += 1
  cs_chars = [chr(c) for c in cs]
  return dict(zip(bs, cs_chars))

BYTE_ENCODER = bytes_to_unicode()
BYTE_DECODER = {v: k for k,v in BYTE_ENCODER.items()}

def token_to_real_bytes(token_str: str) -> bytes:
  """Convert a vocab token to real bytes"""
  return bytes(BYTE_DECODER[c] for c in token_str)

class JSONSchemaMatcher:
  PROMPT_PREFIX: bytes = b'{"prompt": "'
  NAME_PREFIX: bytes = b'", "name": "'
  PARAM_PREFIX: bytes = b'", "parameters": {'

  def __init__(self, fun_defs: list[FunctionDefinition], initial_prompt: bytes):
    self.fun_defs = fun_defs
    self.valid_fun_names = [f.name.encode('utf-8') for f in self.fun_defs]
    self.initial_prompt = initial_prompt
    self.state = MatcherState.START
    self.current_buffer = b""
    self.selected_function = None
    self.evaluated_parameters = set()
    self.current_param_key = None

  def _check_match_type(self, buf: str, target: str | None = None, allowed_targets: list[str] | None = None) -> MatchType:
    """Determine how current buffer correspond to target bytes"""
    
    if allowed_targets is not None:
      matched = [n for n in allowed_targets if n.startswith(buf)]
      if not matched:
        return MatchType.NO_MATCH
      if buf in allowed_targets:
        return MatchType.COMPLETE_MATCH
      return MatchType.PARTIAL_MATCH

    if target is not None:
      if buf == target:
        return MatchType.COMPLETE_MATCH
      if target.startswith(buf):
        return MatchType.PARTIAL_MATCH
      return MatchType.NO_MATCH
    
    return MatchType.NO_MATCH


  def _try_transition(self, match: MatchType, next_state: MatcherState) -> bool:
    """Handle linear transition according to match type."""

    if match == MatchType.COMPLETE_MATCH: 
      self.state = next_state
      self.current_buffer = b""
      return True
    return match == MatchType.PARTIAL_MATCH

  def evaluate_token(self, token_str: str) -> bool:
    """Simulate token insertion and return True if valid"""

    token_bytes = token_to_real_bytes(token_str)
    saved_state = (self.state, self.current_buffer, self.selected_function, self.current_param_key, set(self.evaluated_parameters))
    is_valid = True
    for byte_int in token_bytes:
      byte = bytes([byte_int])
      if not self._evaluate_char(byte):
        is_valid = False
        break
    self.state, self.current_buffer, self.selected_function, self.current_param_key, self.evaluated_parameters = saved_state
    return is_valid

  def consume_token(self, token_str:str) -> None:
    """Apply token and update automata accordingly"""
    token_bytes = token_str.encode('utf-8', errors='surrogateescape')
    for byte_int in token_bytes:
      self._evaluate_char(bytes([byte_int]))

  def _evaluate_char(self, char:bytes) -> bool:
    """Evaluate a character according to state"""

    self.current_buffer += char
    buf = self.current_buffer

    if self.state == MatcherState.START:
      match = self._check_match_type(buf, target=JSONSchemaMatcher.PROMPT_PREFIX)
      return self._try_transition(match, MatcherState.PROMPT)
    
    elif self.state == MatcherState.PROMPT:
      target = self.initial_prompt + JSONSchemaMatcher.NAME_PREFIX
      match = self._check_match_type(buf, target=target)
      return self._try_transition(match, MatcherState.EXPECT_FUN_NAME)
    
    elif self.state == MatcherState.EXPECT_FUN_NAME:
      match = self._check_match_type(buf, allowed_targets=self.valid_fun_names)
      if match == MatchType.COMPLETE_MATCH:
        buf_str = buf.decode('utf-8', errors='surrogateescape')
        self.selected_function = next((f for f in self.fun_defs if f.name == buf_str), None)
        self.state = MatcherState.DONE_FUN_NAME
        self.current_buffer = b""
        return True
      return match == MatchType.PARTIAL_MATCH
      # return self._try_transition(match, MatcherState.DONE_FUN_NAME)
    
    elif self.state == MatcherState.DONE_FUN_NAME:
      target = JSONSchemaMatcher.PARAM_PREFIX
      match = self._check_match_type(buf, target=target)
      return self._try_transition(match, MatcherState.EXPECT_PARAM_KEY)

    elif self.state == MatcherState.EXPECT_PARAM_KEY:
      if not self.selected_function or not self.selected_function.parameters:
        return False
      all_keys = self.selected_function.parameters.keys()
      allowed_targets = [f'"{k}": '.encode('utf-8') for k in all_keys if k not in self.evaluated_parameters]
      match = self._check_match_type(buf, allowed_targets=allowed_targets)
      if match == MatchType.COMPLETE_MATCH:
        buf_str = buf.decode('utf-8', errors='surrogateescape')
        self.current_param_key = buf_str.replace('"', '').replace(':', '').strip()
      return self._try_transition(match, MatcherState.EXPECT_PARAM_VAL)
    
    elif self.state == MatcherState.EXPECT_PARAM_VAL:
      if not self.selected_function or not self.current_param_key or not self.selected_function.parameters:
        return False
      param_field = self.selected_function.parameters[self.current_param_key]
      param_type = param_field.type
      buf_str = buf.decode('utf-8', errors='surrogateescape')
      char_str = char.decode('utf-8', errors='surrogateescape')
      
      all_keys = list(self.selected_function.parameters.keys())
      is_last_param = self.current_param_key == all_keys[-1]
      if is_last_param and char_str == ",":
        return False
      if not is_last_param and char_str == "}":
        return False
      
      is_valid = param_type._validate_buffer_type(buf_str, char_str)
      if is_valid and char_str in (",", "}"):
        self.evaluated_parameters.add(self.current_param_key)
        self.state = MatcherState.EXPECT_COMMA_OR_END
        self.current_buffer = char
      return is_valid
    
    elif self.state == MatcherState.EXPECT_COMMA_OR_END:
      if not self.selected_function or not self.selected_function.parameters:
        return False
      all_keys = self.selected_function.parameters.keys()
      has_more_params = len(self.evaluated_parameters) < len(all_keys)
      target = b", " if has_more_params else b"}}"
      match = self._check_match_type(buf, target=target)

      if match == MatchType.COMPLETE_MATCH:
        if has_more_params:
          self.state = MatcherState.EXPECT_PARAM_KEY
        else:
          self.state = MatcherState.FINISH
        self.current_buffer = b""
        return True
      return match == MatchType.PARTIAL_MATCH
    
    return False
      