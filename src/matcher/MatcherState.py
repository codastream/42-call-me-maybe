from enum import Enum, auto


class MatcherState(Enum):
    START = auto()
    PROMPT = auto()
    EXPECT_FUN_NAME = auto()
    DONE_FUN_NAME = auto()
    EXPECT_PARAM_KEY = auto()
    EXPECT_PARAM_VAL = auto()
    EXPECT_COMMA_OR_END = auto()
    EXPECT_JSON_END = auto()
    FINISH = auto()
