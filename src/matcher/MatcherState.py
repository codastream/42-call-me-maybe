from enum import Enum, auto


class MatcherState(Enum):
    # START = auto()
    # EXPECT_PROMPT = auto()
    EXPECT_STATIC_FUN_NAME_KEY = auto()
    EXPECT_DYNAMIC_FUN_NAME = auto()
    EXPECT_STATIC_PARAMETERS_OBJECT_KEY = auto()
    EXPECT_DYNAMIC_PARAM_KEY = auto()
    EXPECT_VALUE_PARAM_VAL = auto()
    # EXPECT_COMMA_OR_END = auto()
    EXPECT_STATIC_JSON_END = auto()
    FINISH = auto()
