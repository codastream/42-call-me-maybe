from enum import Enum, auto
from dataclasses import dataclass


class AState(Enum):
    FUN_NAME_VAL = auto()
    PARAMS_OBJ_KEY = auto()
    PARAM_KEY = auto()
    PARAM_VAL = auto()
    CLOSE = auto()
    FINISH = auto()


@dataclass
class Transition:
    label: str
    next: "AState | None"


AUTOMATON: dict[AState, Transition] = {
    AState.FUN_NAME_VAL: Transition("Function name (val)", AState.PARAMS_OBJ_KEY),
    AState.PARAMS_OBJ_KEY: Transition("Function name (key)", AState.PARAM_KEY),
    AState.PARAM_KEY: Transition("Param (key)", AState.PARAM_VAL),
    AState.PARAM_VAL: Transition("Param (key)", None),
    AState.CLOSE: Transition("Close", AState.FINISH),
    AState.FINISH: Transition("Finish", None),
}
