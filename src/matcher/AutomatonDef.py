from enum import Enum, auto
from dataclasses import dataclass


class AState(Enum):
    """Possible states of the automaton"""
    FUN_NAME_VAL = auto()
    EMPTY_PARAMS_AND_CLOSE = auto()
    PARAMS_OBJ_KEY = auto()
    PARAM_KEY = auto()
    PARAM_VAL = auto()
    CLOSE = auto()
    FINISH = auto()


@dataclass
class Transition:
    """Representation of current state label and next state

    Next state is None if multiple transitions are possible
    """
    label: str
    next: "AState | None"


AUTOMATON: dict[AState, Transition] = {
    AState.FUN_NAME_VAL: Transition("Function name (val)", None),
    AState.EMPTY_PARAMS_AND_CLOSE: Transition("No params", AState.FINISH),
    AState.PARAMS_OBJ_KEY: Transition("Params", AState.PARAM_KEY),
    AState.PARAM_KEY: Transition("Param (key)", AState.PARAM_VAL),
    AState.PARAM_VAL: Transition("Param (val)", None),
    AState.CLOSE: Transition("Close", AState.FINISH),
    AState.FINISH: Transition("Finish", None),
}
