from typing import Any

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict

from src.matcher.TokenMatcher import TokenMatcher
from src.models.DecodingContext import DecodingContext
from src.models.DecodingMetrics import DecodingMetrics


class DecodingStepState(BaseModel):
    """Decoding state information required by debugging dashboard

    Args:
        BaseModel (pydantic base model)
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    context: DecodingContext
    metrics: DecodingMetrics
    logits: npt.NDArray[np.int64]
    filtered_logits: npt.NDArray[np.int64]
    authorized_token_ids: list[int]
    next_token_id: int
    readable_chunk: str
    generated_text: str
    current_stage: str
    top_tokens_data: list[dict[str, Any]]
    active_pipeline: list[TokenMatcher]
