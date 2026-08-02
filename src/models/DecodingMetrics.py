
from pydantic import BaseModel, Field


class DecodingMetrics(BaseModel):
    """Session metrics

    Args:
        BaseModel (pydantic base model)
    """
    loops: int = 0
    top1_rejected_count: int = 0
    selected_ranks: list[int] = Field(default_factory=list)
    selected_logits: list[float] = Field(default_factory=list)

    @property
    def top1_rejected_pct(self) -> float:
        return self.top1_rejected_count / self.loops if self.loops > 0 else 0.0

    @property
    def avg_rank(self) -> float:
        return sum(self.selected_ranks) / len(self.selected_ranks) if self.selected_ranks else 0.0

    @property
    def avg_logit(self) -> float:
        return sum(self.selected_logits) / len(self.selected_logits) if self.selected_logits else 0.0
