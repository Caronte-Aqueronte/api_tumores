
from pydantic import BaseModel, Field


class PredictRequestDTO(BaseModel):
    first_feature: float = Field(..., ge=0)
    second_feature: float = Field(..., ge=0)
