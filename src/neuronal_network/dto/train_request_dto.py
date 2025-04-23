from pydantic import BaseModel, Field


class TrainRequestDTO(BaseModel):

    max_epoachs: int = Field(..., ge=1)
    learning_rate: float = Field(..., gt=0.0, le=100.0)
    first_feature: int = Field(..., ge=0, le=29)
    second_feature: int = Field(..., ge=0, le=29)
    percentage_to_use: int = Field(..., ge=1, le=100)
