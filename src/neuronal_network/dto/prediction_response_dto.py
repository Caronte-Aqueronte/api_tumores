from pydantic import BaseModel


class PredictionResponseDto(BaseModel):

    x_feature: float
    y_feature: float
    prediction: int
