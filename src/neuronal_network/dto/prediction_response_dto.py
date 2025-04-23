from pydantic import BaseModel


class PredicionResponseDto(BaseModel):

    x_feature: float
    y_feature: float
    prediction: int
