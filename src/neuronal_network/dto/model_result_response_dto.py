from pydantic import BaseModel


class PredictResponseDto(BaseModel):

    prediction: str
