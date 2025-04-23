from typing import List

from pydantic import BaseModel

from neuronal_network.dto.prediction_response_dto import PredictionResponseDto


class ModelResultResponseDto(BaseModel):

    predictions: List[PredictionResponseDto]
