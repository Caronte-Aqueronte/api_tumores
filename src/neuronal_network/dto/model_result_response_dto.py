from typing import List

from neuronal_network.dto.prediction_response_dto import PredicionResponseDto


class ModelResultResponseDto:

    def __init__(self, predictions: List[PredicionResponseDto], acurrancy_percentage: float):
        self.__predictions: List[PredicionResponseDto] = predictions
        self.__acurrancy_percentage: float = acurrancy_percentage
