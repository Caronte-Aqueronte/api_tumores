

from typing import Dict, List, Tuple

from fastapi import HTTPException, status
from numpy import ndarray
from src.neuronal_network.dto.model_result_response_dto import PredictResponseDto
from src.neuronal_network.dto.prediction_response_dto import PredictionResponseDto
from src.neuronal_network.dto.train_request_dto import TrainRequestDTO
from src.neuronal_network.dto.predict_request_dto import PredictRequestDTO
from src.neuronal_network.dto.train_response_dto import TrainResponseDTO
from src.neuronal_network.neuronal.neuronal_network import NeuronalNetwork
from src.neuronal_network.utils.data_handler import DataHandler


class NeuronalNetworkService:

    def __init__(self):
        self.__neuronal_network: NeuronalNetwork = None

    def get_data_for_plot(self) -> Tuple[ndarray, ndarray]:
        data_handler = DataHandler(0, 0, 0, None)
        return data_handler.get_data_for_plot()

    def train_neuronal_network(self, train_request_dto: TrainRequestDTO) -> TrainResponseDTO:

        # iniciamos la red neuronal
        self.__neuronal_network = NeuronalNetwork(
            train_request_dto.max_epochs,
            train_request_dto.learning_rate,
            train_request_dto.first_feature,
            train_request_dto.second_feature,
            train_request_dto.percentage_to_use
        )

        error_per_epoch, final_accuracy, desicion_boundary_points = self.__neuronal_network.train()

        # lo entrenamos y obtenemos la respuesta (el dic para que el front pueda gradicar el error por epoca)
        return TrainResponseDTO(error_per_epoch=error_per_epoch, final_accuracy=final_accuracy, desicion_boundary_points=desicion_boundary_points)

    def predict(self, input: PredictRequestDTO) -> PredictResponseDto:

        if (self.__neuronal_network is None):
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="La red no está entrenada aún.")

        # mandamos a predecir las entradas que nos mandaron
        prediction = self.__neuronal_network.predict(
            input)

        return PredictResponseDto(prediction=prediction)
