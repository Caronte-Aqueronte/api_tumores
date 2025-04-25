

from typing import Dict, List, Tuple

from fastapi import HTTPException, status
from numpy import ndarray
from neuronal_network.dto.model_result_response_dto import ModelResultResponseDto
from neuronal_network.dto.prediction_response_dto import PredictionResponseDto
from neuronal_network.dto.train_request_dto import TrainRequestDTO
from neuronal_network.dto.entry_request_dto import EntryRequestDTO
from neuronal_network.dto.train_response_dto import TrainResponseDTO
from neuronal_network.neuronal.neuronal_network import NeuronalNetwork
from neuronal_network.utils.data_handler import DataHandler
from neuronal_network.utils.entry_list_converter import EntryConverter


class NeuronalNetworkService:

    def __init__(self):
        self.__neuronal_network: NeuronalNetwork = None
        self.__entry_converter = EntryConverter()

    def get_data_for_plot(self) -> Tuple[ndarray, ndarray]:
        data_handler = DataHandler(0, 0, 0, None)
        return data_handler.get_data_for_plot()

    def train_neuronal_network(self, train_request_dto: TrainRequestDTO) -> TrainResponseDTO:

        # iniciamos la red neuronal
        self.__neuronal_network = NeuronalNetwork(
            train_request_dto.max_epoachs,
            train_request_dto.learning_rate,
            train_request_dto.first_feature,
            train_request_dto.second_feature,
            train_request_dto.percentage_to_use
        )

        error_per_epoach, final_acurrancy, desicion_boundary_points = self.__neuronal_network.train()

        # lo entrenamos y obtenemos la respuesta (el dic para que el front pueda gradicar el error por epoca)
        return TrainResponseDTO(error_per_apoach=error_per_epoach, final_acurrancy=final_acurrancy, desicion_boundary_points=desicion_boundary_points)

    def predict(self, inputs: List[EntryRequestDTO]) -> ModelResultResponseDto:

        if (self.__neuronal_network is None):
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="La red no está entrenada aún.")

        # mandamos a predecir las entradas que nos mandaron
        inputs_after, binary_predictions = self.__neuronal_network.predict(
            inputs)

        # con los inputs y las predicciones binarias las convertimos en un formato 1 a 1 (cada input tendra su preduccion)
        predicion_responses: List[PredictionResponseDto] = self.__entry_converter.convert_entries_to_predictions_response_dto(
            inputs_after, binary_predictions)

        return ModelResultResponseDto(predictions=predicion_responses)
