

from typing import Dict
from neuronal_network.dto.train_request_dto import TrainRequestDTO
from neuronal_network.neuronal.neuronal_network import NeuronalNetwork


class NeuronalNetworkService:

    def __init__(self):
        self.__neuronal_service: NeuronalNetwork = None

    def train_neuronal_network(self, train_request_dto: TrainRequestDTO) -> Dict[int, float]:
        # iniciamos la red neuronal
        self.__neuronal_service = NeuronalNetwork(
            train_request_dto.max_epoachs,
            train_request_dto.learning_rate,
            train_request_dto.first_feature,
            train_request_dto.second_feature,
            train_request_dto.percentage_to_use
        )

        # lo entrenamos y obtenemos la respuesta (el dic para que el front pueda gradicar el error por epoca)
        return self.__neuronal_service.train()
