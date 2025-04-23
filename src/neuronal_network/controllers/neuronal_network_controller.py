from typing import Dict, List
from fastapi import APIRouter

from neuronal_network.dto.model_result_response_dto import ModelResultResponseDto
from neuronal_network.dto.train_request_dto import TrainRequestDTO
from neuronal_network.dto.entry_request_dto import EntryRequestDTO
from neuronal_network.services.neuronal_network_service import NeuronalNetworkService


router = APIRouter(
    prefix="/neuronal-network"
)


neuronal_network_service = NeuronalNetworkService()


@router.post("/train", response_model=Dict[int, float])
def train_model(train_request_dto: TrainRequestDTO) -> Dict[int, float]:
    return neuronal_network_service.train_neuronal_network(train_request_dto)


@router.post("/predict", response_model=ModelResultResponseDto)
def predict(inputs: List[EntryRequestDTO]) -> ModelResultResponseDto:
    return neuronal_network_service.predict(inputs)
