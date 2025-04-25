from typing import Dict, List, Tuple
from fastapi import APIRouter
from numpy import ndarray

from neuronal_network.dto.model_result_response_dto import PredictResponseDto
from neuronal_network.dto.prediction_response_dto import PredictionResponseDto
from neuronal_network.dto.train_request_dto import TrainRequestDTO
from neuronal_network.dto.predict_request_dto import PredictRequestDTO
from neuronal_network.dto.train_response_dto import TrainResponseDTO
from neuronal_network.services.neuronal_network_service import NeuronalNetworkService


router = APIRouter(
    prefix="/neuronal-network"
)


neuronal_network_service = NeuronalNetworkService()


@router.get("/", response_model=Tuple[List[List[float]], List[int]])
def get_data_for_plot() -> Tuple[List[List[float]], List[int]]:
    features, labels = neuronal_network_service.get_data_for_plot()
    return features.tolist(), labels.tolist()


@router.post("/train", response_model=TrainResponseDTO)
def train_model(train_request_dto: TrainRequestDTO) -> TrainResponseDTO:
    return neuronal_network_service.train_neuronal_network(train_request_dto)


@router.post("/predict", response_model=PredictResponseDto)
def predict(inputs: PredictRequestDTO) -> PredictResponseDto:
    return neuronal_network_service.predict(inputs)
