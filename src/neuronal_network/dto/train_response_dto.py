
from typing import Dict
from pydantic import BaseModel

from neuronal_network.dto.desicion_boundary_points import DesicionBoundaryPoints


class TrainResponseDTO(BaseModel):
    final_acurrancy: float
    error_per_apoach: Dict[int, float]
    desicion_boundary_points: DesicionBoundaryPoints
