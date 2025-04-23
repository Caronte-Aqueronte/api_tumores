
from typing import Dict
from pydantic import BaseModel


class TrainResponseDTO(BaseModel):
    final_acurrancy: float
    error_per_apoach: Dict[int, float]
