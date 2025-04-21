from pydantic import BaseModel


class TrainRequestDTO(BaseModel):
    max_epoachs: int
    learning_rate: float
    first_feature: int
    second_feature: int
    percentage_to_use: int
