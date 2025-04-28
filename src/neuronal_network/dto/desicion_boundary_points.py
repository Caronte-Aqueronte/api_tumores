from pydantic import BaseModel


class DesicionBoundaryPoints(BaseModel):
    x: float
    y: float
    x2: float
    y2: float
