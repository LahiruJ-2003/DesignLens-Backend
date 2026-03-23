from pydantic import BaseModel
from typing import List, Optional

class CanvasElement(BaseModel):
    id: str
    type: str
    x: float
    y: float
    width: float
    height: float
    fill: Optional[str] = None
    stroke: Optional[str] = None
    fontSize: Optional[float] = None
    text: Optional[str] = None
    visible: Optional[bool] = True

class DesignPayload(BaseModel):
    elements: List[CanvasElement]

class UXScoreResponse(BaseModel):
    overall_score: float
    issues: List[dict]
    suggestions: List[dict]
