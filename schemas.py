from pydantic import BaseModel
from typing import List, Optional

# This file defines the data structures (schemas) that the frontend sends to the backend
# and what the backend sends back. Pydantic automatically validates the data types for us.

class CanvasElement(BaseModel):
    # Represents a single UI element drawn on the canvas (rectangle, text, circle, etc.)
    id: str
    type: str
    x: float
    y: float
    width: float
    height: float
    fill: Optional[str] = None       # background colour (hex string like #3b82f6)
    stroke: Optional[str] = None     # border colour
    fontSize: Optional[float] = None # only used for text elements
    text: Optional[str] = None       # only used for text elements
    visible: Optional[bool] = True

class DesignPayload(BaseModel):
    # The full request body sent from the frontend when asking for a score
    # We also need the frame size so we can normalise element positions correctly
    elements: List[CanvasElement]
    frame_width: Optional[float] = 390.0   # defaults to iPhone 14 width if not provided
    frame_height: Optional[float] = 844.0  # defaults to iPhone 14 height if not provided

class UXScoreResponse(BaseModel):
    # What we send back to the frontend after the model runs
    overall_score: float   # the spatial quality score from ViGT (0 to 100)
    issues: List[dict]     # any design issues detected
    suggestions: List[dict]
