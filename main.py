# Hey! This is the main entry point for the backend API.
# It sets up FastAPI and handles incoming requests from the frontend design tool.

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from schemas import DesignPayload, UXScoreResponse
import torch
import os

from preprocessing import payload_to_graph
from vigt_model import ViGTModel

app = FastAPI(title="Design Lens AI Backend", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Model Initialization
# Loading the model here so we don't have to reload it for every single API request (which would be super slow!)
model = ViGTModel(node_features=6, hidden_dim=64)
weights_path = "vigt_model_weights.pth"

if os.path.exists(weights_path):
    model.load_state_dict(torch.load(weights_path))
    model.eval()
    print("Successfully loaded trained ViGT model weights.")
else:
    print("WARNING: Model weights not found. Model is using untrained parameters.")

@app.get("/")
def read_root():
    return {"message": "Welcome to the Design Lens Vision-Graph Transformer API"}

@app.post("/api/analyze-ui", response_model=UXScoreResponse)
async def analyze_ui(payload: DesignPayload):
    """
    Main endpoint that takes the canvas JSON from the frontend and runs it through the AI model.
    """
    try:
        # Handle entirely empty canvas gracefully to prevent PyTorch zero-dimensional tensor crash
        # (Basically, if there's nothing on the screen, don't break the whole app)
        if not payload.elements:
            return UXScoreResponse(
                overall_score=100.0,
                issues=[{"severity": "info", "message": "Canvas is empty. Add shapes or text to begin AI analysis.", "id": "msg_empty", "elementIds": []}],
                suggestions=[]
            )
            
        # Preprocess UI JSON into PyTorch Graph
        # This converts the raw frontend data into node features and edges that our Graph Neural Network can actually read
        x, edge_index = payload_to_graph(payload)
        batch = torch.zeros(x.size(0), dtype=torch.long)
        
        # Inference using Vision-Graph Transformer
        # Running the data through the model without calculating gradients (saves memory during inference)
        with torch.no_grad():
            score_tensor = model(x, edge_index, batch)
            final_score = score_tensor.item()
            
        return UXScoreResponse(
            overall_score=final_score,
            issues=[{"severity": "info", "message": "Graph analysis successfully mapped architecture logic", "id": "msg_01", "elementIds": []}],
            suggestions=[]
        )
    except Exception as e:
        print(f"Error during analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))
