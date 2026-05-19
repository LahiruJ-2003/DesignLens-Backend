from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from schemas import DesignPayload, UXScoreResponse
import torch
import os

from preprocessing import payload_to_graph
from vigt_model import ViGTModel

app = FastAPI(title="Design Lens AI Backend", version="1.0")

# Allow the Next.js frontend to call this API from a different port
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model once when the server starts so every request doesn't
# have to reload it from disk (that would be very slow)
model = ViGTModel(node_features=8, hidden_dim=64)
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
    # Main endpoint: receives canvas elements from the frontend,
    # runs them through the ViGT model, and returns a quality score
    try:
        # If the canvas is empty, return early to avoid a PyTorch crash
        # (the model can't process a graph with zero nodes)
        if not payload.elements:
            return UXScoreResponse(
                overall_score=100.0,
                issues=[{"severity": "info", "message": "Canvas is empty. Add shapes or text to begin AI analysis.", "id": "msg_empty", "elementIds": []}],
                suggestions=[]
            )

        # Convert the list of UI elements into a graph (nodes + edges)
        x, edge_index = payload_to_graph(payload)
        batch = torch.zeros(x.size(0), dtype=torch.long)

        # Run the graph through the model to get the spatial quality score
        # torch.no_grad() speeds this up since we don't need gradients during inference
        with torch.no_grad():
            score_tensor = model(x, edge_index, batch)
            final_score = score_tensor.item()

        return UXScoreResponse(
            overall_score=final_score,
            issues=[],
            suggestions=[]
        )
    except Exception as e:
        print(f"Error during analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))
