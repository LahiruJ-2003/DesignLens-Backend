import json
import os
import torch
import torch.nn as nn
import torch.optim as optim
from vigt_model import ViGTModel
from schemas import DesignPayload, CanvasElement
from preprocessing import payload_to_graph

def create_mock_payload(data_dict):
    elements = []
    for e in data_dict.get("elements", []):
        elements.append(CanvasElement(**e))
    return DesignPayload(elements=elements)

def load_data():
    if os.path.exists("data/rico_dataset.json"):
        print("Using REAL Rico Dataset for training!")
        with open("data/rico_dataset.json", "r") as f:
            return json.load(f)
    print("Rico dataset missing. Using Synthetic Dataset.")
    with open("data/synthetic_dataset.json", "r") as f:
        return json.load(f)

def train_model():
    print("Loading synthetic dataset...")
    try:
        dataset = load_data()
    except Exception as e:
        print(f"Failed to load dataset: {e}. Please run dataset_generator.py first.")
        return

    # Initialize Model and Optimizer
    model = ViGTModel(node_features=8, hidden_dim=64)
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.MSELoss()
    
    epochs = 20
    
    print(f"Starting ViGT model training for {epochs} epochs...")
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        for sample in dataset:
            optimizer.zero_grad()
            
            payload = create_mock_payload(sample)
            # Transform to graph
            x, edge_index = payload_to_graph(payload)
            target = torch.tensor([[sample["target_score"]]], dtype=torch.float)
            batch = torch.zeros(x.size(0), dtype=torch.long)
            
            # Forward Pass
            pred_score = model(x, edge_index, batch)
            
            # Calculate Errors and Backpropagate
            loss = criterion(pred_score, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{epochs} | Avg Training Loss: {total_loss/len(dataset):.2f}")
        
    torch.save(model.state_dict(), "vigt_model_weights.pth")
    print("Training completely finished! Weights saved to 'vigt_model_weights.pth'")

if __name__ == "__main__":
    train_model()
