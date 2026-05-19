import json
import os
import torch
import torch.nn as nn
import torch.optim as optim
from vigt_model import ViGTModel
from schemas import DesignPayload, CanvasElement
from preprocessing import payload_to_graph

# This script trains the ViGT model on our dataset.
# Run this once after generating the dataset, then the weights are saved
# and the API uses them for scoring without needing to retrain each time.

def create_mock_payload(data_dict):
    # Each sample in our dataset is a dictionary.
    # This function converts it into the same format the real API receives
    # so we can reuse the same preprocessing pipeline for training.
    elements = []
    for e in data_dict.get("elements", []):
        elements.append(CanvasElement(**e))
    return DesignPayload(elements=elements)

def load_data():
    # Use the RICO dataset if available (real Android UIs = better training data)
    # Fall back to the synthetic dataset if RICO hasn't been parsed yet
    if os.path.exists("data/rico_dataset.json"):
        print("Using REAL Rico Dataset for training!")
        with open("data/rico_dataset.json", "r") as f:
            return json.load(f)
    print("Rico dataset missing. Using Synthetic Dataset.")
    with open("data/synthetic_dataset.json", "r") as f:
        return json.load(f)

def train_model():
    print("Loading dataset...")
    try:
        dataset = load_data()
    except Exception as e:
        print(f"Failed to load dataset: {e}. Please run dataset_generator.py first.")
        return

    # Set up the model, optimiser, and loss function
    # Adam is a good default optimiser and MSE is appropriate for regression tasks
    model = ViGTModel(node_features=8, hidden_dim=64)
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.MSELoss()

    epochs = 20
    print(f"Starting training for {epochs} epochs...")
    model.train()

    for epoch in range(epochs):
        total_loss = 0
        for sample in dataset:
            optimizer.zero_grad()

            # Convert the dataset sample into a graph the model can process
            payload = create_mock_payload(sample)
            x, edge_index = payload_to_graph(payload)

            # The target score is what the model should output for this design
            target = torch.tensor([[sample["target_score"]]], dtype=torch.float)
            batch = torch.zeros(x.size(0), dtype=torch.long)

            # Forward pass: get the model's predicted score
            pred_score = model(x, edge_index, batch)

            # Calculate how wrong the prediction was and update the model weights
            loss = criterion(pred_score, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs} | Avg Loss: {total_loss/len(dataset):.2f}")

    # Save the trained weights so the API can load them without retraining
    torch.save(model.state_dict(), "vigt_model_weights.pth")
    print("Training finished! Weights saved to vigt_model_weights.pth")

if __name__ == "__main__":
    train_model()
