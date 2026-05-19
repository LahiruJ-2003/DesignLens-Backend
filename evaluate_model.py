import json
import os
import torch
from vigt_model import ViGTModel
from schemas import DesignPayload, CanvasElement
from preprocessing import payload_to_graph
import math
import time

# This script evaluates how accurate the trained model is.
# Run it after training to see the MAE, RMSE, and classification accuracy.
# I used the last 20% of the dataset as the test set (the model never saw these during training).

def create_mock_payload(data_dict):
    # Same helper as in train.py — converts a dataset dictionary into
    # the DesignPayload format that the preprocessing pipeline expects
    elements = []
    for e in data_dict.get("elements", []):
        elements.append(CanvasElement(**e))
    return DesignPayload(elements=elements)

def load_data():
    # Load whichever dataset is available (RICO takes priority over synthetic)
    if os.path.exists("data/rico_dataset.json"):
        with open("data/rico_dataset.json", "r") as f:
            return json.load(f)
    with open("data/synthetic_dataset.json", "r") as f:
        return json.load(f)

def evaluate_model():
    print("="*50)
    print("   ViGT Model Evaluation & Accuracy Report")
    print("="*50)
    print("Loading test dataset...")
    try:
        dataset = load_data()
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return

    # Take the last 20% of samples as the test set
    # These were not used during training so this gives an honest accuracy reading
    test_size = max(1, int(len(dataset) * 0.2))
    test_data = dataset[-test_size:]
    print(f"Test samples loaded: {len(test_data)}")

    model = ViGTModel(node_features=8, hidden_dim=64)
    if os.path.exists("vigt_model_weights.pth"):
        model.load_state_dict(torch.load("vigt_model_weights.pth"))
        print("Loaded trained weights: vigt_model_weights.pth")
    else:
        print("Warning: No trained weights found.")

    model.eval()

    total_l1_error = 0
    total_squared_error = 0
    inference_times = []

    # I use a threshold of 70 to convert the regression score into a binary
    # good/bad classification. This lets me report classification accuracy
    # alongside the regression metrics.
    classification_threshold = 70.0
    tp = tn = fp = fn = 0

    print("\nRunning inference on test set...")
    with torch.no_grad():
        for sample in test_data:
            payload = create_mock_payload(sample)
            x, edge_index = payload_to_graph(payload)
            target = sample.get("target_score", 50.0)
            batch = torch.zeros(x.size(0), dtype=torch.long)

            # Measure how long inference takes per design
            start_time = time.time()
            pred_score = model(x, edge_index, batch).item()
            inference_times.append(time.time() - start_time)

            error = abs(pred_score - target)
            total_l1_error += error
            total_squared_error += error ** 2

            # Check if the model's good/bad decision matches the true label
            pred_label = 1 if pred_score >= classification_threshold else 0
            true_label = 1 if target >= classification_threshold else 0
            if pred_label == 1 and true_label == 1:
                tp += 1
            elif pred_label == 1 and true_label == 0:
                fp += 1
            elif pred_label == 0 and true_label == 1:
                fn += 1
            else:
                tn += 1

    # MAE tells us the average error in score points (lower = better)
    mae = total_l1_error / len(test_data)
    # RMSE penalises large errors more heavily than MAE
    rmse = math.sqrt(total_squared_error / len(test_data))
    avg_inference = (sum(inference_times) / len(inference_times)) * 1000

    # A simple way to express regression accuracy as a percentage
    pseudo_regression_accuracy = max(0, 100 - mae)

    classification_accuracy = 100.0 * (tp + tn) / len(test_data)
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1_score = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0

    print("\n" + "="*50)
    print("                FINAL METRICS")
    print("="*50)
    print(f"Mean Absolute Error (MAE):          {mae:.2f} points")
    print(f"Root Mean Squared Error (RMSE):     {rmse:.2f} points")
    print(f"Regression Accuracy (100 - MAE):    {pseudo_regression_accuracy:.2f}%")
    print(f"Inference Time (avg):               {avg_inference:.2f} ms/graph")
    print("="*50)
    print("   CLASSIFICATION METRICS (threshold = 70)")
    print("="*50)
    print(f"Classification Accuracy:            {classification_accuracy:.2f}%")
    print(f"Precision:                          {precision:.4f}")
    print(f"Recall:                             {recall:.4f}")
    print(f"F1 Score:                           {f1_score:.4f}")
    print("="*50)

if __name__ == "__main__":
    evaluate_model()
