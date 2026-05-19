import json
import os
# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
from vigt_model import ViGTModel
from schemas import DesignPayload, CanvasElement
from preprocessing import payload_to_graph
import math
import time

# This function builds a fake 'payload' object from our dataset dictionaries
# so we can feed it into the same pipeline that the real API uses.
def create_mock_payload(data_dict):
    elements = []
    for e in data_dict.get("elements", []):
        elements.append(CanvasElement(**e))
    return DesignPayload(elements=elements)

def load_data():
    if os.path.exists("data/rico_dataset.json"):
        with open("data/rico_dataset.json", "r") as f:
            return json.load(f)
    with open("data/synthetic_dataset.json", "r") as f:
        return json.load(f)

# Main script to test how well the model is actually performing on unseen data
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

    # Splitting the data: we use the last 20% as our test set
    # (simulating a basic train/test split here for evaluation)
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
    classification_threshold = 70.0
    tp = tn = fp = fn = 0
    
    print("\nRunning Inference on Test Set...")
    with torch.no_grad(): # Don't track gradients since we're just evaluating (saves RAM)
        for sample in test_data:
            # Recreate the frontend request format
            payload = create_mock_payload(sample)
            x, edge_index = payload_to_graph(payload)
            target = sample.get("target_score", 50.0) # default fallback
            batch = torch.zeros(x.size(0), dtype=torch.long)
            
            start_time = time.time()
            pred_score = model(x, edge_index, batch).item()
            inference_times.append(time.time() - start_time)
            
            error = abs(pred_score - target)
            total_l1_error += error
            total_squared_error += error ** 2

            # Here we threshold the regression score (0-100) into a pass/fail classification
            # e.g., if it's over 70, it's considered a "Good Design" (1), else "Bad Design" (0)
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

    mae = total_l1_error / len(test_data)
    rmse = math.sqrt(total_squared_error / len(test_data))
    avg_inference = (sum(inference_times) / len(inference_times)) * 1000 # in ms
    
    # Regression-style score accuracy
    pseudo_regression_accuracy = max(0, 100 - mae)

    # Classification-style metrics from thresholding the regression output
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
