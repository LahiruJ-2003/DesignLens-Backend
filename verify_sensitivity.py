import torch
from vigt_model import ViGTModel
from preprocessing import payload_to_graph
from schemas import DesignPayload, CanvasElement

def test_sensitivity():
    model = ViGTModel(node_features=6, hidden_dim=64)
    model.load_state_dict(torch.load("vigt_model_weights.pth"))
    model.eval()

    # Case 1: Good Design (Spaced out)
    good_elements = [
        CanvasElement(id="1", type="rectangle", x=10, y=10, width=100, height=100, fill="#ff0000"),
        CanvasElement(id="2", type="rectangle", x=200, y=10, width=100, height=100, fill="#00ff00"),
        CanvasElement(id="3", type="rectangle", x=400, y=10, width=100, height=100, fill="#0000ff")
    ]
    
    # Case 2: Bad Design (Heavy Overlap)
    bad_elements = [
        CanvasElement(id="1", type="rectangle", x=10, y=10, width=200, height=200, fill="#ff0000"),
        CanvasElement(id="2", type="rectangle", x=15, y=15, width=100, height=100, fill="#00ff00"),
        CanvasElement(id="3", type="rectangle", x=20, y=20, width=50, height=50, fill="#0000ff")
    ]

    # Case 3: Excellent Design (10 Spaced nodes)
    excellent_elements = [
        CanvasElement(id=str(i), type="rectangle", x=i*80, y=i*80, width=50, height=50, fill="#0000ff")
        for i in range(10)
    ]

    for label, elements in [("Good-Spaced", good_elements), ("Bad-Overlapping", bad_elements), ("Excellent-10", excellent_elements)]:
        payload = DesignPayload(elements=elements)
        x, edge_index = payload_to_graph(payload)
        batch = torch.zeros(x.size(0), dtype=torch.long)
        
        with torch.no_grad():
            score = model(x, edge_index, batch).item()
            print(f"Design {label} Score: {score:.2f}")

if __name__ == "__main__":
    test_sensitivity()
