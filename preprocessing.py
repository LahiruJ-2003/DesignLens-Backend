import torch
from schemas import DesignPayload
from typing import Tuple

def compute_distance(el1, el2):
    cx1, cy1 = el1.x + el1.width/2, el1.y + el1.height/2
    cx2, cy2 = el2.x + el2.width/2, el2.y + el2.height/2
    return ((cx1 - cx2)**2 + (cy1 - cy2)**2)**0.5

def extract_color_features(hex_color: str):
    if not hex_color or hex_color.startswith('rgb') or hex_color.startswith('var'):
        return [0.0, 0.0, 0.0]
    hex_color = hex_color.lstrip('#')
    if len(hex_color) == 6:
        r, g, b = tuple(int(hex_color[i:i+2], 16)/255.0 for i in (0, 2, 4))
        return [r, g, b]
    return [0.0, 0.0, 0.0]

def payload_to_graph(payload: DesignPayload) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Transforms the JSON payload representing the UI Canvas into PyTorch Tensors.
    Returns elements as Node Features (x) and adjacency mapping as (edge_index)
    """
    node_features = []
    edges = []
    
    # 1. Extract Node Features
    for el in payload.elements:
        rgb = extract_color_features(el.fill)
        features = [
            el.x / 1000.0,
            el.y / 1000.0,
            el.width / 1000.0,
            el.height / 1000.0,
            rgb[0], rgb[1]  # Using Red & Green normalized values as simple visual proxy
        ]
        node_features.append(features)
        
    x = torch.tensor(node_features, dtype=torch.float)
    
    # 2. Extract Edges (Connect if spatially close to each other)
    for i, el1 in enumerate(payload.elements):
        for j, el2 in enumerate(payload.elements):
            if i != j:
                dist = compute_distance(el1, el2)
                if dist < 300: # Distance threshold for spatial relationship Edge
                    edges.append([i, j])
                    
    if len(edges) == 0:
        # Add self loops if no edges exist to prevent graph computation errors
        for i in range(len(payload.elements)):
            edges.append([i, i])
            
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    
    return x, edge_index
