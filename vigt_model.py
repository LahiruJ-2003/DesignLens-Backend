import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, global_mean_pool

class ViGTModel(nn.Module):
    """
    Vision-Graph Transformer (ViGT)
    Analyzes UI spatial relationships via Graph Attention Networks.
    """
    def __init__(self, node_features=6, hidden_dim=64):
        super(ViGTModel, self).__init__()
        
        # Graph Attention Convolution Layers
        # Allow UI elements heavily overlapping or aligned to influence each other
        self.conv1 = GATConv(node_features, hidden_dim, heads=4, concat=True)
        self.conv2 = GATConv(hidden_dim * 4, hidden_dim, heads=1, concat=False)
        
        # Regression head for predicting the final UX Score (0-100)
        self.fc1 = nn.Linear(hidden_dim, 32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x, edge_index, batch):
        # x: Node features [num_nodes, num_features]
        # edge_index: Graph connectivity [2, num_edges]
        # batch: Batch assignments for pooling
        
        # Layer 1
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        
        # Layer 2
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        
        # Global Pooling (Condense the entire graph into a single vector)
        x = global_mean_pool(x, batch)
        
        # Fully connected layers to output the score
        x = self.fc1(x)
        x = torch.relu(x)
        score = self.fc2(x)
        
        # Sigmoid to normalize between 0 and 100
        score = torch.sigmoid(score) * 100.0
        
        return score
