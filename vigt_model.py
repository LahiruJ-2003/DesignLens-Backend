import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, global_mean_pool

# This is the Vision-Graph Transformer (ViGT) model.
# The idea is to treat a UI design as a graph — each element is a node,
# and elements that are close together are connected by edges.
# The model then learns which spatial arrangements look good or bad.

class ViGTModel(nn.Module):

    def __init__(self, node_features=8, hidden_dim=64):
        super(ViGTModel, self).__init__()

        # First graph attention layer: takes the 8 input features per element
        # and expands them to 256 features (64 hidden * 4 attention heads).
        # Using multiple attention heads lets the model focus on different
        # relationships between elements at the same time.
        self.conv1 = GATConv(node_features, hidden_dim, heads=4, concat=True)

        # Second graph attention layer: compresses back down to 64 features.
        # By this point the model has learned spatial relationships between elements.
        self.conv2 = GATConv(hidden_dim * 4, hidden_dim, heads=1, concat=False)

        # Fully connected layers that map the graph representation to a single score
        self.fc1 = nn.Linear(hidden_dim, 32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x, edge_index, batch):
        # x          = node feature matrix (one row per UI element)
        # edge_index = which elements are connected (close to each other)
        # batch      = tells the model which elements belong to which design
        #              (needed when processing multiple designs at once)

        # Pass through the two graph attention layers with ReLU activation
        x = self.conv1(x, edge_index)
        x = torch.relu(x)

        x = self.conv2(x, edge_index)
        x = torch.relu(x)

        # Pool all element features into a single vector representing the whole design
        # global_mean_pool takes the average across all nodes in the graph
        x = global_mean_pool(x, batch)

        # Pass through the final layers to produce a single quality score
        x = self.fc1(x)
        x = torch.relu(x)
        score = self.fc2(x)

        # Sigmoid squashes the output to 0-1, then we multiply by 100
        # to get a human-readable score between 0 and 100
        score = torch.sigmoid(score) * 100.0

        return score
