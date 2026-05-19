import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, global_mean_pool

class ViGTModel(nn.Module):
    """
    vig model tbh
    figures out which ui things are close to each other using graph networks or smth
    """
    def __init__(self, node_features=8, hidden_dim=64):
        super(ViGTModel, self).__init__()
        
        # graph layers
        # let overlapping stuff affect each other lol
        self.conv1 = GATConv(node_features, hidden_dim, heads=4, concat=True)
        self.conv2 = GATConv(hidden_dim * 4, hidden_dim, heads=1, concat=False)
        
        # outputs the final score (0-100)
        self.fc1 = nn.Linear(hidden_dim, 32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x, edge_index, batch):
        # x is the features
        # edge is how they connect
        # batch is for pooling idk
        
        # first layer
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        
        # second layer
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        
        # squash everything into one vector
        x = global_mean_pool(x, batch)
        
        # output the score
        x = self.fc1(x)
        x = torch.relu(x)
        score = self.fc2(x)
        
        # normalize to 0-100
        score = torch.sigmoid(score) * 100.0
        
        return score
