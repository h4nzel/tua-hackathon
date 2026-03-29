import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data

class GNNLandmarkSelector(nn.Module):
    """
    GNN for evaluating the topographical importance of LunarNodes.
    Inputs: [elevation, slope, illumination, roughness, crater_rim, degree]
    Output: Importance Score (0-1)
    """
    
    def __init__(self, in_channels: int = 6, hidden_channels: int = 16, out_channels: int = 1):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels // 2)
        self.fc = nn.Linear(hidden_channels // 2, out_channels)
        self.activation = nn.Sigmoid()
        
    def forward(self, data: Data):
        x, edge_index = data.x, data.edge_index
        
        # Pass 1
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        
        # Pass 2
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        
        # Final fully connected reduction to single scalar per node
        x = self.fc(x)
        return self.activation(x)
