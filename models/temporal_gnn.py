import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric_temporal.nn import GConvLSTM

class TemporalGNN(nn.Module):
    """
    Temporal Graph Neural Network for link prediction.
    """
    
    def __init__(self, node_features, hidden_channels, num_layers=2):
        """
        Initialize the Temporal GNN model.
        
        Args:
            node_features (int): Number of node features
            hidden_channels (int): Number of hidden channels
            num_layers (int): Number of GNN layers
        """
        super(TemporalGNN, self).__init__()
        
        self.node_features = node_features
        self.hidden_channels = hidden_channels
        
        # GConvLSTM layer from PyTorch Geometric Temporal
        self.temporal_conv = GConvLSTM(
            in_channels=node_features,
            out_channels=hidden_channels,
            K=num_layers,  # Number of filter taps
            normalization='sym'
        )
        
        # Link prediction layers
        self.link_predictor = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 1)
        )
    
    def forward(self, x, edge_index, edge_weight=None, h=None, c=None):
        """
        Forward pass through the model.
        
        Args:
            x (torch.Tensor): Node features [num_nodes, num_features]
            edge_index (torch.Tensor): Edge indices [2, num_edges]
            edge_weight (torch.Tensor): Edge weights [num_edges]
            h (torch.Tensor): Hidden state from previous time step [num_nodes, hidden_channels]
            c (torch.Tensor): Cell state from previous time step [num_nodes, hidden_channels]
            
        Returns:
            torch.Tensor: Node embeddings [num_nodes, hidden_channels]
            torch.Tensor: Hidden state from current time step [num_nodes, hidden_channels]
            torch.Tensor: Cell state from current time step [num_nodes, hidden_channels]
        """
        # Forward pass through temporal graph convolution
        out, h, c = self.temporal_conv(x, edge_index, edge_weight, h, c)
        return out, h, c
    
    def predict_link(self, h, edge_index):
        """
        Predict the probability of a link between nodes.
        
        Args:
            h (torch.Tensor): Node embeddings [num_nodes, hidden_channels]
            edge_index (torch.Tensor): Edge indices [num_edges, 2]
            
        Returns:
            torch.Tensor: Link prediction probabilities [num_edges]
        """
        # Get node pairs
        src, dst = edge_index
        
        # Get node embeddings for the pairs
        h_src = h[src]
        h_dst = h[dst]
        
        # Concatenate embeddings
        h_edge = torch.cat([h_src, h_dst], dim=1)
        
        # Predict link probability
        return torch.sigmoid(self.link_predictor(h_edge))