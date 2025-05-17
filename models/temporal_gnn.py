import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class TemporalGNN(nn.Module):
    """
    Temporal Graph Neural Network for link prediction.
    """
    
    def __init__(self, node_features, hidden_channels, num_layers=2, dropout=0.3):
        """
        Initialize the Temporal GNN model.
        
        Args:
            node_features (int): Number of node features
            hidden_channels (int): Number of hidden channels
            num_layers (int): Number of GNN layers
            dropout (float): Dropout probability
        """
        super(TemporalGNN, self).__init__()
        
        self.node_features = node_features
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        
        # GNN layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # İlk katman
        self.convs.append(GCNConv(node_features, hidden_channels))
        self.batch_norms.append(nn.BatchNorm1d(hidden_channels))
        
        # Ara katmanlar
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.batch_norms.append(nn.BatchNorm1d(hidden_channels))
        
        # Edge prediction layers - daha basit mimari
        self.edge_predictor = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 1)
        )
        
        self.dropout = dropout
        
    def forward(self, x, edge_index):
        """
        Forward pass through the model.
        
        Args:
            x (torch.Tensor): Node features [num_nodes, num_features]
            edge_index (torch.Tensor): Edge indices [2, num_edges]
            
        Returns:
            torch.Tensor: Node embeddings [num_nodes, hidden_channels]
        """
        # Apply GNN layers with batch norm
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        return x
    
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
        src = edge_index[:, 0]
        dst = edge_index[:, 1]
        
        # Debug için assert ekle
        assert src.max().item() < h.shape[0], f"src max: {src.max().item()}, h.shape: {h.shape}"
        assert dst.max().item() < h.shape[0], f"dst max: {dst.max().item()}, h.shape: {h.shape}"
        
        # Get node embeddings for the pairs
        h_src = h[src]
        h_dst = h[dst]
        
        # Concatenate embeddings
        h_edge = torch.cat([h_src, h_dst], dim=1)
        
        # Predict link probability
        pred = self.edge_predictor(h_edge)
        
        return torch.sigmoid(pred).squeeze()