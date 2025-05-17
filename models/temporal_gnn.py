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
        
        # Use JIT-compiled operations where possible
        self.input_norm = torch.jit.script(nn.BatchNorm1d(node_features))
        
        # GNN layers with larger capacity for A5000
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # First layer with increased channels
        self.convs.append(GCNConv(node_features, hidden_channels * 2))
        self.batch_norms.append(torch.jit.script(nn.BatchNorm1d(hidden_channels * 2)))
        
        # Middle layers
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels * 2, hidden_channels * 2))
            self.batch_norms.append(torch.jit.script(nn.BatchNorm1d(hidden_channels * 2)))
        
        # Last layer
        self.convs.append(GCNConv(hidden_channels * 2, hidden_channels))
        self.batch_norms.append(torch.jit.script(nn.BatchNorm1d(hidden_channels)))
        
        # Edge prediction layers with increased capacity
        self.edge_predictor = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels * 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_channels * 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_channels),
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
        # Input normalization
        x = self.input_norm(x)
        
        # Apply GNN layers with residual connections
        h = x
        for i, conv in enumerate(self.convs[:-1]):  # All layers except last
            h_new = conv(h, edge_index)
            h_new = self.batch_norms[i](h_new)
            h_new = F.relu(h_new)
            h_new = F.dropout(h_new, p=self.dropout, training=self.training)
            if h.shape == h_new.shape:  # Add residual if shapes match
                h = h + h_new
            else:
                h = h_new
        
        # Last layer without residual
        h = self.convs[-1](h, edge_index)
        h = self.batch_norms[-1](h)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        
        return h
    
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
        
        # Process in chunks if needed for large graphs
        chunk_size = 32768  # Adjust based on GPU memory
        if h_edge.shape[0] > chunk_size:
            preds = []
            for i in range(0, h_edge.shape[0], chunk_size):
                chunk = h_edge[i:i + chunk_size]
                pred_chunk = torch.sigmoid(self.edge_predictor(chunk)).squeeze()
                preds.append(pred_chunk)
            return torch.cat(preds)
        else:
            return torch.sigmoid(self.edge_predictor(h_edge)).squeeze()