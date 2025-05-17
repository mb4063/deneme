import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class TemporalGAT(nn.Module):
    def __init__(self, node_features, hidden_channels, num_heads=4, dropout=0.3):
        super(TemporalGAT, self).__init__()
        
        self.node_features = node_features
        self.hidden_channels = hidden_channels
        self.num_heads = num_heads
        
        # İlk GAT katmanı
        self.gat1 = GATConv(
            in_channels=node_features,
            out_channels=hidden_channels // num_heads,
            heads=num_heads,
            dropout=dropout
        )
        
        # İkinci GAT katmanı
        self.gat2 = GATConv(
            in_channels=hidden_channels,
            out_channels=hidden_channels // num_heads,
            heads=num_heads,
            dropout=dropout,
            concat=False  # Son katmanda concatenation yapmıyoruz
        )
        
        # Temporal attention layer
        self.temporal_attention = nn.Sequential(
            nn.Linear(1, hidden_channels // num_heads),  # Boyut düzeltildi
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // num_heads, 1),  # Boyut düzeltildi
            nn.Sigmoid()
        )
        
        # Edge prediction layers
        self.edge_predictor = nn.Sequential(
            nn.Linear((hidden_channels // num_heads) * 2, hidden_channels // num_heads),  # Boyut düzeltildi
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // num_heads, 1)
        )
        
        # Batch normalization layers
        self.batch_norm1 = nn.BatchNorm1d(hidden_channels)  # İlk GAT çıkışı için
        self.batch_norm2 = nn.BatchNorm1d(hidden_channels // num_heads)  # İkinci GAT çıkışı için
        
        self.dropout = dropout
        
    def forward(self, x, edge_index, edge_time=None):
        # İlk GAT katmanı
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gat1(x, edge_index)
        x = self.batch_norm1(x)
        x = F.elu(x)
        
        # İkinci GAT katmanı
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gat2(x, edge_index)
        x = self.batch_norm2(x)
        x = F.elu(x)
        
        # Temporal attention if time info available
        if edge_time is not None:
            # Normalize time differences
            time_diffs = edge_time.unsqueeze(-1)
            time_weights = self.temporal_attention(time_diffs)
            x = x * time_weights
            
        return x
    
    def predict_link(self, h, edge_index):
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