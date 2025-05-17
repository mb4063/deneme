import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
import math
from torch.utils.checkpoint import checkpoint

class TemporalGAT(nn.Module):
    def __init__(self, node_features, hidden_channels, num_heads=4, dropout=0.3):
        super(TemporalGAT, self).__init__()
        
        self.node_features = node_features
        self.hidden_channels = hidden_channels
        self.num_heads = num_heads
        self.dropout = dropout
        
        # Optimize input projection for large feature dimensions
        projection_dim = min(hidden_channels * 4, node_features * 2)
        self.input_projection = nn.Sequential(
            nn.Linear(node_features, projection_dim),
            nn.ReLU(),
            nn.BatchNorm1d(projection_dim),
            nn.Dropout(dropout),
            nn.Linear(projection_dim, hidden_channels)
        )
        self.proj_norm = nn.BatchNorm1d(hidden_channels)
        
        # Optimize GAT dimensions
        self.gat1_out_channels = hidden_channels // num_heads
        self.gat1_total_channels = self.gat1_out_channels * num_heads
        
        self.gat2_out_channels = hidden_channels // num_heads
        self.gat2_total_channels = self.gat2_out_channels * num_heads
        
        # Memory-efficient GAT layers
        self.gat1 = GATConv(
            hidden_channels,
            self.gat1_out_channels,
            heads=num_heads,
            concat=True,
            dropout=dropout,
            add_self_loops=True,
            bias=True
        )
        
        self.gat2 = GATConv(
            self.gat1_total_channels,
            self.gat2_out_channels,
            heads=num_heads,
            concat=True,
            dropout=dropout,
            add_self_loops=True,
            bias=True
        )
        
        # JIT-compiled batch norms for speed
        self.batch_norm1 = torch.jit.script(nn.BatchNorm1d(self.gat1_total_channels))
        self.batch_norm2 = torch.jit.script(nn.BatchNorm1d(self.gat2_total_channels))
        
        # Simplified temporal attention
        self.temporal_attention = nn.Sequential(
            nn.Linear(1, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, self.gat2_total_channels)
        )
        
        # Memory-efficient edge predictor
        predictor_hidden = min(self.gat2_total_channels, 512)
        self.edge_predictor = nn.Sequential(
            nn.Linear(self.gat2_total_channels * 2, predictor_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(predictor_hidden, 1)
        )
        
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm1d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)

    @torch.cuda.amp.autocast()
    def forward(self, x, edge_index, edge_time=None):
        """
        Memory-efficient forward pass with gradient checkpointing.
        """
        # Initial projection with memory optimization
        with torch.cuda.amp.autocast():
            x = checkpoint(self.input_projection, x, use_reentrant=False)
            x = self.proj_norm(x)
            x = F.relu(x, inplace=True)
            
            # Process in optimized chunks
            chunk_size = 1000000  # 1M edges per chunk
            num_chunks = (edge_index.size(1) + chunk_size - 1) // chunk_size
            
            # First GAT layer
            chunk_outputs = []
            for i in range(num_chunks):
                start_idx = i * chunk_size
                end_idx = min((i + 1) * chunk_size, edge_index.size(1))
                chunk_indices = edge_index[:, start_idx:end_idx]
                
                # Process chunk with temporal attention if available
                if edge_time is not None:
                    chunk_time = edge_time[start_idx:end_idx]
                    time_weights = self.temporal_attention(chunk_time.unsqueeze(-1))
                    chunk_output = checkpoint(
                        lambda x, ei, w: self.gat1(x, ei) * w,
                        x, chunk_indices, time_weights,
                        use_reentrant=False
                    )
                else:
                    chunk_output = checkpoint(
                        self.gat1, x, chunk_indices,
                        use_reentrant=False
                    )
                
                chunk_outputs.append(chunk_output)
                
                # Aggressive memory cleanup
                if i % 50 == 0:
                    torch.cuda.empty_cache()
            
            # Efficient aggregation
            x = torch.stack(chunk_outputs).mean(dim=0)
            x = self.batch_norm1(x)
            x = F.elu(x, inplace=True)
            
            # Second GAT layer with similar optimizations
            chunk_outputs = []
            for i in range(num_chunks):
                start_idx = i * chunk_size
                end_idx = min((i + 1) * chunk_size, edge_index.size(1))
                chunk_indices = edge_index[:, start_idx:end_idx]
                
                if edge_time is not None:
                    chunk_time = edge_time[start_idx:end_idx]
                    time_weights = self.temporal_attention(chunk_time.unsqueeze(-1))
                    chunk_output = checkpoint(
                        lambda x, ei, w: self.gat2(x, ei) * w,
                        x, chunk_indices, time_weights,
                        use_reentrant=False
                    )
                else:
                    chunk_output = checkpoint(
                        self.gat2, x, chunk_indices,
                        use_reentrant=False
                    )
                
                chunk_outputs.append(chunk_output)
                
                if i % 50 == 0:
                    torch.cuda.empty_cache()
            
            x = torch.stack(chunk_outputs).mean(dim=0)
            x = self.batch_norm2(x)
            x = F.elu(x, inplace=True)
            
            return x

    @torch.cuda.amp.autocast()
    def predict_link(self, h, edge_index):
        """
        Memory-efficient link prediction.
        """
        chunk_size = 65536  # Increased for A5000
        logits = []
        
        for i in range(0, edge_index.size(1), chunk_size):
            chunk_indices = edge_index[:, i:i + chunk_size]
            
            # Efficient feature combination
            h_combined = torch.cat([
                h[chunk_indices[0]],
                h[chunk_indices[1]]
            ], dim=1)
            
            # Predict chunk
            with torch.cuda.amp.autocast():
                logit_chunk = self.edge_predictor(h_combined).squeeze()
            
            logits.append(logit_chunk)
            
            # Clean up
            del h_combined
            if i % 50 == 0:
                torch.cuda.empty_cache()
        
        return torch.cat(logits)

    @torch.cuda.amp.autocast()
    def validation_forward(self, x, edge_index, edge_time=None):
        """
        Optimized forward pass for validation/inference.
        """
        with torch.no_grad():
            # Initial projection
            x = self.input_projection(x)
            x = self.proj_norm(x)
            x = F.relu(x, inplace=True)
            
            # Process in larger chunks for validation
            chunk_size = 2000000  # 2M edges per chunk for validation
            num_chunks = (edge_index.size(1) + chunk_size - 1) // chunk_size
            
            # First layer
            chunk_outputs = []
            for i in range(num_chunks):
                start_idx = i * chunk_size
                end_idx = min((i + 1) * chunk_size, edge_index.size(1))
                chunk_indices = edge_index[:, start_idx:end_idx]
                
                chunk_output = self.gat1(x, chunk_indices)
                if edge_time is not None:
                    chunk_time = edge_time[start_idx:end_idx]
                    time_weights = self.temporal_attention(chunk_time.unsqueeze(-1))
                    chunk_output = chunk_output * time_weights
                
                chunk_outputs.append(chunk_output)
                torch.cuda.empty_cache()
            
            x = torch.stack(chunk_outputs).mean(dim=0)
            x = self.batch_norm1(x)
            x = F.elu(x, inplace=True)
            
            # Second layer
            chunk_outputs = []
            for i in range(num_chunks):
                start_idx = i * chunk_size
                end_idx = min((i + 1) * chunk_size, edge_index.size(1))
                chunk_indices = edge_index[:, start_idx:end_idx]
                
                chunk_output = self.gat2(x, chunk_indices)
                if edge_time is not None:
                    chunk_time = edge_time[start_idx:end_idx]
                    time_weights = self.temporal_attention(chunk_time.unsqueeze(-1))
                    chunk_output = chunk_output * time_weights
                
                chunk_outputs.append(chunk_output)
                torch.cuda.empty_cache()
            
            x = torch.stack(chunk_outputs).mean(dim=0)
            x = self.batch_norm2(x)
            x = F.elu(x, inplace=True)
            
            return x 