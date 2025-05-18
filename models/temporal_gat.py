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
        
        # Input projection with proper dimensions and explicit intermediate size
        intermediate_dim = hidden_channels * 2
        self.input_projection = nn.Sequential(
            nn.Linear(node_features, intermediate_dim),
            nn.ReLU(),
            nn.LayerNorm(intermediate_dim),  # Replace BatchNorm with LayerNorm for better stability
            nn.Dropout(dropout),
            nn.Linear(intermediate_dim, hidden_channels)
        )
        self.proj_norm = nn.LayerNorm(hidden_channels)  # Replace BatchNorm with LayerNorm
        
        # Optimize GAT dimensions
        self.gat1_out_channels = hidden_channels // num_heads
        self.gat1_total_channels = self.gat1_out_channels * num_heads
        
        self.gat2_out_channels = hidden_channels // num_heads
        self.gat2_total_channels = self.gat2_out_channels * num_heads
        
        # Memory-efficient GAT layers with edge dimension
        self.gat1 = GATConv(
            hidden_channels,
            self.gat1_out_channels,
            heads=num_heads,
            concat=True,
            dropout=dropout,
            add_self_loops=True,
            bias=True,
            edge_dim=self.hidden_channels
        )
        
        self.gat2 = GATConv(
            self.gat1_total_channels,
            self.gat2_out_channels,
            heads=num_heads,
            concat=True,
            dropout=dropout,
            add_self_loops=True,
            bias=True,
            edge_dim=self.hidden_channels
        )
        
        # Replace BatchNorm with LayerNorm
        self.norm1 = nn.LayerNorm(self.gat1_total_channels)
        self.norm2 = nn.LayerNorm(self.gat2_total_channels)
        
        # Simplified temporal attention with stability measures
        self.temporal_attention = nn.Sequential(
            nn.Linear(1, hidden_channels),
            nn.LayerNorm(hidden_channels),  # Add normalization
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, self.gat2_total_channels),
            nn.Tanh()  # Bound the output
        )
        
        # Memory-efficient edge predictor with stability measures
        predictor_hidden = min(self.gat2_total_channels, 512)
        self.edge_predictor = nn.Sequential(
            nn.Linear(self.gat2_total_channels * 2, predictor_hidden),
            nn.LayerNorm(predictor_hidden),  # Add normalization
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(predictor_hidden, 1)
        )
        
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # Use a more stable initialization
            nn.init.xavier_uniform_(module.weight, gain=1.0)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.LayerNorm, nn.BatchNorm1d)):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    @torch.cuda.amp.autocast()
    def forward(self, x, edge_index, edge_time=None):
        """
        Memory-efficient forward pass with gradient checkpointing and stability measures.
        """
        with torch.cuda.amp.autocast():
            # Initial projection with stability measures
            batch_size, feat_dim = x.size()
            assert feat_dim == self.node_features, f"Expected input features of size {self.node_features}, but got {feat_dim}"
            
            # Project node features
            x = self.input_projection(x)
            x = self.proj_norm(x)
            x = F.relu(x)
            
            # Clamp values to prevent extreme activations
            x = torch.clamp(x, min=-10.0, max=10.0)
            
            # Process in optimized chunks with stability measures
            chunk_size = 1000000  # 1M edges per chunk
            num_chunks = (edge_index.size(1) + chunk_size - 1) // chunk_size
            
            # First GAT layer
            chunk_outputs = []
            for i in range(num_chunks):
                start_idx = i * chunk_size
                end_idx = min((i + 1) * chunk_size, edge_index.size(1))
                chunk_indices = edge_index[:, start_idx:end_idx]
                
                if edge_time is not None:
                    chunk_time = edge_time[start_idx:end_idx]
                    time_weights = self.temporal_attention(chunk_time.unsqueeze(-1))
                    time_weights = torch.clamp(time_weights, min=-3.0, max=3.0)  # Prevent extreme temporal weights
                    
                    with torch.cuda.amp.autocast(enabled=False):
                        x_float32 = x.float()
                        time_weights_float32 = time_weights.float()
                        chunk_output = checkpoint(
                            lambda x_lambda, ei_lambda, w_lambda: self.gat1(x_lambda, ei_lambda, edge_attr=w_lambda),
                            x_float32, chunk_indices, time_weights_float32,
                            use_reentrant=False
                        )
                else:
                    num_edges_in_chunk = chunk_indices.size(1)
                    dummy_edge_attr = torch.zeros(num_edges_in_chunk, self.hidden_channels, device=x.device, dtype=x.dtype)
                    
                    with torch.cuda.amp.autocast(enabled=False):
                        x_float32 = x.float()
                        dummy_edge_attr_float32 = dummy_edge_attr.float()
                        chunk_output = checkpoint(
                            lambda x_lambda, ei_lambda, ea_lambda: self.gat1(x_lambda, ei_lambda, edge_attr=ea_lambda),
                            x_float32, chunk_indices, dummy_edge_attr_float32,
                            use_reentrant=False
                        )
                
                # Add residual connection and normalize
                if i == 0:
                    chunk_output = chunk_output + x_float32  # Residual for first chunk
                chunk_output = torch.clamp(chunk_output, min=-10.0, max=10.0)  # Prevent extreme values
                chunk_outputs.append(chunk_output)
                
                if i % 50 == 0:
                    torch.cuda.empty_cache()
            
            # Efficient aggregation with stability
            x = torch.stack(chunk_outputs).mean(dim=0)
            x = self.norm1(x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            
            # Second GAT layer with similar stability measures
            chunk_outputs = []
            for i in range(num_chunks):
                start_idx = i * chunk_size
                end_idx = min((i + 1) * chunk_size, edge_index.size(1))
                chunk_indices = edge_index[:, start_idx:end_idx]
                
                if edge_time is not None:
                    chunk_time = edge_time[start_idx:end_idx]
                    time_weights = self.temporal_attention(chunk_time.unsqueeze(-1))
                    time_weights = torch.clamp(time_weights, min=-3.0, max=3.0)
                    
                    with torch.cuda.amp.autocast(enabled=False):
                        x_float32 = x.float()
                        time_weights_float32 = time_weights.float()
                        chunk_output = checkpoint(
                            lambda x_lambda, ei_lambda, w_lambda: self.gat2(x_lambda, ei_lambda, edge_attr=w_lambda),
                            x_float32, chunk_indices, time_weights_float32,
                            use_reentrant=False
                        )
                else:
                    num_edges_in_chunk = chunk_indices.size(1)
                    dummy_edge_attr = torch.zeros(num_edges_in_chunk, self.hidden_channels, device=x.device, dtype=x.dtype)
                    
                    with torch.cuda.amp.autocast(enabled=False):
                        x_float32 = x.float()
                        dummy_edge_attr_float32 = dummy_edge_attr.float()
                        chunk_output = checkpoint(
                            lambda x_lambda, ei_lambda, ea_lambda: self.gat2(x_lambda, ei_lambda, edge_attr=ea_lambda),
                            x_float32, chunk_indices, dummy_edge_attr_float32,
                            use_reentrant=False
                        )
                
                # Add residual connection and normalize
                if i == 0:
                    chunk_output = chunk_output + x_float32
                chunk_output = torch.clamp(chunk_output, min=-10.0, max=10.0)
                chunk_outputs.append(chunk_output)
                
                if i % 50 == 0:
                    torch.cuda.empty_cache()
            
            x = torch.stack(chunk_outputs).mean(dim=0)
            x = self.norm2(x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            
            return x

    @torch.cuda.amp.autocast()
    def predict_link(self, h, edge_index):
        """
        Memory-efficient link prediction with stability measures.
        """
        chunk_size = 65536
        logits = []
        
        for i in range(0, edge_index.size(1), chunk_size):
            chunk_indices = edge_index[:, i:i + chunk_size]
            
            h_combined = torch.cat([
                h[chunk_indices[0]],
                h[chunk_indices[1]]
            ], dim=1)
            
            # Predict chunk with stability measures
            with torch.cuda.amp.autocast(enabled=False):
                h_combined_float32 = h_combined.float()
                logit_chunk = self.edge_predictor(h_combined_float32).squeeze(-1)
                logit_chunk = torch.clamp(logit_chunk, min=-10.0, max=10.0)  # Prevent extreme logits
            
            logits.append(logit_chunk)
            
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
                
                if edge_time is not None:
                    chunk_time = edge_time[start_idx:end_idx]
                    time_weights = self.temporal_attention(chunk_time.unsqueeze(-1))
                    chunk_output = self.gat1(x, chunk_indices, edge_attr=time_weights)
                else:
                    chunk_output = self.gat1(x, chunk_indices)
                
                chunk_outputs.append(chunk_output)
                torch.cuda.empty_cache()
            
            x = torch.stack(chunk_outputs).mean(dim=0)
            x = self.norm1(x)
            x = F.elu(x, inplace=True)
            
            # Second layer
            chunk_outputs = []
            for i in range(num_chunks):
                start_idx = i * chunk_size
                end_idx = min((i + 1) * chunk_size, edge_index.size(1))
                chunk_indices = edge_index[:, start_idx:end_idx]
                
                if edge_time is not None:
                    chunk_time = edge_time[start_idx:end_idx]
                    time_weights = self.temporal_attention(chunk_time.unsqueeze(-1))
                    chunk_output = self.gat2(x, chunk_indices, edge_attr=time_weights)
                else:
                    chunk_output = self.gat2(x, chunk_indices)
                
                chunk_outputs.append(chunk_output)
                torch.cuda.empty_cache()
            
            x = torch.stack(chunk_outputs).mean(dim=0)
            x = self.norm2(x)
            x = F.elu(x, inplace=True)
            
            return x 