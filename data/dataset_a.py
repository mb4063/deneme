import torch
import pandas as pd
import numpy as np
from torch_geometric_temporal.signal import DynamicGraphTemporalSignal
import os
from tqdm import tqdm
from datetime import datetime

class EventBasedDataset:
    """
    Dataset class for event-based temporal data with proper temporal handling.
    """
    
    def __init__(self, name, time_window):
        """
        Initialize the dataset.
        
        Args:
            name (str): Name of the dataset
            time_window (int): Size of the time window for prediction (in seconds)
        """
        self.name = name
        self.time_window = time_window
        self.temporal_signal = None  # Will hold DynamicGraphTemporalSignal
        self.node_feature_path = "data/node_features.csv"
        self.edge_type_path = "data/edge_type_features.csv"
        
    def load_data(self, data_path):
        """
        Load data with temporal awareness and proper negative example handling.
        """
        print(f"Loading data from {data_path}")
        
        # Load data using pandas
        df = pd.read_csv(
            data_path,
            header=None,
            engine='c',
            dtype={
                0: np.int32,  # src_id
                1: np.int32,  # dst_id
                2: np.int32,  # edge_type
                3: np.float32,  # timestamp
                4: np.int32  # label
            }
        )
        
        # Sort by timestamp
        df = df.sort_values(by=3)
        
        # Create time windows
        timestamps = df[3].unique()
        timestamps.sort()
        
        # Process node features
        all_nodes = np.unique(np.concatenate([df[0].unique(), df[1].unique()]))
        num_nodes = len(all_nodes)
        node_features = self._process_node_features(num_nodes)
        
        # Create temporal graph snapshots
        edge_indices = []
        edge_weights = []
        target_signals = []
        
        # Group by time windows
        for t in timestamps:
            window_data = df[df[3] == t]
            
            # Edge index for this timestamp
            edges = torch.tensor(window_data[[0, 1]].values.T, dtype=torch.long)
            edge_indices.append(edges)
            
            # Edge weights (can be edge types or other features)
            weights = torch.tensor(window_data[2].values, dtype=torch.float)
            edge_weights.append(weights)
            
            # Target signals (labels)
            targets = torch.tensor(window_data[4].values, dtype=torch.float)
            target_signals.append(targets)
        
        # Create DynamicGraphTemporalSignal
        self.temporal_signal = DynamicGraphTemporalSignal(
            edge_indices=edge_indices,
            edge_weights=edge_weights,
            features=node_features,
            targets=target_signals
        )
        
    def _process_node_features(self, num_nodes):
        """Process and return node features for all timestamps"""
        try:
            if os.path.exists(self.node_feature_path):
                node_df = pd.read_csv(self.node_feature_path)
                features = torch.from_numpy(node_df.values).float()
                if len(features) < num_nodes:
                    # Pad with zeros if needed
                    pad_size = num_nodes - len(features)
                    features = torch.cat([features, torch.zeros(pad_size, features.size(1))])
                return features
            else:
                # Default to one-hot encoding if no features available
                return torch.eye(num_nodes, dtype=torch.float)
        except Exception as e:
            print(f"Warning: Using default node features. Error: {e}")
            return torch.eye(num_nodes, dtype=torch.float)