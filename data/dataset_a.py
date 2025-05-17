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
        self.node_features = None
        self.edge_indices = None
        self.edge_features = None
        self.edge_timestamps = None
        self.targets = None
        self.node_feature_path = "data/node_features.csv"
        self.edge_type_path = "data/edge_type_features.csv"
        
        # New temporal attributes
        self.temporal_edges = []  # List of (src, dst, timestamp, label) tuples
        self.temporal_windows = []  # List of time windows
        self.min_time = float('inf')
        self.max_time = float('-inf')
        
    def load_data(self, data_path):
        """
        Load data with temporal awareness and proper negative example handling.
        """
        print(f"Loading data from {data_path}")
        
        # Count lines for progress bar
        print("Counting total lines...")
        total_lines = sum(1 for _ in open(data_path))
        print(f"Total lines: {total_lines:,}")
        
        # Load data using pandas with optimized settings
        print("Loading CSV data...")
        df = pd.read_csv(
            data_path, 
            header=None,
            engine='c',
            dtype={
                0: np.int32,  # src_id
                1: np.int32,  # dst_id
                2: np.int32,  # edge_type
                3: np.float32,  # timestamp
                4: np.int32  # label (if exists, otherwise will be dropped)
            },
            memory_map=True
        )
        
        # Check if we have labels in the data
        has_labels = df.shape[1] > 4
        
        # Convert to numpy arrays efficiently
        sources = df.iloc[:, 0].to_numpy(dtype=np.int32)
        targets = df.iloc[:, 1].to_numpy(dtype=np.int32)
        edge_types = df.iloc[:, 2].to_numpy(dtype=np.int32)
        timestamps = df.iloc[:, 3].to_numpy(dtype=np.float32)
        
        if has_labels:
            labels = df.iloc[:, 4].to_numpy(dtype=np.int32)
        else:
            # If no labels, assume all edges are positive
            labels = np.ones_like(sources, dtype=np.int32)
        
        # Update temporal bounds
        self.min_time = min(self.min_time, timestamps.min())
        self.max_time = max(self.max_time, timestamps.max())
        
        # Store temporal edges
        temporal_data = list(zip(sources, targets, timestamps, labels))
        temporal_data.sort(key=lambda x: x[2])  # Sort by timestamp
        self.temporal_edges.extend(temporal_data)
        
        # Create time windows with optimized vectorization
        print("Creating temporal windows...")
        window_size = self.time_window
        
        # Calculate window boundaries more efficiently
        min_time = timestamps.min()
        max_time = timestamps.max()
        num_windows = int(np.ceil((max_time - min_time) / window_size))
        window_starts = min_time + np.arange(num_windows) * window_size
        window_ends = window_starts + window_size
        
        # Pre-sort timestamps and corresponding data
        sort_indices = np.argsort(timestamps)
        sorted_timestamps = timestamps[sort_indices]
        sorted_sources = sources[sort_indices]
        sorted_targets = targets[sort_indices]
        sorted_labels = labels[sort_indices]
        
        # Use binary search to find window boundaries
        self.temporal_windows = []
        
        print(f"Processing {num_windows} windows...")
        for start, end in tqdm(zip(window_starts, window_ends), total=num_windows):
            # Binary search for start and end indices
            start_idx = np.searchsorted(sorted_timestamps, start)
            end_idx = np.searchsorted(sorted_timestamps, end)
            
            if start_idx < end_idx:  # Only create window if it contains edges
                window_edges = list(zip(
                    sorted_sources[start_idx:end_idx],
                    sorted_targets[start_idx:end_idx],
                    sorted_timestamps[start_idx:end_idx],
                    sorted_labels[start_idx:end_idx]
                ))
                
                self.temporal_windows.append({
                    'start': start,
                    'end': end,
                    'edges': window_edges
                })
                
                # Free memory
                if len(self.temporal_windows) % 100 == 0:
                    torch.cuda.empty_cache()
        
        # Free memory
        del sorted_timestamps, sorted_sources, sorted_targets, sorted_labels
        torch.cuda.empty_cache()
        
        print(f"Created {len(self.temporal_windows)} temporal windows")
        
        # Convert edges to tensors efficiently
        print("Converting to PyTorch tensors...")
        edge_data = np.array(self.temporal_edges, dtype=[
            ('src', np.int32),
            ('dst', np.int32),
            ('time', np.float32),
            ('label', np.int32)
        ])
        
        self.edge_timestamps = torch.from_numpy(edge_data['time']).float()
        self.targets = torch.from_numpy(edge_data['label']).float()
        
        # Process node mapping
        print("\nBuilding node mapping...")
        all_node_ids = np.unique(np.concatenate([sources, targets]))
        num_nodes = len(all_node_ids)
        print(f"Number of unique nodes: {num_nodes}")
        
        # Create node ID mapping
        print("Creating node ID mapping...")
        node_id_map = {old_id: new_id for new_id, old_id in enumerate(sorted(all_node_ids))}
        
        # Map node IDs and validate
        print("Mapping node IDs...")
        mapped_sources = np.array([node_id_map[src] for src in sources], dtype=np.int64)
        mapped_targets = np.array([node_id_map[dst] for dst in targets], dtype=np.int64)
        
        # Validate indices
        max_node_idx = num_nodes - 1
        assert mapped_sources.max() <= max_node_idx, f"Source index {mapped_sources.max()} >= num_nodes {num_nodes}"
        assert mapped_targets.max() <= max_node_idx, f"Target index {mapped_targets.max()} >= num_nodes {num_nodes}"
        assert mapped_sources.min() >= 0, f"Negative source index found: {mapped_sources.min()}"
        assert mapped_targets.min() >= 0, f"Negative target index found: {mapped_targets.min()}"
        
        # Update edge indices with mapped values
        self.edge_indices = torch.from_numpy(
            np.stack([mapped_sources, mapped_targets])
        ).long()
        
        # Create edge features
        if self.edge_type_path and os.path.exists(self.edge_type_path):
            self.edge_features = torch.from_numpy(self._process_edge_features(edge_types)).float()
        
        # Process node features
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._process_node_features(num_nodes, device)
        
        print(f"\nDataset loaded successfully:")
        print(f"- {len(self.edge_indices[0]):,} edges")
        print(f"- {num_nodes:,} nodes")
        print(f"- Node features: {self.node_features.shape}")
        if self.edge_features is not None:
            print(f"- Edge features: {self.edge_features.shape}")
        print(f"- Edge indices range: [{self.edge_indices.min().item()}, {self.edge_indices.max().item()}]")
        print(f"- Number of unique nodes: {num_nodes}")
        print(f"- Timestamp range: [{self.min_time}, {self.max_time}]")
        print(f"- Number of time windows: {len(self.temporal_windows)}")
        
        # Additional validation
        if self.edge_indices.max() >= num_nodes:
            raise ValueError(f"Edge indices contain invalid node IDs. Max index: {self.edge_indices.max()}, num_nodes: {num_nodes}")
        
    def _process_edge_features(self, edge_types):
        """Process edge type features with temporal information."""
        print(f"Loading edge type features from {self.edge_type_path}...")
        edge_type_df = pd.read_csv(self.edge_type_path)
        
        # Create feature matrix
        unique_types = np.unique(edge_types)
        edge_type_features = np.zeros((len(edge_types), edge_type_df.shape[1] - 1), dtype=np.float32)
        
        for type_id in tqdm(unique_types, desc="Processing edge types"):
            mask = edge_types == type_id
            if mask.any():
                features = edge_type_df[edge_type_df.iloc[:, 0] == type_id].iloc[:, 1:].values
                if len(features) > 0:
                    edge_type_features[mask] = features[0]
        
        return np.concatenate([np.expand_dims(edge_types, axis=1), edge_type_features], axis=1)
    
    def _process_node_features(self, num_nodes, device):
        """Process node features with temporal information."""
        try:
            if self.node_feature_path and os.path.exists(self.node_feature_path):
                node_df = pd.read_csv(self.node_feature_path)
                node_features = self._create_node_feature_matrix(node_df, num_nodes)
                self.node_features = torch.from_numpy(node_features).to(device)
            else:
                self.node_features = torch.eye(num_nodes, dtype=torch.float32, device=device)
        except Exception as e:
            print(f"\nWarning: Using default node features. Error: {e}")
            self.node_features = torch.eye(num_nodes, dtype=torch.float32, device=device)
    
    def _create_node_feature_matrix(self, node_df, num_nodes):
        """Create node feature matrix with proper handling of temporal features."""
        node_features = node_df.values
        if len(node_features) != num_nodes:
            if len(node_features) < num_nodes:
                pad_size = num_nodes - len(node_features)
                node_features = np.pad(node_features, ((0, pad_size), (0, 0)), mode='constant')
            else:
                node_features = node_features[:num_nodes]
        
        feature_dims = []
        for j in range(node_features.shape[1]):
            feature = node_features[:, j]
            unique_vals = np.unique(feature[feature != -1])
            feature_dims.append(len(unique_vals) + 1)
        
        total_dims = sum(feature_dims)
        processed_features = np.zeros((num_nodes, total_dims), dtype=np.float32)
        
        current_dim = 0
        for j in range(node_features.shape[1]):
            feature = node_features[:, j]
            unique_vals = np.unique(feature[feature != -1])
            n_categories = feature_dims[j]
            
            mask_missing = feature == -1
            processed_features[mask_missing, current_dim + n_categories - 1] = 1
            
            valid_mask = ~mask_missing
            if valid_mask.any():
                val_to_idx = {val: idx for idx, val in enumerate(unique_vals)}
                valid_indices = np.where(valid_mask)[0]
                valid_values = feature[valid_mask]
                for idx, val in zip(valid_indices, valid_values):
                    processed_features[idx, current_dim + val_to_idx[val]] = 1
            
            current_dim += n_categories
        
        return processed_features
        
    def preprocess(self):
        """
        Preprocess the loaded data with temporal awareness.
        """
        print("Preprocessing data...")
        
        print("Processing temporal features...")
        with torch.cuda.amp.autocast():
            # Normalize timestamps to [0, 1] while preserving temporal gaps
            time_range = self.max_time - self.min_time
            self.edge_timestamps = (self.edge_timestamps - self.min_time) / time_range
            
            # Create temporal features if they don't exist
            if self.edge_features is None:
                self.edge_features = self._create_temporal_features()
            else:
                # Add temporal features to existing features
                time_features = self._create_temporal_features()
                self.edge_features = torch.cat([self.edge_features, time_features], dim=1)
        
        print("Preprocessing complete")
    
    def _create_temporal_features(self):
        """
        Create temporal features using vectorized operations.
        Returns:
            torch.Tensor: Temporal features tensor
        """
        timestamps = self.edge_timestamps.numpy()
        
        # Convert timestamps to datetime using vectorized operations
        # Convert float32 to float64 for datetime conversion
        timestamps_float = timestamps.astype(np.float64)
        
        # Get integer and fractional parts
        int_timestamps = timestamps_float.astype(np.int64)
        
        # Vectorized datetime conversion
        dates = np.array([datetime.fromtimestamp(ts) for ts in int_timestamps])
        
        # Extract temporal features efficiently using vectorized operations
        hour_of_day = np.array([d.hour for d in dates], dtype=np.float32) / 24.0
        day_of_week = np.array([d.weekday() for d in dates], dtype=np.float32) / 7.0
        month = np.array([d.month for d in dates], dtype=np.float32) / 12.0
        
        # Time since first event (normalized)
        time_since_start = (timestamps - timestamps.min()) / (timestamps.max() - timestamps.min())
        
        # Combine features and return tensor
        return torch.from_numpy(
            np.stack([
                hour_of_day,
                day_of_week,
                month,
                time_since_start
            ], axis=1)
        ).float()