import torch
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from torch_geometric.data import Data

# Define ForecastingData inheriting from torch_geometric.data.Data
class ForecastingData(Data):
    def __init__(self, x=None, edge_index=None, edge_attr=None, 
                 target_positive_edge_index=None, 
                 num_nodes_for_target_context=None, # Number of nodes for the context of target_positive_edge_index
                 **kwargs):
        super().__init__(x=x, edge_index=edge_index, edge_attr=edge_attr, **kwargs)
        self.target_positive_edge_index = target_positive_edge_index
        # This stores the number of nodes relevant for the target_positive_edge_index. 
        # In our case, since node features and indices are global, this is self.num_total_nodes from the dataset.
        self.num_nodes_for_target_context = num_nodes_for_target_context

    def __inc__(self, key, value, *args, **kwargs):
        # Node indices in edge_index and target_positive_edge_index are global (0 to num_total_nodes-1).
        # So, they should not be incremented by the number of nodes in individual graphs during batching.
        if key == 'edge_index' or key == 'target_positive_edge_index':
            return 0 # Do not increment source/target node indices in these tensors.
        # For other keys, use the default increment behavior from the parent Data class
        return super().__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key, value, *args, **kwargs):
        # num_nodes_for_target_context is a scalar-like attribute per graph.
        # When batching, we want it to be collated into a tensor [batch_size] or similar,
        # not concatenated along a dimension if it were, say, a multi-element tensor itself.
        # If it's a simple Python int/float or a 0-dim tensor, default collation might be fine (becomes a list or 1D tensor).
        # If it's a tensor(scalar), it will be stacked. If it's a python number, it becomes a list.
        if key == 'num_nodes_for_target_context':
            return None # Let PyG handle it by creating a list; or stack if it's a tensor.
        return super().__cat_dim__(key, value, *args, **kwargs)

class TemporalSignal:
    """Custom temporal signal class for forecasting data (list of ForecastingData objects)."""
    def __init__(self, forecasting_data_list):
        self.instances = forecasting_data_list # Should be a list of ForecastingData objects
        self.t = 0
        self.length = len(forecasting_data_list)
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        return self.instances[idx]
    
    def __iter__(self):
        self.t = 0
        return self
    
    def __next__(self):
        if self.t < self.length:
            instance = self.instances[self.t]
            self.t += 1
            return instance
        else:
            raise StopIteration

class EventBasedDataset:
    """
    Dataset class for event-based temporal data, preparing forecasting instances.
    """
    
    def __init__(self, name, time_window=None): # time_window not directly used yet
        """
        Initialize the dataset.
        
        Args:
            name (str): Name of the dataset
            time_window (int, optional): Not actively used in current 1-step forecast.
        """
        self.name = name
        # self.time_window = time_window # Store if needed later
        self.temporal_signal = None
        self.node_feature_path = "data/node_features.csv"
        # self.edge_type_path = "data/edge_type_features.csv" # Not used
        self.node_encoder = LabelEncoder() # Store encoder for consistent mapping
        self.num_total_nodes = 0 # Store total number of unique nodes
        
    def load_data(self, data_path):
        """
        Load data and prepare forecasting instances (input_graph_at_t-1, target_links_at_t).
        """
        print(f"Loading data from {data_path}")
        
        df = pd.read_csv(
            data_path,
            header=None,
            names=['source', 'target', 'edge_type', 'timestamp'],
            dtype={
                'source': np.int32,
                'target': np.int32,
                'edge_type': np.int32, # Assuming edge_type is used as edge_attr
                'timestamp': np.int64
            },
            engine='c',
            memory_map=True
        )
        
        print(f"Loaded {len(df):,} edges")
        
        print("Processing node IDs...")
        all_nodes_series = pd.concat([df['source'], df['target']]).unique()
        self.node_encoder.fit(all_nodes_series)
        self.num_total_nodes = len(self.node_encoder.classes_)
        
        df['source_mapped'] = self.node_encoder.transform(df['source'])
        df['target_mapped'] = self.node_encoder.transform(df['target'])
        
        df.sort_values('timestamp', inplace=True)
        unique_timestamps = df['timestamp'].unique()
        
        print(f"Time range: {unique_timestamps.min()} to {unique_timestamps.max()}")
        print(f"Number of unique nodes (total): {self.num_total_nodes:,}")
        print(f"Number of unique timestamps: {len(unique_timestamps):,}")

        static_node_features = self._process_node_features(self.num_total_nodes)
        
        forecasting_instances = []
        
        print("Processing temporal data into forecasting instances...")
        # We need at least two timestamps to form an (input, target) pair
        if len(unique_timestamps) < 2:
            print("Not enough timestamps to create forecasting instances. Need at least 2.")
            self.temporal_signal = TemporalSignal([]) # Empty signal
            return

        for i in tqdm(range(len(unique_timestamps) - 1), desc="Creating Forecasting Instances"):
            input_ts = unique_timestamps[i]
            target_ts = unique_timestamps[i+1]
            
            # --- Input Graph Data (at input_ts) ---
            input_edges_df = df[df['timestamp'] == input_ts]
            if input_edges_df.empty: # Handle snapshots with no edges
                input_edge_index_t = torch.empty((2,0), dtype=torch.long)
                input_edge_attr_t = torch.empty((0,), dtype=torch.float) # Assuming edge_type is attr
            else:
                input_edge_index_t = torch.from_numpy(
                    input_edges_df[['source_mapped', 'target_mapped']].values.T
                ).long()
                input_edge_attr_t = torch.from_numpy(
                    input_edges_df['edge_type'].values
                ).float() # Using edge_type as attribute
            
            # Node features for input are the static features for all nodes
            # input_x_t = static_node_features (Now passed as x to ForecastingData)

            # --- Target Links Data (at target_ts) ---
            target_links_df = df[df['timestamp'] == target_ts]
            if target_links_df.empty:
                target_positive_edge_index_t = torch.empty((2,0), dtype=torch.long)
            else:
                target_positive_edge_index_t = torch.from_numpy(
                    target_links_df[['source_mapped', 'target_mapped']].values.T
                ).long()

            # Node features for target (mainly for num_nodes, assuming static features for now)
            # target_x_for_num_nodes_t = static_node_features (Info captured by num_total_nodes)
            
            forecasting_instances.append(ForecastingData(
                x=static_node_features, # Node features for input graph (global)
                edge_index=input_edge_index_t, # Edges for input graph (global indices)
                edge_attr=input_edge_attr_t, # Attributes for input graph edges
                target_positive_edge_index=target_positive_edge_index_t, # Target positive edges (global indices)
                num_nodes_for_target_context=torch.tensor([self.num_total_nodes], dtype=torch.long) # Pass as a tensor for consistent batching
            ))
            
        self.temporal_signal = TemporalSignal(forecasting_instances)
        
        print("\nDataset Summary (Forecasting Mode):")
        if len(forecasting_instances) > 0:
            avg_input_edges = np.mean([inst.edge_index.size(1) for inst in forecasting_instances])
            avg_target_edges = np.mean([inst.target_positive_edge_index.size(1) for inst in forecasting_instances])
            print(f"Number of forecasting instances: {len(forecasting_instances):,}")
            print(f"Number of nodes: {self.num_total_nodes:,}") # Based on all unique nodes ever seen
            print(f"Average input edges per instance: {avg_input_edges:,.2f}")
            print(f"Average target positive edges per instance: {avg_target_edges:,.2f}")
            print(f"Node feature dimensions: {static_node_features.shape}")
        else:
            print("No forecasting instances were created.")

    def _process_node_features(self, num_total_nodes):
        """Vectorized node feature processing for the total number of unique nodes."""
        # --- FORCED DEBUGGING: Always use one-hot encoding --- 
        # print(f"[DEBUG] FORCING one-hot encoding for {num_total_nodes} nodes, ignoring {self.node_feature_path}.")
        # return torch.eye(num_total_nodes, dtype=torch.float32)
        # --- END FORCED DEBUGGING ---

        # Original logic (commented out for debugging):
        try:
            if os.path.exists(self.node_feature_path):
                node_df = pd.read_csv(
                    self.node_feature_path,
                    memory_map=True,
                    engine='c'
                )
                # Check for NaNs in the DataFrame immediately after loading
                if node_df.isnull().values.any():
                    print(f"Warning: NaNs found in {self.node_feature_path} after loading. Attempting to fill with 0.")
                    node_df.fillna(0, inplace=True) # Fill NaNs with 0

                features_np = node_df.values
                
                # Check for NaNs in the numpy array (e.g. if non-numeric data caused NaNs not caught by fillna on mixed-type df)
                if np.isnan(features_np).any():
                    print(f"Warning: NaNs found in features_np from {self.node_feature_path}. Replacing with 0.")
                    features_np = np.nan_to_num(features_np, nan=0.0) # Replace NaNs with 0

                # Ensure features have the correct number of rows (num_total_nodes)
                if features_np.shape[0] < num_total_nodes:
                    padding = np.zeros((num_total_nodes - features_np.shape[0], features_np.shape[1]))
                    features_np = np.vstack([features_np, padding])
                elif features_np.shape[0] > num_total_nodes:
                     print(f"Warning: Node features file has {features_np.shape[0]} rows, but only {num_total_nodes} unique nodes found in edges. Truncating features.")
                     features_np = features_np[:num_total_nodes, :]
                
                features = torch.from_numpy(features_np).float()
                return features
            else:
                print(f"No node features file found at '{self.node_feature_path}', using one-hot encodings for {num_total_nodes} nodes.")
                return torch.eye(num_total_nodes, dtype=torch.float32)
        except Exception as e:
            print(f"Warning: Using default one-hot node features due to error: {e}")
            return torch.eye(num_total_nodes, dtype=torch.float32)

# Example usage (optional, for testing)
if __name__ == '__main__':
    # Create a dummy CSV for testing
    dummy_data = {
        'source': [0, 0, 1, 2, 0, 3, 3, 4, 1],
        'target': [1, 2, 2, 0, 3, 1, 4, 1, 4],
        'edge_type': [0, 1, 0, 0, 1, 0, 1, 0, 0],
        'timestamp': [100, 100, 100, 100, 200, 200, 300, 300, 300]
    }
    dummy_df = pd.DataFrame(dummy_data)
    dummy_csv_path = 'dummy_edges.csv'
    dummy_df.to_csv(dummy_csv_path, index=False, header=False)

    # Test dataset loading
    dataset = EventBasedDataset(name='dummy_test')
    dataset.load_data(dummy_csv_path)

    if dataset.temporal_signal and len(dataset.temporal_signal) > 0:
        print(f"\nFirst forecasting instance:")
        first_instance = dataset.temporal_signal[0]
        print(f"  Input X shape: {first_instance.x.shape}")
        print(f"  Input Edge Index shape: {first_instance.edge_index.shape}")
        print(f"  Input Edge Attr shape: {first_instance.edge_attr.shape if first_instance.edge_attr is not None else 'None'}")
        print(f"  Target Positive Edge Index shape: {first_instance.target_positive_edge_index.shape}")
        
        print("\nIterating through temporal signal:")
        for i, instance in enumerate(dataset.temporal_signal):
            print(f"Instance {i}: Input edges={instance.edge_index.size(1)}, Target positive edges={instance.target_positive_edge_index.size(1)}")
    else:
        print("No instances in temporal signal.")
        
    # Clean up dummy file
    if os.path.exists(dummy_csv_path):
        os.remove(dummy_csv_path)