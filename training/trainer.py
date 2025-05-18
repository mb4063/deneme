import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from tqdm import tqdm
from scipy.sparse import csr_matrix

class TemporalLinkPredictionTrainer:
    """
    Trainer for temporal link prediction.
    """
    
    def __init__(self, model, lr=0.001, weight_decay=5e-4):
        """
        Initialize the trainer.
        
        Args:
            model (nn.Module): Model to train
            lr (float): Learning rate
            weight_decay (float): Weight decay
        """
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.criterion = nn.BCELoss()
        
    def train(self, dataset, num_epochs, batch_size):
        self.model.train()
        
        for epoch in range(num_epochs):
            h, c = None, None  # Initialize LSTM hidden states
            total_loss = 0
            
            # Iterate over temporal snapshots
            for time_idx, snapshot in enumerate(dataset.temporal_signal):
                # Get data for current timestamp
                x = snapshot.x
                edge_index = snapshot.edge_index
                edge_weight = snapshot.edge_attr
                target = snapshot.y
                
                # Forward pass
                out, h, c = self.model(x, edge_index, edge_weight, h, c)
                
                # Predict links
                pred = self.model.predict_link(out, edge_index)
                
                # Calculate loss
                loss = self.criterion(pred, target)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
            
            # Print epoch statistics
            avg_loss = total_loss / (time_idx + 1)
            print(f"Epoch {epoch}: Average Loss = {avg_loss:.4f}")
    
    def _generate_link_prediction_samples(self, dataset):
        """
        Generate positive and negative samples for link prediction with temporal awareness.
        """
        print("Preparing temporal splits...")
        
        # Sort edges by timestamp (should already be sorted, but verify)
        sorted_indices = torch.argsort(dataset.edge_timestamps)
        sorted_edge_indices = dataset.edge_indices[:, sorted_indices]
        sorted_timestamps = dataset.edge_timestamps[sorted_indices]
        sorted_targets = dataset.targets[sorted_indices]
        
        # Split data by time
        print("Creating temporal splits...")
        n_edges = sorted_edge_indices.size(1)
        train_ratio, val_ratio = 0.7, 0.15
        
        train_size = int(n_edges * train_ratio)
        val_size = int(n_edges * val_ratio)
        
        # Split ensuring temporal order
        train_end_time = sorted_timestamps[train_size].item()
        val_end_time = sorted_timestamps[train_size + val_size].item()
        
        print(f"Time-based split points:")
        print(f"- Train end time: {train_end_time}")
        print(f"- Validation end time: {val_end_time}")
        
        # Create splits
        train_edges = sorted_edge_indices[:, :train_size]
        train_times = sorted_timestamps[:train_size]
        train_targets = sorted_targets[:train_size]
        
        val_edges = sorted_edge_indices[:, train_size:train_size+val_size]
        val_times = sorted_timestamps[train_size:train_size+val_size]
        val_targets = sorted_targets[train_size:train_size+val_size]
        
        test_edges = sorted_edge_indices[:, train_size+val_size:]
        test_times = sorted_timestamps[train_size+val_size:]
        test_targets = sorted_targets[train_size+val_size:]
        
        # Generate negative samples with temporal awareness
        print("\nGenerating temporally-aware negative samples...")
        print("- Train set...")
        train_neg = self._generate_temporal_negative_samples(
            train_edges, train_size, sorted_timestamps[:train_size]
        )
        
        print("- Validation set...")
        val_neg = self._generate_temporal_negative_samples(
            val_edges, val_size, sorted_timestamps[train_size:train_size+val_size],
            known_edges=train_edges  # Only use training edges as known
        )
        
        print("- Test set...")
        test_neg = self._generate_temporal_negative_samples(
            test_edges, len(test_edges[0]), sorted_timestamps[train_size+val_size:],
            known_edges=torch.cat([train_edges, val_edges], dim=1)  # Use both train and val as known
        )
        
        # Create samples
        train_samples = (train_edges.t(), train_neg.t(), train_times)
        val_samples = (val_edges.t(), val_neg.t(), val_times)
        test_samples = (test_edges.t(), test_neg.t(), test_times)
        
        return train_samples, val_samples, test_samples
    
    def _generate_temporal_negative_samples(self, positive_edges, num_samples, timestamps, known_edges=None):
        """
        Generate negative samples with temporal awareness using fully vectorized operations.
        """
        device = positive_edges.device
        print(f"Generating {num_samples:,} negative samples...")
        
        # Move operations to CPU for faster set operations
        positive_edges = positive_edges.cpu().numpy()
        if known_edges is not None:
            known_edges = known_edges.cpu().numpy()
            
        # Move timestamps to CPU if they're on GPU
        if timestamps.is_cuda:
            timestamps = timestamps.cpu()
        timestamps_np = timestamps.numpy()
        
        # Get unique nodes and create time-aware sampling weights
        edge_index = positive_edges.T
        all_nodes = np.unique(edge_index.flatten())
        num_nodes = max(all_nodes) + 1  # Use max node ID + 1 for matrix size
        print(f"Processing {len(all_nodes):,} unique nodes (matrix size: {num_nodes})")
        
        # Vectorized temporal weight calculation
        print("Computing temporal weights...")
        current_time = timestamps_np.max()
        time_diff = current_time - timestamps_np
        time_weights = np.exp(-0.1 * time_diff)  # Decay factor of 0.1
        
        # Create sparse temporal weight matrix
        print("Building temporal weight matrix...")
        rows = np.concatenate([edge_index[:, 0], edge_index[:, 1]])
        cols = np.concatenate([edge_index[:, 1], edge_index[:, 0]])
        weights = np.concatenate([time_weights, time_weights])
        
        # Validate indices before creating sparse matrix
        assert rows.max() < num_nodes, f"Row index {rows.max()} >= num_nodes {num_nodes}"
        assert cols.max() < num_nodes, f"Col index {cols.max()} >= num_nodes {num_nodes}"
        
        # Aggregate weights using sparse matrix operations
        weight_matrix = csr_matrix(
            (weights, (rows, cols)),
            shape=(num_nodes, num_nodes)
        )
        
        # Compute node sampling weights
        print("Computing node sampling weights...")
        node_weights = np.array(weight_matrix.sum(axis=1)).flatten()
        node_degrees = np.array(weight_matrix.getnnz(axis=1)).flatten()
        
        # Normalize weights
        node_weights = np.divide(
            node_weights,
            node_degrees,
            out=np.zeros_like(node_weights),
            where=node_degrees != 0
        )
        
        # Compute sampling weights only for nodes that appear in edges
        sampling_weights = np.zeros(num_nodes)
        sampling_weights[all_nodes] = 1.0 / (node_weights[all_nodes] + 1)  # +1 to avoid division by zero
        sampling_weights = sampling_weights / sampling_weights.sum()
        
        # Create edge existence mask matrix
        print("Creating edge mask matrix...")
        edge_mask = weight_matrix.astype(bool)
        if known_edges is not None:
            known_edge_mask = csr_matrix(
                (np.ones(len(known_edges)), (known_edges[:, 0], known_edges[:, 1])),
                shape=(num_nodes, num_nodes)
            )
            edge_mask = edge_mask + known_edge_mask
        
        # Generate negative samples efficiently
        print("Sampling negative edges...")
        negative_edges = []
        total_sampled = 0
        batch_size = min(num_samples * 2, 1000000)
        
        with tqdm(total=num_samples, desc="Generating edges") as pbar:
            while total_sampled < num_samples:
                # Sample nodes based on weights
                src = np.random.choice(all_nodes, batch_size, p=sampling_weights[all_nodes])
                dst = np.random.choice(all_nodes, batch_size, p=sampling_weights[all_nodes])
                
                # Remove self-loops
                no_self_loops = src != dst
                src = src[no_self_loops]
                dst = dst[no_self_loops]
                
                if len(src) == 0:
                    continue
                
                # Check edge existence using sparse matrix
                # Convert to COO format for faster indexing
                edge_mask_coo = edge_mask.tocoo()
                existing_edges = set(zip(edge_mask_coo.row, edge_mask_coo.col))
                
                # Vectorized edge checking
                candidate_edges = list(zip(src, dst))
                valid_edges = [(s, d) for s, d in candidate_edges 
                             if (s, d) not in existing_edges and (d, s) not in existing_edges]
                
                if valid_edges:
                    # Convert valid edges to array
                    valid_edges = np.array(valid_edges)
                    edges_to_add = min(len(valid_edges), num_samples - total_sampled)
                    negative_edges.append(valid_edges[:edges_to_add])
                    total_sampled += edges_to_add
                    pbar.update(edges_to_add)
                
                if total_sampled >= num_samples:
                    break
        
        # Combine and convert to tensor
        negative_edges = np.vstack(negative_edges)[:num_samples]
        return torch.from_numpy(negative_edges.T).to(device)
    
    def _create_data_loader(self, samples, batch_size, pin_memory=True, num_workers=4):
        """
        Create an optimized data loader for GPU training.
        """
        pos_edge_index, neg_edge_index, timestamps = samples
        
        # Calculate optimal batch size based on data size and A5000 memory
        num_samples = pos_edge_index.size(0)
        # Optimize for higher VRAM usage (targeting ~18GB VRAM usage)
        optimal_batch_size = max(num_samples // 300, 8192)  # Reduced to 300 batches for higher memory utilization
        
        print(f"\nDataloader settings:")
        print(f"- Total samples: {num_samples}")
        print(f"- Batch size: {optimal_batch_size}")
        print(f"- Number of batches: {num_samples // optimal_batch_size}")
        
        # Pre-allocate and reserve CUDA memory
        torch.cuda.empty_cache()
        torch.cuda.memory.empty_cache()
        if torch.cuda.is_available():
            # Reserve 75% of VRAM for training
            torch.cuda.set_per_process_memory_fraction(0.75)
        
        # Move tensors to CPU and pin memory
        pos_edge_index = pos_edge_index.cpu()
        neg_edge_index = neg_edge_index.cpu()
        timestamps = timestamps.cpu()
        
        # Create dataset with memory optimization
        dataset = TensorDataset(pos_edge_index, neg_edge_index, timestamps)
        
        # Create data loader with optimized settings for A5000
        data_loader = DataLoader(
            dataset,
            batch_size=optimal_batch_size,
            shuffle=True,
            pin_memory=True,  # Always use pinned memory
            num_workers=16,  # Increased for better CPU utilization
            persistent_workers=True,
            prefetch_factor=8,  # Increased prefetch for smoother GPU feeding
            drop_last=False  # Keep all samples
        )
        
        return data_loader
    
    def _move_batch_to_device(self, batch, device):
        """
        Move batch data to the specified device efficiently.
        """
        return tuple(t.to(device, non_blocking=True) if torch.is_tensor(t) else t for t in batch)