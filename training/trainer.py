import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from tqdm.auto import tqdm
from scipy.sparse import csr_matrix

# For AMP
from torch.cuda.amp import GradScaler, autocast

class TemporalLinkPredictionTrainer:
    """
    Trainer for temporal link prediction, now using batched graph data.
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
        self.criterion = nn.BCEWithLogitsLoss()
        self.scaler = GradScaler() # Initialize GradScaler for AMP
        self.max_grad_norm = 1.0  # Add gradient clipping threshold
        
    def train(self, train_loader, val_loader, test_loader, num_epochs, patience=10, checkpoint_dir=None, neg_sampling_ratio=1):
        """
        Train the model with early stopping and checkpointing using batched ForecastingData.
        
        Args:
            train_loader: PyG DataLoader for training.
            val_loader: PyG DataLoader for validation.
            test_loader: PyG DataLoader for final testing.
            num_epochs (int): Number of epochs to train.
            patience (int): Number of epochs to wait for improvement before early stopping (based on val_loss).
            checkpoint_dir (str): Directory to save model checkpoints.
            neg_sampling_ratio (int): Ratio of negative to positive samples.
        """
        self.model.train() 
        best_val_loss_for_early_stopping = float('inf') # For early stopping based on loss
        best_val_f1_for_checkpointing = float('-inf')  # For checkpointing based on F1
        patience_counter = 0
        history = {
            'train_loss': [], 
            'val_loss': [], 'val_auc': [], 'val_ap': [], 'val_f1': []
        }
        device = next(self.model.parameters()).device
        
        for epoch in tqdm(range(num_epochs), desc="Epochs"):
            self.model.train()
            total_train_loss = 0
            batch_count = 0
            
            # train_loader now yields Batch objects
            for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{num_epochs}", leave=False)):
                try:
                    batch = batch.to(device)
                    
                    if batch.target_positive_edge_index.size(1) == 0:
                        continue

                    self.optimizer.zero_grad(set_to_none=True)

                    with autocast(): 
                        node_embeddings = self.model(batch.x, batch.edge_index, batch.edge_attr) 
                        
                        pos_pred_logits = self.model.predict_link(node_embeddings, batch.target_positive_edge_index)
                        pos_target = torch.ones_like(pos_pred_logits)

                        num_nodes_global_context = batch.num_nodes_for_target_context[0].item()
                        neg_edge_index = self._generate_negative_edges_for_snapshot(
                            batch.target_positive_edge_index, 
                            num_nodes_global_context,
                            num_negative_ratio=neg_sampling_ratio 
                        )
                        
                        current_snapshot_logits = [pos_pred_logits]
                        current_snapshot_targets = [pos_target]

                        if neg_edge_index.size(1) > 0:
                            neg_pred_logits = self.model.predict_link(node_embeddings, neg_edge_index)
                            neg_target = torch.zeros_like(neg_pred_logits)
                            current_snapshot_logits.append(neg_pred_logits)
                            current_snapshot_targets.append(neg_target)
                        
                        combined_logits = torch.cat(current_snapshot_logits)
                        combined_targets = torch.cat(current_snapshot_targets)
                        
                        if combined_logits.numel() == 0:
                            continue
                        
                        loss = self.criterion(combined_logits, combined_targets.to(combined_logits.dtype))
                    
                    # Scale loss and backprop
                    self.scaler.scale(loss).backward()
                    
                    # Unscale before gradient clipping
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    
                    total_train_loss += loss.item()
                    batch_count += 1
                    
                    # Clear memory more aggressively
                    del node_embeddings, pos_pred_logits, neg_edge_index
                    del current_snapshot_logits, current_snapshot_targets
                    del combined_logits, combined_targets, loss
                    torch.cuda.empty_cache()
                    
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print(f"WARNING: Out of memory in batch {batch_idx}. Skipping batch.")
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        continue
                    else:
                        raise e
            
            avg_train_loss = total_train_loss / batch_count if batch_count > 0 else 0
            history['train_loss'].append(avg_train_loss)
            
            # Validation phase (pass val_loader)
            val_loss, val_auc, val_ap, val_f1 = self.evaluate(val_loader, neg_sampling_ratio=neg_sampling_ratio)
            history['val_loss'].append(val_loss)
            history['val_auc'].append(val_auc)
            history['val_ap'].append(val_ap)
            history['val_f1'].append(val_f1)
            
            print(f"Epoch {epoch + 1}/{num_epochs}: Train Loss = {avg_train_loss:.4f}, Val Loss = {val_loss:.4f}, Val AUC = {val_auc:.4f}, Val AP = {val_ap:.4f}, Val F1 = {val_f1:.4f}")
            
            # Checkpoint saving based on best validation F1
            if val_f1 > best_val_f1_for_checkpointing:
                best_val_f1_for_checkpointing = val_f1
                if checkpoint_dir is not None:
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    checkpoint_path = os.path.join(checkpoint_dir, f'model_best.pt') # Overwrites if new best F1
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'val_loss': val_loss, # Save relevant metrics at this checkpoint
                        'val_f1': val_f1,
                        'val_auc': val_auc,
                        'val_ap': val_ap
                    }, checkpoint_path)
                    print(f"Saved best model checkpoint to {checkpoint_path} (Val F1: {best_val_f1_for_checkpointing:.4f}, Val Loss: {val_loss:.4f})")

            # Early stopping check based on validation loss
            if val_loss < best_val_loss_for_early_stopping:
                best_val_loss_for_early_stopping = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered after {epoch + 1} epochs due to no improvement in validation loss for {patience} epochs.")
                    break
        
        if checkpoint_dir is not None:
            best_model_path = os.path.join(checkpoint_dir, 'model_best.pt')
            if os.path.exists(best_model_path):
                print(f"Loading best model from {best_model_path} for final testing.")
                checkpoint = torch.load(best_model_path, map_location=device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                print("No best model checkpoint found for final testing. Using last model state.")

        print("\nEvaluating on the test set...")
        test_loss, test_auc, test_ap, test_f1 = self.evaluate(test_loader, neg_sampling_ratio=neg_sampling_ratio)
        print(f"Test Metrics: Loss = {test_loss:.4f}, AUC = {test_auc:.4f}, AP = {test_ap:.4f}, F1 = {test_f1:.4f}")
        
        test_metrics_dict = {
            'loss': test_loss,
            'auc': test_auc,
            'ap': test_ap,
            'f1': test_f1
        }
        
        return history, test_metrics_dict
    
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

    def _prepare_batch(self, batch_edges, batch_times, device):
        """Vectorized batch preparation"""
        # Stack edges and times for parallel processing
        edges = torch.stack(batch_edges)
        times = torch.stack(batch_times) if batch_times is not None else None
        
        # Move to device in one operation
        edges = edges.to(device)
        if times is not None:
            times = times.to(device)
        
        return edges, times

    def _compute_loss_batch(self, pred, target):
        """Vectorized loss computation"""
        # Use torch.nn.functional for optimized operations
        return torch.nn.functional.binary_cross_entropy_with_logits(
            pred.view(-1),
            target.view(-1),
            reduction='mean'
        )

    def _generate_negative_samples(self, edge_index, num_nodes, num_samples):
        """Vectorized negative sampling"""
        # Create edge set for O(1) lookup
        edge_set = {(i.item(), j.item()) for i, j in edge_index.t()}
        
        # Generate candidate edges in parallel
        src = torch.randint(0, num_nodes, (num_samples,))
        dst = torch.randint(0, num_nodes, (num_samples,))
        
        # Vectorized filtering of invalid edges
        mask = torch.tensor([
            (s.item(), d.item()) not in edge_set and s != d
            for s, d in zip(src, dst)
        ])
        
        return torch.stack([src[mask], dst[mask]])

    def train_epoch(self, dataset, batch_size, device):
        """Vectorized training epoch"""
        self.model.train()
        total_loss = 0
        
        # Process data in batches
        num_batches = len(dataset) // batch_size
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = start_idx + batch_size
            
            # Get batch data
            batch_edges = dataset.edge_indices[start_idx:end_idx]
            batch_weights = dataset.edge_weights[start_idx:end_idx]
            
            # Process batch in parallel
            edges, weights = self._prepare_batch(batch_edges, batch_weights, device)
            
            # Forward pass
            self.optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                out = self.model(dataset.features, edges, weights)
                loss = self._compute_loss_batch(out, dataset.targets[start_idx:end_idx])
            
            # Backward pass with gradient scaling
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            total_loss += loss.item()
        
        return total_loss / num_batches

    def _generate_negative_edges_for_snapshot(self, positive_edge_index, num_nodes, num_negative_ratio=1):
        """
        Generates negative edges for a single snapshot (now, target snapshot in forecasting).
        Ensures no overlap with positive_edge_index and avoids self-loops.
        Args:
            positive_edge_index (torch.Tensor): Tensor of positive edges [2, num_positive_edges].
            num_nodes (int): Number of nodes in the snapshot.
            num_negative_ratio (int): Ratio of negative to positive samples.
        Returns:
            torch.Tensor: Tensor of negative edges [2, num_negative_samples].
        """
        device = positive_edge_index.device
        num_positive_edges = positive_edge_index.size(1)
        num_negative_samples_to_generate = num_positive_edges * num_negative_ratio

        if num_nodes <= 1 or num_negative_samples_to_generate == 0:
            return torch.empty((2, 0), dtype=torch.long, device=device)

        # Maximum possible edges (excluding self-loops) is num_nodes * (num_nodes - 1)
        # If directed, it's num_nodes * (num_nodes - 1). If undirected, num_nodes * (num_nodes - 1) / 2.
        # For negative sampling, we consider directed pairs and filter later if model assumes undirected.
        max_possible_edges = num_nodes * (num_nodes - 1)
        if num_positive_edges >= max_possible_edges: # Graph is complete or over-specified
            return torch.empty((2, 0), dtype=torch.long, device=device)
        
        # Adjust if requested negative samples exceed available non-edges
        num_negative_samples_to_generate = min(num_negative_samples_to_generate, max_possible_edges - num_positive_edges)
        if num_negative_samples_to_generate <= 0:
            return torch.empty((2, 0), dtype=torch.long, device=device)

        neg_edge_list = []
        
        # Create a set of positive edges for fast lookups (on CPU for efficiency with Python sets)
        # Consider (u,v) and (v,u) if the graph is treated as undirected by GAT's add_self_loops or similar
        # For now, strictly use the given positive_edge_index for exclusion.
        positive_set = set()
        pos_cpu = positive_edge_index.cpu().numpy()
        for i in range(pos_cpu.shape[1]):
            positive_set.add(tuple(pos_cpu[:, i]))
            # If your model/GAT layer implies undirected edges or adds reverse edges,
            # you might want to add positive_set.add((pos_cpu[1, i], pos_cpu[0, i])) as well.

        attempts = 0
        max_attempts_factor = 5 # Allow more attempts in sparse graphs
        max_attempts = num_negative_samples_to_generate * max_attempts_factor + 1000 # Heuristic

        while len(neg_edge_list) < num_negative_samples_to_generate and attempts < max_attempts:
            attempts += 1
            # Generate random pairs
            src = torch.randint(0, num_nodes, (num_negative_samples_to_generate - len(neg_edge_list),), device=device)
            dst = torch.randint(0, num_nodes, (num_negative_samples_to_generate - len(neg_edge_list),), device=device)

            for i in range(src.size(0)):
                u, v = src[i].item(), dst[i].item()
                if u == v:  # No self-loops
                    continue
                # Check if edge (u,v) is not in positive_set
                # Also check if (v,u) is not in positive_set if treating as undirected for negative sampling
                # This current check is for directed negative sampling.
                if (u,v) not in positive_set: # and (v,u) not in positive_set (if undirected consideration for neg)
                    neg_edge_list.append([u,v])
                    positive_set.add((u,v)) # Add to positive_set to avoid re-sampling this negative
                    if len(neg_edge_list) >= num_negative_samples_to_generate:
                        break
            if len(neg_edge_list) >= num_negative_samples_to_generate:
                 break
        
        if not neg_edge_list:
            return torch.empty((2,0), dtype=torch.long, device=device)
            
        return torch.tensor(neg_edge_list, dtype=torch.long, device=device).t().contiguous()

    def evaluate(self, eval_loader, neg_sampling_ratio=1):
        """
        Evaluate the model on a given dataset using batched ForecastingData.
        
        Args:
            eval_loader: PyG DataLoader for evaluation.
            neg_sampling_ratio (int): Ratio of negative to positive samples.
        Returns:
            Tuple: (average_loss, auc, ap, f1)
        """
        self.model.eval()
        device = next(self.model.parameters()).device
        total_loss = 0
        all_logits_batch = [] # Store logits from all batches
        all_targets_batch = [] # Store targets from all batches
        
        # eval_loader yields Batch objects directly
        for batch_idx, batch in enumerate(tqdm(eval_loader, desc="Evaluating", leave=False)):
            batch = batch.to(device)
            
            if batch.target_positive_edge_index.size(1) == 0 and batch.numel() == 0: # Check if batch is effectively empty
                # This condition might be too strict or needs refinement based on how empty batches are represented.
                # If a batch has no positive edges for target, but has input graph, it should still be processed for negative sampling.
                print("DEBUG: Skipping an apparently empty batch in evaluate")
                continue
            
            node_embeddings = self.model(batch.x, batch.edge_index, batch.edge_attr)
            
            if torch.isnan(node_embeddings).any():
                print(f"DEBUG: NaNs found in node_embeddings during evaluate for batch {batch_idx}. Input x NaNs: {torch.isnan(batch.x).any()}")

            # predict_link now returns logits
            pos_pred_logits = self.model.predict_link(node_embeddings, batch.target_positive_edge_index)
            if torch.isnan(pos_pred_logits).any():
                print(f"DEBUG: NaNs found in pos_pred_logits during evaluate for batch {batch_idx}.")
            pos_target = torch.ones_like(pos_pred_logits)

            neg_edge_index = self._generate_negative_edges_for_snapshot(
                batch.target_positive_edge_index, 
                batch.num_nodes_for_target_context[0].item(),
                num_negative_ratio=neg_sampling_ratio
            )
            
            current_snapshot_logits = []
            current_snapshot_targets = []

            if batch.target_positive_edge_index.size(1) > 0:
                current_snapshot_logits.append(pos_pred_logits)
                current_snapshot_targets.append(pos_target)
            
            if neg_edge_index.size(1) > 0:
                neg_pred_logits = self.model.predict_link(node_embeddings, neg_edge_index)
                if torch.isnan(neg_pred_logits).any():
                    print(f"DEBUG: NaNs found in neg_pred_logits during evaluate for batch {batch_idx}.")
                neg_target = torch.zeros_like(neg_pred_logits)
                current_snapshot_logits.append(neg_pred_logits)
                current_snapshot_targets.append(neg_target)
            
            if not current_snapshot_logits: # If no positive and no negative samples
                # This can happen if target_positive_edge_index is empty AND no negative edges were generated.
                # print(f"DEBUG: No logits generated for batch {batch_idx} in evaluate. Skipping batch.")
                continue

            batch_logits = torch.cat(current_snapshot_logits)
            batch_targets = torch.cat(current_snapshot_targets)
            
            if batch_logits.numel() == 0:
                # print(f"DEBUG: Empty logits/targets for batch {batch_idx} after cat. Skipping batch.")
                continue

            # BCEWithLogitsLoss expects raw logits
            loss = self.criterion(batch_logits, batch_targets.to(batch_logits.dtype))
            total_loss += loss.item()
            
            all_logits_batch.append(batch_logits.detach()) # Detach before storing
            all_targets_batch.append(batch_targets.detach())
        
        if not all_logits_batch or not all_targets_batch: # No data processed
            print("Warning: No data processed during evaluation. Metrics will be zero.")
            return 0.0, 0.0, 0.0, 0.0 # Loss, AUC, AP, F1

        final_preds = torch.cat(all_logits_batch)
        final_targets = torch.cat(all_targets_batch)

        # Ensure predictions are float for metrics calculation
        # Move to CPU before converting to numpy
        final_preds_float = final_preds.cpu().float().numpy()
        final_targets_int = final_targets.cpu().long().numpy()

        # Defensive check for NaNs in predictions before metric calculation
        if np.isnan(final_preds_float).any():
            print("ERROR: NaNs found in model predictions (final_preds_float) before calculating metrics.")
            print("       Replacing NaNs with 0.5 to allow metrics computation, but the model is producing NaNs!")
            final_preds_float = np.nan_to_num(final_preds_float, nan=0.5) # Replace NaNs with a neutral value

        # Check for NaNs in targets (should not happen if data loading is correct)
        if np.isnan(final_targets_int).any():
            print("ERROR: NaNs found in targets (final_targets_int) before calculating metrics. This is unexpected!")
            # Depending on the desired behavior, one might need to filter these out or handle them.
            # For now, we let it proceed, but sklearn functions might error or produce incorrect results.
        
        try:
            # Check for single class in targets, which can lead to issues with some metrics
            unique_target_values = np.unique(final_targets_int)
            if len(unique_target_values) < 2:
                print(f"Warning: Only one class ({unique_target_values}) present in targets. AUC and AP will be set to 0.")
                auc = 0.0
                ap = 0.0
                # F1 score can still be computed, but might be 0 or 1 depending on predictions.
                # Ensure no NaNs in predictions if we reached here after the above replacement
                if np.any(np.isnan(final_preds_float)):
                     print("ERROR: NaNs still in final_preds_float despite replacement. Setting F1 to 0.")
                     f1 = 0.0
                else:
                     f1 = f1_score(final_targets_int, (final_preds_float >= 0.5).astype(int), zero_division=0)
            else:
                # Proceed with metric calculation if there are at least two classes in targets
                # and ensure predictions are not NaN (already handled by nan_to_num above)
                if np.any(np.isnan(final_preds_float)):
                    print("ERROR: NaNs detected in final_preds_float just before sklearn calls, despite nan_to_num. Metrics set to 0.")
                    auc, ap, f1 = 0.0, 0.0, 0.0
                else:
                    auc = roc_auc_score(final_targets_int, final_preds_float)
                    ap = average_precision_score(final_targets_int, final_preds_float)
                    f1 = f1_score(final_targets_int, (final_preds_float >= 0.5).astype(int), zero_division=0)
        except ValueError as e:
            print(f"ValueError during metrics calculation: {e}. NaNs in input or single class in targets are common causes.")
            auc, ap, f1 = 0.0, 0.0, 0.0 # Default values in case of any error
            
        avg_eval_loss = total_loss / len(eval_loader) if len(eval_loader) > 0 else 0
        return avg_eval_loss, auc, ap, f1