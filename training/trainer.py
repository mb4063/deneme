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
        # Enable automatic mixed precision training
        self.scaler = torch.cuda.amp.GradScaler()
        self.optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.7, patience=7, verbose=True
        )
        # Use BCEWithLogitsLoss instead of BCELoss for numerical stability
        self.criterion = nn.BCEWithLogitsLoss()
        
    def train(self, dataset, num_epochs=100, batch_size=32, patience=15, checkpoint_dir=None):
        """
        Train the model.
        
        Args:
            dataset: Dataset to train on
            num_epochs (int): Number of epochs
            batch_size (int): Batch size
            patience (int): Patience for early stopping
            checkpoint_dir (str): Directory to save checkpoints
            
        Returns:
            tuple: (history, test_metrics)
        """
        print("Preparing data for training...")
        
        # Move dataset to GPU and pin memory for faster data transfer
        train_samples, val_samples, test_samples = self._generate_link_prediction_samples(dataset)
        
        # Create data loaders with optimized settings
        train_loader = self._create_data_loader(train_samples, batch_size, pin_memory=True, num_workers=4)
        val_loader = self._create_data_loader(val_samples, batch_size, pin_memory=True, num_workers=4)
        test_loader = self._create_data_loader(test_samples, batch_size, pin_memory=True, num_workers=4)
        
        # Initialize variables for training
        best_val_loss = float('inf')
        best_epoch = 0
        patience_counter = 0
        max_grad_norm = 2.0
        
        # Initialize history
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_auc': [],
            'val_ap': [],
            'val_f1': [],
            'learning_rates': []
        }
        
        print(f"\nStarting training:")
        print(f"- Total epochs: {num_epochs}")
        print(f"- Batches per epoch: {len(train_loader)}")
        print(f"- Total batches: {num_epochs * len(train_loader)}")
        
        # Enable CUDA benchmarking for faster training
        torch.backends.cudnn.benchmark = True
        
        # Training loop with progress bar
        epoch_pbar = tqdm(range(num_epochs), desc='Training Progress', position=0)
        
        for epoch in epoch_pbar:
            epoch_pbar.set_description(f"Epoch {epoch + 1}/{num_epochs}")
            
            # Train
            train_loss = self._train_epoch(train_loader, dataset.node_features, 
                                         dataset.edge_indices, max_grad_norm,
                                         epoch=epoch, num_epochs=num_epochs)
            
            # Validate
            val_loss, val_metrics = self._validate(val_loader, dataset.node_features, 
                                                 dataset.edge_indices)
            
            # Update learning rate
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Update history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_auc'].append(val_metrics['auc'])
            history['val_ap'].append(val_metrics['ap'])
            history['val_f1'].append(val_metrics['f1'])
            history['learning_rates'].append(current_lr)
            
            # Update progress bar description
            epoch_pbar.set_postfix({
                'train_loss': f'{train_loss:.4f}',
                'val_loss': f'{val_loss:.4f}',
                'val_auc': f'{val_metrics["auc"]:.4f}',
                'lr': f'{current_lr:.6f}'
            })
            
            # Print epoch summary
            print(f"\nEpoch {epoch + 1}/{num_epochs} Summary:")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  Val AUC: {val_metrics['auc']:.4f}")
            print(f"  Val AP: {val_metrics['ap']:.4f}")
            print(f"  Val F1: {val_metrics['f1']:.4f}")
            print(f"  Learning Rate: {current_lr:.6f}")
            
            # Check for improvement
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                patience_counter = 0
                
                # Save checkpoint
                if checkpoint_dir:
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    checkpoint = {
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'scheduler_state_dict': self.scheduler.state_dict(),
                        'scaler_state_dict': self.scaler.state_dict(),
                        'val_loss': val_loss,
                        'val_metrics': val_metrics
                    }
                    torch.save(checkpoint, os.path.join(checkpoint_dir, 'best_model.pt'))
                    print(f"  Saved checkpoint (best model so far)")
            else:
                patience_counter += 1
                print(f"  No improvement for {patience_counter} epochs")
            
            # Early stopping
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
        
        print(f"\nTraining completed. Best epoch: {best_epoch+1}")
        
        # Load best model
        if checkpoint_dir and os.path.exists(os.path.join(checkpoint_dir, 'best_model.pt')):
            checkpoint = torch.load(os.path.join(checkpoint_dir, 'best_model.pt'))
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
        # Evaluate on test set
        _, test_metrics = self._validate(test_loader, dataset.node_features, dataset.edge_indices)
        
        return history, test_metrics
    
    def _train_epoch(self, train_loader, node_features, edge_indices, max_grad_norm, epoch=0, num_epochs=1):
        """
        Train for one epoch with mixed precision training and gradient accumulation.
        """
        self.model.train()
        total_loss = 0
        device = next(self.model.parameters()).device
        num_nodes = node_features.size(0)
        
        # Ensure edge_indices is properly formatted
        if edge_indices.dim() == 1:
            edge_indices = edge_indices.view(2, -1)
        elif edge_indices.dim() == 2 and edge_indices.size(0) != 2:
            edge_indices = edge_indices.t()
            
        # Validate edge indices
        assert edge_indices.max() < num_nodes, f"Edge index {edge_indices.max()} >= num_nodes {num_nodes}"
        assert edge_indices.min() >= 0, f"Negative edge index found: {edge_indices.min()}"
        
        # Use tqdm for progress tracking
        batch_pbar = tqdm(
            train_loader,
            desc=f'Epoch {epoch + 1}/{num_epochs}',
            leave=False,
            position=1
        )
        
        # Gradient accumulation settings
        accumulation_steps = 4  # Adjust based on GPU memory
        self.optimizer.zero_grad(set_to_none=True)
        
        for batch_idx, batch in enumerate(batch_pbar):
            # Move batch to device
            batch = self._move_batch_to_device(batch, device)
            pos_edge_index, neg_edge_index, _ = batch
            
            # Validate batch indices
            for edge_set, name in [(pos_edge_index, "positive"), (neg_edge_index, "negative")]:
                if edge_set.dim() == 1:
                    edge_set = edge_set.view(2, -1)
                elif edge_set.dim() == 2 and edge_set.size(0) != 2:
                    edge_set = edge_set.t()
                
                max_idx = edge_set.max()
                min_idx = edge_set.min()
                if max_idx >= num_nodes:
                    raise ValueError(f"Batch {name} edge index {max_idx} >= num_nodes {num_nodes}")
                if min_idx < 0:
                    raise ValueError(f"Batch {name} edge index {min_idx} < 0")
            
            # Forward pass with mixed precision
            with torch.cuda.amp.autocast():
                # Compute node embeddings for this batch
                node_embeddings = self.model(node_features, edge_indices)
                
                # Predict links
                pos_pred = self.model.predict_link(node_embeddings, pos_edge_index)
                neg_pred = self.model.predict_link(node_embeddings, neg_edge_index)
                
                # Create targets with label smoothing
                pos_target = torch.ones_like(pos_pred) * 0.9  # Label smoothing
                neg_target = torch.zeros_like(neg_pred) * 0.1
                
                # Combine predictions and targets
                pred = torch.cat([pos_pred, neg_pred])
                target = torch.cat([pos_target, neg_target])
                
                # Compute loss with stability
                loss = self.criterion(pred, target) / accumulation_steps
            
            # Backward pass with gradient scaling
            self.scaler.scale(loss).backward()
            
            # Update weights every accumulation_steps
            if (batch_idx + 1) % accumulation_steps == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
            
            total_loss += loss.item() * accumulation_steps
            
            # Update progress bar with smoothed loss
            smoothing = 0.1
            if batch_idx == 0:
                smoothed_loss = loss.item() * accumulation_steps
            else:
                smoothed_loss = smoothing * loss.item() * accumulation_steps + (1 - smoothing) * smoothed_loss
            
            batch_pbar.set_postfix({
                'batch': f'{batch_idx+1}/{len(train_loader)}',
                'loss': f'{smoothed_loss:.4f}',
                'avg_loss': f'{total_loss/(batch_idx+1):.4f}'
            })
            
            # Clear memory
            del batch, pos_edge_index, neg_edge_index, pos_pred, neg_pred, node_embeddings
            if (batch_idx + 1) % 50 == 0:  # Periodic memory cleanup
                torch.cuda.empty_cache()
        
        return total_loss / len(train_loader)
    
    def _validate(self, val_loader, node_features, edge_indices):
        """
        Validate the model with mixed precision inference.
        """
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_targets = []
        device = next(self.model.parameters()).device
        
        # Ensure edge_indices is properly formatted
        if edge_indices.dim() == 1:
            edge_indices = edge_indices.view(2, -1)
        elif edge_indices.dim() == 2 and edge_indices.size(0) != 2:
            edge_indices = edge_indices.t()
        
        with torch.no_grad(), torch.cuda.amp.autocast():
            # Get node embeddings using validation_forward
            node_embeddings = self.model.validation_forward(node_features, edge_indices)
            
            for batch in val_loader:
                # Move batch to device
                batch = self._move_batch_to_device(batch, device)
                pos_edge_index, neg_edge_index, _ = batch
                
                # Ensure batch edge indices are properly formatted
                if pos_edge_index.dim() == 1:
                    pos_edge_index = pos_edge_index.view(2, -1)
                elif pos_edge_index.dim() == 2 and pos_edge_index.size(0) != 2:
                    pos_edge_index = pos_edge_index.t()
                    
                if neg_edge_index.dim() == 1:
                    neg_edge_index = neg_edge_index.view(2, -1)
                elif neg_edge_index.dim() == 2 and neg_edge_index.size(0) != 2:
                    neg_edge_index = neg_edge_index.t()
                
                # Predict links using cached node embeddings
                pos_pred = self.model.predict_link(node_embeddings, pos_edge_index)
                neg_pred = self.model.predict_link(node_embeddings, neg_edge_index)
                
                # Create targets
                pos_target = torch.ones_like(pos_pred)
                neg_target = torch.zeros_like(neg_pred)
                
                # Combine predictions and targets
                pred = torch.cat([pos_pred, neg_pred])
                target = torch.cat([pos_target, neg_target])
                
                # Compute loss
                loss = self.criterion(pred, target)
                total_loss += loss.item()
                
                # Store predictions and targets for metrics
                all_preds.append(pred.cpu())
                all_targets.append(target.cpu())
                
                # Clear memory
                del batch, pos_edge_index, neg_edge_index, pos_pred, neg_pred
                torch.cuda.empty_cache()
                
        # Compute metrics
        all_preds = torch.cat(all_preds).numpy()
        all_targets = torch.cat(all_targets).numpy()
        
        auc = roc_auc_score(all_targets, all_preds)
        ap = average_precision_score(all_targets, all_preds)
        f1 = f1_score(all_targets, all_preds > 0.5)
        
        metrics = {
            'auc': auc,
            'ap': ap,
            'f1': f1
        }
        
        return total_loss / len(val_loader), metrics
    
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