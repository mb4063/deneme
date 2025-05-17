import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

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
        self.optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.7, patience=7, verbose=True
        )
        self.criterion = nn.BCELoss()
        
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
        
        # Generate link prediction samples
        train_samples, val_samples, test_samples = self._generate_link_prediction_samples(dataset)
        
        # Create data loaders
        train_loader = self._create_data_loader(train_samples, batch_size)
        val_loader = self._create_data_loader(val_samples, batch_size)
        test_loader = self._create_data_loader(test_samples, batch_size)
        
        # Initialize variables for training
        best_val_loss = float('inf')
        best_epoch = 0
        patience_counter = 0
        max_grad_norm = 2.0  # Gradient clipping değeri artırıldı
        
        # Initialize history
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_auc': [],
            'val_ap': [],
            'val_f1': [],
            'learning_rates': []
        }
        
        print(f"Starting training for {num_epochs} epochs...")
        
        # Training loop
        for epoch in range(num_epochs):
            # Train
            train_loss = self._train_epoch(train_loader, dataset.node_features, 
                                         dataset.edge_indices, max_grad_norm)
            
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
            
            # Print progress
            print(f"Epoch {epoch+1}/{num_epochs} - "
                  f"Train Loss: {train_loss:.4f}, "
                  f"Val Loss: {val_loss:.4f}, "
                  f"Val AUC: {val_metrics['auc']:.4f}, "
                  f"Val AP: {val_metrics['ap']:.4f}, "
                  f"Val F1: {val_metrics['f1']:.4f}, "
                  f"LR: {current_lr:.6f}")
            
            # Check for improvement
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                patience_counter = 0
                
                # Save checkpoint
                if checkpoint_dir:
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'scheduler_state_dict': self.scheduler.state_dict(),
                        'val_loss': val_loss,
                        'val_metrics': val_metrics
                    }, os.path.join(checkpoint_dir, 'best_model.pt'))
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
                
        print(f"Training completed. Best epoch: {best_epoch+1}")
        
        # Load best model
        if checkpoint_dir and os.path.exists(os.path.join(checkpoint_dir, 'best_model.pt')):
            checkpoint = torch.load(os.path.join(checkpoint_dir, 'best_model.pt'))
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
        # Evaluate on test set
        _, test_metrics = self._validate(test_loader, dataset.node_features, dataset.edge_indices)
        
        return history, test_metrics
    
    def _train_epoch(self, train_loader, node_features, edge_indices, max_grad_norm):
        """
        Train for one epoch.
        
        Args:
            train_loader: Data loader for training
            node_features: Node features
            edge_indices: Edge indices
            
        Returns:
            float: Training loss
        """
        self.model.train()
        total_loss = 0
        
        for batch in train_loader:
            # Get batch data
            pos_edge_index, neg_edge_index, _ = batch
            
            # Forward pass
            node_embeddings = self.model(node_features, edge_indices)
            
            # Predict links
            pos_pred = self.model.predict_link(node_embeddings, pos_edge_index)
            neg_pred = self.model.predict_link(node_embeddings, neg_edge_index)
            
            # Create targets
            pos_target = torch.ones_like(pos_pred)
            neg_target = torch.zeros_like(neg_pred)
            
            # Combine predictions and targets
            pred = torch.cat([pos_pred, neg_pred])
            target = torch.cat([pos_target, neg_target])

            # Boyut kontrolü için assert ekle
            assert pred.shape == target.shape, f"Prediction shape: {pred.shape}, Target shape: {target.shape}"

            # Compute loss
            loss = self.criterion(pred, target)
            
            # Backward pass with gradient clipping
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
            self.optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / len(train_loader)
    
    def _validate(self, val_loader, node_features, edge_indices):
        """
        Validate the model.
        
        Args:
            val_loader: Data loader for validation
            node_features: Node features
            edge_indices: Edge indices
            
        Returns:
            tuple: (validation loss, validation metrics)
        """
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch in val_loader:
                # Get batch data
                pos_edge_index, neg_edge_index, _ = batch
                
                # Forward pass
                node_embeddings = self.model(node_features, edge_indices)
                
                # Predict links
                pos_pred = self.model.predict_link(node_embeddings, pos_edge_index)
                neg_pred = self.model.predict_link(node_embeddings, neg_edge_index)
                
                # Create targets
                pos_target = torch.ones_like(pos_pred)
                neg_target = torch.zeros_like(neg_pred)
                
                # Combine predictions and targets
                pred = torch.cat([pos_pred, neg_pred])
                target = torch.cat([pos_target, neg_target])
                
                # Boyut kontrolü için assert ekle
                assert pred.shape == target.shape, f"Prediction shape: {pred.shape}, Target shape: {target.shape}"

                # Compute loss
                loss = self.criterion(pred, target)
                total_loss += loss.item()
                
                # Store predictions and targets for metrics
                all_preds.append(pred.cpu().numpy())
                all_targets.append(target.cpu().numpy())
                
        # Compute metrics
        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)
        
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
        Generate positive and negative samples for link prediction.
        
        Args:
            dataset: Dataset
            
        Returns:
            tuple: (train_samples, val_samples, test_samples)
        """
        # Sort edges by timestamp
        sorted_indices = torch.argsort(dataset.edge_timestamps)
        sorted_edge_indices = dataset.edge_indices[:, sorted_indices]
        sorted_timestamps = dataset.edge_timestamps[sorted_indices]
        sorted_targets = dataset.targets[sorted_indices]
        
        # Split data by time
        n_edges = sorted_edge_indices.size(1)
        train_ratio, val_ratio = 0.7, 0.15
        
        train_size = int(n_edges * train_ratio)
        val_size = int(n_edges * val_ratio)
        
        train_edges = sorted_edge_indices[:, :train_size]
        train_times = sorted_timestamps[:train_size]
        train_targets = sorted_targets[:train_size]
        
        val_edges = sorted_edge_indices[:, train_size:train_size+val_size]
        val_times = sorted_timestamps[train_size:train_size+val_size]
        val_targets = sorted_targets[train_size:train_size+val_size]
        
        test_edges = sorted_edge_indices[:, train_size+val_size:]
        test_times = sorted_timestamps[train_size+val_size:]
        test_targets = sorted_targets[train_size+val_size:]
        
        # Generate negative samples
        train_neg = self._generate_negative_samples(train_edges, train_size)
        val_neg = self._generate_negative_samples(val_edges, val_size)
        test_neg = self._generate_negative_samples(test_edges, len(test_edges[0]))
        
        # Create samples
        train_samples = (train_edges.t(), train_neg.t(), train_times)
        val_samples = (val_edges.t(), val_neg.t(), val_times)
        test_samples = (test_edges.t(), test_neg.t(), test_times)
        
        return train_samples, val_samples, test_samples
    
    def _generate_negative_samples(self, positive_edges, num_samples):
        """
        Generate negative edge samples (edges that don't exist).
        
        Args:
            positive_edges (torch.Tensor): Positive edge indices
            num_samples (int): Number of negative samples to generate
            
        Returns:
            torch.Tensor: Negative edge indices
        """
        # Get all nodes
        all_nodes = torch.unique(positive_edges)
        num_nodes = all_nodes.size(0)
        
        # Create a set of positive edges for fast lookup
        positive_edge_set = set(map(tuple, positive_edges.t().tolist()))
        
        # Generate random node pairs
        negative_edges = []
        while len(negative_edges) < num_samples:
            # Sample random node pairs
            src = torch.randint(0, num_nodes, (1,)).item()
            dst = torch.randint(0, num_nodes, (1,)).item()
            
            # Skip self-loops and existing edges
            if src != dst and (src, dst) not in positive_edge_set and (dst, src) not in positive_edge_set:
                negative_edges.append([src, dst])
                
        return torch.tensor(negative_edges, dtype=torch.long).t()
    
    def _create_data_loader(self, samples, batch_size):
        """
        Create a data loader for link prediction samples.
        
        Args:
            samples: Link prediction samples
            batch_size (int): Batch size
            
        Returns:
            DataLoader: Data loader
        """
        pos_edge_index, neg_edge_index, timestamps = samples
        
        # Boyutları kontrol et ve yazdır
        print(f"pos_edge_index shape: {pos_edge_index.shape}")
        print(f"neg_edge_index shape: {neg_edge_index.shape}")
        print(f"timestamps shape: {timestamps.shape}")
        
        # Tensörlerin ilk boyutlarının eşit olduğundan emin ol
        assert pos_edge_index.shape[0] == neg_edge_index.shape[0] == timestamps.shape[0], \
            f"Size mismatch: pos={pos_edge_index.shape[0]}, neg={neg_edge_index.shape[0]}, time={timestamps.shape[0]}"
        
        # Create dataset
        dataset = TensorDataset(pos_edge_index, neg_edge_index, timestamps)
        
        # Create data loader
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        return data_loader