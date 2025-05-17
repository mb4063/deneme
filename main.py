import os
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from data.dataset_a import EventBasedDataset
from models.temporal_gat import TemporalGAT
from training.trainer import TemporalLinkPredictionTrainer

def parse_args():
    parser = argparse.ArgumentParser(description='Temporal Link Prediction')
    parser.add_argument('--hidden_channels', type=int, default=256,
                        help='Number of hidden channels')
    parser.add_argument('--num_heads', type=int, default=16,
                        help='Number of attention heads')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='Dropout probability')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay')
    parser.add_argument('--batch_size', type=int, default=512,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--patience', type=int, default=15,
                        help='Patience for early stopping')
    parser.add_argument('--time_window', type=int, default=7,
                        help='Time window for prediction (in days)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')
    
    return parser.parse_args()

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    
def plot_training_history(history, save_path):
    plt.figure(figsize=(12, 4))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    # Plot metrics
    plt.subplot(1, 2, 2)
    plt.plot(history['val_auc'], label='AUC')
    plt.plot(history['val_ap'], label='AP')
    plt.plot(history['val_f1'], label='F1')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.title('Validation Metrics')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    # Parse arguments
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Create directories
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # Print GPU information
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"PyTorch Version: {torch.__version__}")
        
        # Enable CUDA optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True  # Allow TF32 on Ampere
        torch.backends.cudnn.allow_tf32 = True
        
        # Set memory allocation settings
        torch.cuda.empty_cache()
        torch.cuda.set_per_process_memory_fraction(0.95)  # Use 95% of available GPU memory
    else:
        print("No GPU available, using CPU")
    
    # Calculate optimal batch size based on available GPU memory
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.get_device_properties(0).total_memory
        # Adjust batch size based on available memory (rough estimation)
        optimal_batch = min(args.batch_size, int(gpu_mem / (2 * 1024 * 1024 * 1024) * 1024))
        args.batch_size = optimal_batch
        print(f"Adjusted batch size to {optimal_batch} based on GPU memory")
    
    # Load dataset A
    print("Loading and preprocessing dataset...")
    dataset = EventBasedDataset(name='dataset_a', time_window=args.time_window)
    dataset.load_data('data/edges_train_A.csv')
    dataset.preprocess()
    
    # Move dataset to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset.node_features = dataset.node_features.to(device)
    dataset.edge_indices = dataset.edge_indices.to(device)
    dataset.edge_features = dataset.edge_features.to(device)
    dataset.edge_timestamps = dataset.edge_timestamps.to(device)
    dataset.targets = dataset.targets.to(device)
    
    print(f"Training on dataset A using {device}")
    
    # Create GAT model
    model = TemporalGAT(
        node_features=dataset.node_features.size(1),
        hidden_channels=args.hidden_channels,
        num_heads=args.num_heads,
        dropout=args.dropout
    ).to(device)
    
    # Enable gradient checkpointing for memory efficiency
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
    
    # Create trainer
    trainer = TemporalLinkPredictionTrainer(
        model=model,
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Train model
    history, test_metrics = trainer.train(
        dataset=dataset,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        checkpoint_dir=os.path.join(args.checkpoint_dir, dataset.name)
    )
    
    # Plot training history
    plot_training_history(
        history=history,
        save_path=f"results/dataset_a_training_history.png"
    )
    
    # Print test metrics
    print(f"\nTest metrics for dataset A:")
    print(f"  AUC: {test_metrics['auc']:.4f}")
    print(f"  AP: {test_metrics['ap']:.4f}")
    print(f"  F1: {test_metrics['f1']:.4f}")

if __name__ == '__main__':
    main()