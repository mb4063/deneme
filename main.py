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
    parser.add_argument('--hidden_channels', type=int, default=64,
                        help='Number of hidden channels')
    parser.add_argument('--num_heads', type=int, default=4,
                        help='Number of attention heads')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout probability')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay')
    parser.add_argument('--batch_size', type=int, default=32,
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
    
    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # Load dataset A
    dataset = EventBasedDataset(name='dataset_a', time_window=args.time_window)
    dataset.load_data('data/edges_train_A.csv')
    dataset.preprocess()
    
    print(f"Training on dataset A")
    
    # Create GAT model
    model = TemporalGAT(
        node_features=dataset.node_features.size(1),
        hidden_channels=args.hidden_channels,
        num_heads=args.num_heads,
        dropout=args.dropout
    )
    
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
    print(f"Test metrics for dataset A:")
    print(f"  AUC: {test_metrics['auc']:.4f}")
    print(f"  AP: {test_metrics['ap']:.4f}")
    print(f"  F1: {test_metrics['f1']:.4f}")

if __name__ == "__main__":
    main()