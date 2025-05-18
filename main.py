import os
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
import json
from data.dataset_a import EventBasedDataset, TemporalSignal
from models.temporal_gat import TemporalGAT
from training.trainer import TemporalLinkPredictionTrainer
from torch_geometric.loader import DataLoader

def parse_args():
    parser = argparse.ArgumentParser(description='Temporal Link Prediction - Forecasting Mode')
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Number of graphs to batch together for GNN processing')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--time_window', type=int, default=None,
                        help='Time window (not actively used for 1-step forecasting setup)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints_forecast',
                        help='Directory to save checkpoints')
    parser.add_argument('--data_path', type=str, default='data/edges_train_A_sample.csv',
                        help='Path to the dataset CSV file')
    parser.add_argument('--neg_sampling_ratio', type=int, default=1,
                        help='Ratio of negative to positive samples during training/evaluation')
    
    return parser.parse_args()

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    
def plot_training_history(history, save_path):
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.get('train_loss', []), label='Train Loss')
    plt.plot(history.get('val_loss', []), label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.get('val_auc', []), label='Val AUC')
    plt.plot(history.get('val_ap', []), label='Val AP')
    plt.plot(history.get('val_f1', []), label='Val F1')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.title('Validation Metrics (Forecasting)')
    plt.legend()
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

def save_training_history_to_file(history, save_path):
    """Saves the training history dictionary to a JSON file."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(history, f, indent=4)
    print(f"Training history saved to {save_path}")

def main():
    args = parse_args()
    set_seed(args.seed)
    
    unique_checkpoint_dir = os.path.join(args.checkpoint_dir, os.path.splitext(os.path.basename(args.data_path))[0])
    os.makedirs(unique_checkpoint_dir, exist_ok=True)
    results_dir = os.path.join('results', os.path.splitext(os.path.basename(args.data_path))[0])
    os.makedirs(results_dir, exist_ok=True)
    
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    else:
        print("No GPU available, using CPU")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("Loading and preprocessing dataset for forecasting...")
    dataset = EventBasedDataset(name=os.path.splitext(os.path.basename(args.data_path))[0])
    dataset.load_data(args.data_path)

    full_signal_instances = dataset.temporal_signal.instances

    if not full_signal_instances:
        print("No forecasting instances were loaded. Exiting.")
        return

    num_total_instances = len(full_signal_instances)
    train_size = int(0.7 * num_total_instances)
    val_size = int(0.15 * num_total_instances)
    test_size = num_total_instances - train_size - val_size

    print(f"Splitting dataset into: Train ({train_size}), Validation ({val_size}), Test ({test_size}) forecasting instances")

    train_signal_instances = full_signal_instances[:train_size]
    val_signal_instances = full_signal_instances[train_size : train_size + val_size]
    test_signal_instances = full_signal_instances[train_size + val_size:]
    
    if not train_signal_instances:
        print("No training instances after split. Exiting.")
        return
    if not val_signal_instances:
        print("Warning: No validation instances after split. Validation metrics will be zero.")
    if not test_signal_instances:
        print("Warning: No test instances after split. Test metrics will be zero.")

    node_feature_dim = train_signal_instances[0].x.size(1)
    print(f"Node feature dimension for GAT model: {node_feature_dim}")

    num_dataloader_workers = 0 if args.batch_size == 1 else 2

    train_loader = DataLoader(train_signal_instances, batch_size=args.batch_size, shuffle=True, num_workers=num_dataloader_workers, pin_memory=True) if train_signal_instances else None
    val_loader = DataLoader(val_signal_instances, batch_size=args.batch_size, shuffle=False, num_workers=num_dataloader_workers, pin_memory=True) if val_signal_instances else None
    test_loader = DataLoader(test_signal_instances, batch_size=args.batch_size, shuffle=False, num_workers=num_dataloader_workers, pin_memory=True) if test_signal_instances else None 

    if not train_loader:
        print("Train loader is empty. Cannot proceed with training. Exiting.")
        return

    print(f"Train DataLoader: {len(train_loader) if train_loader else 0} batches, Val DataLoader: {len(val_loader) if val_loader else 0} batches, Test DataLoader: {len(test_loader) if test_loader else 0} batches")

    model = TemporalGAT(
        node_features=node_feature_dim,
        hidden_channels=args.hidden_channels,
        num_heads=args.num_heads,
        dropout=args.dropout
    ).to(device)
    
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
    
    print(f"Training on {dataset.name} using {device} (Forecasting Mode)")
    
    trainer = TemporalLinkPredictionTrainer(
        model=model,
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    history, test_metrics = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        num_epochs=args.epochs,
        patience=args.patience,
        checkpoint_dir=unique_checkpoint_dir,
        neg_sampling_ratio=args.neg_sampling_ratio
    )
    
    plot_training_history(
        history=history,
        save_path=os.path.join(results_dir, f"{dataset.name}_training_history_forecast.png")
    )
    save_training_history_to_file(
        history=history,
        save_path=os.path.join(results_dir, f"{dataset.name}_training_history_forecast.json")
    )
    
    print(f"\nTest metrics for {dataset.name} (Forecasting Mode):")
    print(f"  Loss: {test_metrics.get('loss', float('nan')):.4f}")
    print(f"  AUC: {test_metrics.get('auc', 0.0):.4f}")
    print(f"  AP: {test_metrics.get('ap', 0.0):.4f}")
    print(f"  F1: {test_metrics.get('f1', 0.0):.4f}")

if __name__ == '__main__':
    main()