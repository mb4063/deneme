import os
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from data.dataset_a import EventBasedDataset
from models.temporal_gat import TemporalGAT
from training.trainer import TemporalLinkPredictionTrainer
from google.colab import drive
import sys
import subprocess

def setup_environment():
    """
    Colab/Kaggle ortamını hazırla
    """
    # GPU kontrolü
    if torch.cuda.is_available():
        print(f"GPU bulundu: {torch.cuda.get_device_name(0)}")
        print(f"Kullanılabilir GPU sayısı: {torch.cuda.device_count()}")
    else:
        print("GPU bulunamadı, CPU kullanılacak!")
        
    # Gerekli kütüphanelerin kurulumu
    subprocess.check_call([sys.executable, "-m", "pip", "install", "torch-geometric"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "torch-geometric-temporal"])

def mount_drive():
    """
    Google Drive'ı bağla (Colab için)
    """
    try:
        drive.mount('/content/drive')
        print("Google Drive bağlandı")
        return True
    except:
        print("Google Drive bağlanamadı - Kaggle ortamında olabilirsiniz")
        return False

def parse_args():
    parser = argparse.ArgumentParser(description='Temporal Link Prediction - Cloud Training')
    parser.add_argument('--hidden_channels', type=int, default=128,
                        help='Number of hidden channels')
    parser.add_argument('--num_heads', type=int, default=8,
                        help='Number of attention heads')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='Dropout probability')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--patience', type=int, default=15,
                        help='Patience for early stopping')
    parser.add_argument('--time_window', type=int, default=7,
                        help='Time window for prediction (in days)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--platform', type=str, choices=['colab', 'kaggle'], default='colab',
                        help='Platform to run on')
    
    return parser.parse_args()

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def main():
    # Ortamı hazırla
    setup_environment()
    
    # Argümanları parse et
    args = parse_args()
    
    # Seed ayarla
    set_seed(args.seed)
    
    # Platform kontrolü ve veri yolu ayarları
    if args.platform == 'colab':
        is_mounted = mount_drive()
        if is_mounted:
            base_path = '/content/drive/MyDrive/temporal_link_prediction'  # Google Drive'da oluşturacağınız klasör
            os.makedirs(base_path, exist_ok=True)
        else:
            print("Google Drive bağlanamadı!")
            return
    else:  # Kaggle
        base_path = '/kaggle/working/temporal_link_prediction'
        os.makedirs(base_path, exist_ok=True)
    
    # Checkpoint ve sonuç klasörlerini oluştur
    checkpoint_dir = os.path.join(base_path, 'checkpoints')
    results_dir = os.path.join(base_path, 'results')
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    # Dataset yükle
    dataset = EventBasedDataset(name='dataset_a', time_window=args.time_window)
    dataset.load_data('data/edges_train_A.csv')
    dataset.preprocess()
    
    print(f"Training on dataset A with GPU support")
    
    # Model oluştur ve GPU'ya taşı
    model = TemporalGAT(
        node_features=dataset.node_features.size(1),
        hidden_channels=args.hidden_channels,
        num_heads=args.num_heads,
        dropout=args.dropout
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Dataset'i GPU'ya taşı
    dataset.node_features = dataset.node_features.to(device)
    dataset.edge_indices = dataset.edge_indices.to(device)
    dataset.edge_features = dataset.edge_features.to(device)
    dataset.edge_timestamps = dataset.edge_timestamps.to(device)
    dataset.targets = dataset.targets.to(device)
    
    # Trainer oluştur
    trainer = TemporalLinkPredictionTrainer(
        model=model,
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Modeli eğit
    history, test_metrics = trainer.train(
        dataset=dataset,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        checkpoint_dir=checkpoint_dir
    )
    
    # Sonuçları kaydet
    plt.figure(figsize=(12, 4))
    
    # Loss grafiği
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    # Metrikler grafiği
    plt.subplot(1, 2, 2)
    plt.plot(history['val_auc'], label='AUC')
    plt.plot(history['val_ap'], label='AP')
    plt.plot(history['val_f1'], label='F1')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.title('Validation Metrics')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'training_history.png'))
    
    # Test metriklerini yazdır ve kaydet
    print(f"\nTest metrics:")
    print(f"  AUC: {test_metrics['auc']:.4f}")
    print(f"  AP: {test_metrics['ap']:.4f}")
    print(f"  F1: {test_metrics['f1']:.4f}")
    
    # Metrikleri dosyaya kaydet
    with open(os.path.join(results_dir, 'test_metrics.txt'), 'w') as f:
        f.write(f"Test Metrics:\n")
        f.write(f"AUC: {test_metrics['auc']:.4f}\n")
        f.write(f"AP: {test_metrics['ap']:.4f}\n")
        f.write(f"F1: {test_metrics['f1']:.4f}\n")

if __name__ == "__main__":
    main() 