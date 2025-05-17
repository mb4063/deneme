import torch
import pandas as pd
import numpy as np
from torch_geometric_temporal.signal import DynamicGraphTemporalSignal
import os

class EventBasedDataset:
    """
    Dataset class for event-based temporal data.
    """
    
    def __init__(self, name, time_window):
        """
        Initialize the dataset.
        
        Args:
            name (str): Name of the dataset
            time_window (int): Size of the time window for prediction
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
        
    def load_data(self, data_path):
        """
        Load data from file.
        
        Args:
            data_path (str): Path to the data file
        """
        print(f"Loading data from {data_path}")
        
        # Load data from CSV
        df = pd.read_csv(data_path, header=None)
        
        # WSDM formatına göre düzenleme:
        # edges_train_A.csv formatı: src_id, dst_id, edge_type, timestamp
        if len(df.columns) == 4:
            print("Detected edges_train format with 4 columns (src_id, dst_id, edge_type, timestamp)")
            sources = df.iloc[:, 0].values
            targets = df.iloc[:, 1].values
            edge_types = df.iloc[:, 2].values
            timestamps = df.iloc[:, 3].values
            
            # Test veri seti için label ekleme ihtiyacı yok, tüm kenarları pozitif olarak işaretliyoruz
            labels = np.ones(len(sources), dtype=np.float32)
            
            # Kenar türü özelliklerini yüklemeye çalış
            edge_type_features = None
            try:
                if self.edge_type_path and os.path.exists(self.edge_type_path):
                    # Kategorik edge type özelliklerini yükle
                    edge_type_df = pd.read_csv(self.edge_type_path)
                    edge_type_features = []
                    for edge_type in edge_types:
                        # edge_type_df'de ilgili edge_type'ın özelliklerini bul
                        type_features = edge_type_df[edge_type_df.iloc[:, 0] == edge_type].iloc[:, 1:].values
                        if len(type_features) > 0:
                            edge_type_features.append(type_features[0])
                        else:
                            # Eğer edge type bulunamazsa sıfır vektörü kullan
                            edge_type_features.append(np.zeros(edge_type_df.shape[1] - 1))
                    edge_type_features = np.array(edge_type_features, dtype=np.float32)
                    print(f"Loaded edge type features from {self.edge_type_path}")
                    # Edge type özelliklerini ve edge type'ı birleştir
                    features = np.concatenate([np.expand_dims(edge_types, axis=1), edge_type_features], axis=1).astype(np.float32)
            except Exception as e:
                print(f"Warning: Could not load edge type features: {e}")
                # Sadece edge type'ı özellik olarak kullan
                features = np.expand_dims(edge_types, axis=1).astype(np.float32)
        
        # Test veri seti formatı: src_id, dst_id, edge_type, start_time, end_time, label
        elif len(df.columns) == 6:
            print("Detected test data format with 6 columns (src, dst, edge_type, start_time, end_time, label)")
            sources = df.iloc[:, 0].values
            targets = df.iloc[:, 1].values
            edge_types = df.iloc[:, 2].values
            
            # Test veri setinde start_time ve end_time var. Biz başlangıç zamanını kullanıyoruz.
            timestamps = df.iloc[:, 3].values
            labels = df.iloc[:, 5].values
            
            # Kenar türünü bir özellik olarak kullanıyoruz
            features = np.expand_dims(edge_types, axis=1).astype(np.float32)
            
        # Genel veri seti formatı: bunun dışındaki formatlar için
        else:
            print(f"Using general format with {len(df.columns)} columns")
            sources = df.iloc[:, 0].values
            targets = df.iloc[:, 1].values
            
            # Extract features (assuming columns 2 to -3 are features)
            if len(df.columns) > 4:
                features = df.iloc[:, 2:-2].values
            else:
                features = np.zeros((len(sources), 1), dtype=np.float32)
            
            # Extract timestamps and labels
            if len(df.columns) >= 3:
                timestamps = df.iloc[:, -2].values
            else:
                timestamps = np.zeros(len(sources))
                
            if len(df.columns) >= 4:
                labels = df.iloc[:, -1].values  
            else:
                labels = np.ones(len(sources), dtype=np.float32)
        
        # Node id mapping
        all_node_ids = np.unique(np.concatenate([sources, targets]))
        node_id_map = {id_: idx for idx, id_ in enumerate(all_node_ids)}
        mapped_sources = np.array([node_id_map[x] for x in sources])
        mapped_targets = np.array([node_id_map[x] for x in targets])
        self.edge_indices = torch.tensor([mapped_sources, mapped_targets], dtype=torch.long)
        
        # Create edge features
        self.edge_features = torch.tensor(features, dtype=torch.float)
        
        # Create edge timestamps
        self.edge_timestamps = torch.tensor(timestamps, dtype=torch.float)
        
        # Create targets
        self.targets = torch.tensor(labels, dtype=torch.float)
        
        # Node features'ı yükle ve eksik değerleri işle
        try:
            if self.node_feature_path and os.path.exists(self.node_feature_path):
                node_df = pd.read_csv(self.node_feature_path)
                print(f"Loaded node features from {self.node_feature_path}")
                
                # Eksik değerleri (-1) işle
                # Her özellik için ayrı bir "eksik değer" kategorisi oluştur
                node_features = node_df.values
                n_features = node_features.shape[1]
                
                # Her özellik için -1 değerlerini yeni bir kategori olarak işaretle
                processed_features = []
                for i in range(n_features):
                    feature = node_features[:, i]
                    unique_vals = np.unique(feature[feature != -1])
                    n_categories = len(unique_vals) + 1  # +1 for missing value category
                    
                    # One-hot encoding uygula
                    one_hot = np.zeros((len(feature), n_categories))
                    for j, val in enumerate(feature):
                        if val == -1:
                            one_hot[j, -1] = 1  # Eksik değer kategorisi
                        else:
                            # Değerin indeksini bul
                            val_idx = np.where(unique_vals == val)[0][0]
                            one_hot[j, val_idx] = 1
                    
                    processed_features.append(one_hot)
                
                # Tüm özellikleri birleştir
                processed_features = np.concatenate(processed_features, axis=1)
                self.node_features = torch.tensor(processed_features, dtype=torch.float)
            else:
                # Node features yoksa one-hot encoding kullan
                self.node_features = torch.eye(len(all_node_ids), dtype=torch.float)
        except Exception as e:
            print(f"Warning: Using default node features. Error: {e}")
            self.node_features = torch.eye(len(all_node_ids), dtype=torch.float)
        
        print(f"Loaded {len(self.edge_indices[0])} edges, {len(all_node_ids)} nodes")
        
    def preprocess(self):
        """
        Preprocess the loaded data.
        """
        print("Preprocessing data...")
        
        # Normalize timestamps
        min_time = self.edge_timestamps.min()
        max_time = self.edge_timestamps.max()
        self.edge_timestamps = (self.edge_timestamps - min_time) / (max_time - min_time)
        
        # Add timestamp as a feature
        timestamp_feature = self.edge_timestamps.unsqueeze(1)
        if self.edge_features is not None:
            self.edge_features = torch.cat([self.edge_features, timestamp_feature], dim=1)
        else:
            self.edge_features = timestamp_feature
        
        print("Preprocessing complete")