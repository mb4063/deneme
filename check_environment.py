import torch
import torch_geometric
import torch_geometric_temporal
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

print("PyTorch version:", torch.__version__)
print("PyTorch Geometric version:", torch_geometric.__version__)
print("PyTorch Geometric Temporal version:", torch_geometric_temporal.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA version:", torch.version.cuda)
    print("GPU device:", torch.cuda.get_device_name(0))

print("Environment check completed successfully!")