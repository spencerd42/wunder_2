import torch
from torch.utils.data import Dataset
from pandas import DataFrame
from sklearn.model_selection import train_test_split
import numpy as np
from torch.utils.data import DataLoader

class SubsequenceDataset(Dataset):
    def __init__(self, data, window_size, y_indices):
        self.window_size = window_size
        self.y_indices = y_indices
        
        # CRITICAL: Convert DataFrame to numpy array ONCE
        self.data_array = data.values.astype(np.float32)
    
    def __len__(self):
        return len(self.y_indices)
    
    def __getitem__(self, idx):
        y_idx = self.y_indices[idx]
        
        # FAST: Direct numpy slicing (no pandas overhead)
        x = self.data_array[y_idx - self.window_size:y_idx, 3:35]
        y = self.data_array[y_idx, 3:35]
        
        # Use from_numpy (shares memory) instead of tensor (copies)
        x = torch.from_numpy(x)
        y = torch.from_numpy(y)
        
        return x, y

def split_data(data: DataFrame, window_size):
    y_indices = []
    
    for seq_ix in range(517):
        for step_ix in range(window_size, 1000):
            y_indices.append((seq_ix * 1000) + step_ix)
    
    train_indices, other_indices = train_test_split(y_indices, test_size=0.3, shuffle=True, random_state=42)
    val_indices, test_indices = train_test_split(other_indices, test_size=0.5, shuffle=True, random_state=42)
            
    return train_indices, val_indices, test_indices

def make_dataloaders(data: DataFrame, window_size, batch_size):
    """
    Create train, validation, and test dataloaders with proper splitting
    
    Split strategy:
    - Training: 70%
    - Validation: 15%
    - Test: 15%
    """
    
    train_indices, val_indices, test_indices = split_data(data, window_size)
    
    # Create dataset instances
    train_dataset = SubsequenceDataset(data, window_size, train_indices)
    val_dataset = SubsequenceDataset(data, window_size, val_indices)
    test_dataset = SubsequenceDataset(data, window_size, test_indices)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=4,
        persistent_workers=True,
        pin_memory=True if torch.cuda.is_available() else False
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=4,
        persistent_workers=True,
        pin_memory=True if torch.cuda.is_available() else False
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=4,
        persistent_workers=True,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    return train_loader, val_loader, test_loader