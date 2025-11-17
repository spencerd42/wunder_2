import torch
from torch.utils.data import Dataset
from pandas import DataFrame
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class SubsequenceDataset(Dataset):
    def __init__(self, data, window_size=100):
        """
        Args:
            data: pandas DataFrame with your time series data
            window_size: number of previous rows to use as input (default: 100)
        """
        self.data = data
        self.window_size = window_size
        
        # Filter indices where prediction is needed
        self.valid_indices = []
        for i in range(len(data)):
            if data.iloc[i]['need_prediction'] and i >= window_size:
                self.valid_indices.append(i)
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        # Get the actual data index
        data_idx = self.valid_indices[idx]
        
        # x: previous 100 rows (features)
        x = self.data.iloc[data_idx - self.window_size:data_idx].values
        
        # y: current row (target)
        y = self.data.iloc[data_idx].values
        
        # Convert to torch tensors
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        
        return x, y

def split_and_scale(data: DataFrame):
    # Get the unique sequence IDs
    unique_seqs = data['seq_ix'].unique()

    # Split the sequence IDs into train and test sets
    train_seqs, test_seqs = train_test_split(unique_seqs, test_size=0.2, random_state=42)

    # Use boolean indexing to filter the original dataframe
    train_data = data[data['seq_ix'].isin(train_seqs)].reset_index(drop=True)
    test_data = data[data['seq_ix'].isin(test_seqs)].reset_index(drop=True)
    
    feature_cols = [str(x) for x in range(32)]
    
    scaler = StandardScaler()
    train_data[feature_cols] = scaler.fit_transform(train_data[feature_cols])
    test_data[feature_cols] = scaler.transform(test_data[feature_cols])
    
    return train_data, test_data, scaler