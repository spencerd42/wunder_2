import os


class LightningTrainingConfig:
    """Configuration for PyTorch Lightning training"""
    
    def __init__(self):
        # Data parameters
        self.input_size = 32
        self.window_size = 100
        self.batch_size = 256
        
        # Get the directory of the current file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.data_path = os.path.join(current_dir, '../data/train.parquet')
        
        # Model parameters (TFT-specific)
        self.hidden_size = 64
        self.num_heads = 4
        self.num_lstm_layers = 2
        self.dropout = 0.1
        
        # Training parameters
        self.num_epochs = 100
        self.learning_rate = 0.0005
        
        # Learning rate scheduler parameters
        self.lr_scheduler_factor = 0.5
        self.lr_scheduler_patience = 8
        
        # Early stopping parameters
        self.early_stopping_patience = 10
        self.early_stopping_min_delta = 1e-5
        
        # Checkpoint parameters
        self.checkpoint_dir = "checkpoints"
        
        # Other parameters
        self.random_seed = 42
    
    def to_dict(self):
        """Convert config to dictionary"""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
