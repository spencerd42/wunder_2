class TrainingConfig:
    """Centralized configuration for training"""
    def __init__(self):
        self.input_size = 32
        self.hidden_size = 64
        self.fc_size = 64
        self.num_layers = 2
        self.learning_rate = 0.001
        self.num_epochs = 100
        self.batch_size = 64
        self.window_size = 100
        self.model_path = 'checkpoints/model_20251119_180809.pth'
        
        # Early stopping parameters
        self.early_stopping_patience = 8
        self.early_stopping_min_delta = 1e-5
        
        # Learning rate scheduler parameters
        self.lr_scheduler_factor = 0.5
        self.lr_scheduler_patience = 8
        
        # Checkpoint parameters
        self.checkpoint_dir = "checkpoints"
        self.save_best_model_only = True
        
        self.random_seed = 42
        
    def to_dict(self):
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}