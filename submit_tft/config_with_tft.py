class TrainingConfig:
    """Centralized configuration for training"""
    def __init__(self, model_type='lstm'):
        """
        Args:
            model_type: One of 'lstm', 'attention', or 'tft'
        """
        self.model_type = model_type  # 'lstm', 'attention', or 'tft'
        
        # Common parameters
        self.input_size = 32
        self.num_epochs = 100
        self.batch_size = 256
        self.window_size = 100
        self.checkpoint_dir = "checkpoints"
        self.save_best_model_only = True
        self.random_seed = 42
        
        # Early stopping parameters
        self.early_stopping_patience = 10
        self.early_stopping_min_delta = 1e-5
        
        # Learning rate scheduler parameters
        self.lr_scheduler_factor = 0.5
        self.lr_scheduler_patience = 5
        
        # Model-specific configurations
        if model_type == 'lstm':
            self.hidden_size = 64
            self.fc_size = 32
            self.num_layers = 1
            self.learning_rate = 0.001
            self.model_path = 'checkpoints/best_lstm_model.pth'
            
        elif model_type == 'attention':
            self.hidden_size = 64
            self.fc_size = 64
            self.num_layers = 1
            self.learning_rate = 0.001
            self.model_path = 'checkpoints/best_attention_model.pth'
            self.early_stopping_patience = 15
            self.lr_scheduler_patience = 7
            
            # Attention-specific dropout
            self.lstm_dropout = 0.2
            self.attention_dropout = 0.3
            self.fc_dropout = 0.2
            self.attention_l2_weight = 0.001
            
        elif model_type == 'tft':
            # TFT-specific parameters
            self.hidden_size = 32
            self.num_heads = 2
            self.num_lstm_layers = 1
            self.dropout = 0.1
            self.learning_rate = 0.0005  # Lower learning rate for TFT
            self.model_path = 'checkpoints/model_tft_20251119_171731.pth'
            self.early_stopping_patience = 20  # TFT needs more patience
            self.lr_scheduler_patience = 8
            
        else:
            raise ValueError(f"Unknown model_type: {model_type}. Choose 'lstm', 'attention', or 'tft'")
    
    def to_dict(self):
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
