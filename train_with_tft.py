import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch import GradScaler
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
from sklearn.metrics import r2_score, mean_squared_error
import json
import logging
from datetime import datetime
from pathlib import Path

# Import models - UPDATE THIS LINE to use your new architecture file
from architecture_with_tft import TimeSeriesLSTM, AttentionLSTM, TemporalFusionTransformer
from dataset import make_dataloaders
from config_with_tft import TrainingConfig

TIMESTAMP = 'no_time'

# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logging(checkpoint_dir: str) -> logging.Logger:
    """Setup logging to file and console"""
    global TIMESTAMP
    TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger('train')
    logger.setLevel(logging.DEBUG)
    
    # File handler
    fh = logging.FileHandler(f'{checkpoint_dir}/train_{TIMESTAMP}.log')
    fh.setLevel(logging.DEBUG)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger

# ============================================================================
# EARLY STOPPING CLASS
# ============================================================================

class EarlyStopping:
    """Early stopping to prevent overfitting"""
    def __init__(self, patience: int = 10, min_delta: float = 0.0,
                 model_save_path: str = None, verbose: bool = True):
        """
        Args:
            patience: Number of epochs with no improvement after which training will be stopped
            min_delta: Minimum change in monitored value to qualify as improvement
            model_save_path: Path to save the best model
            verbose: Whether to print messages
        """
        self.patience = patience
        self.min_delta = min_delta
        self.model_save_path = model_save_path
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_epoch = 0
    
    def __call__(self, val_loss: float, model: nn.Module, epoch: int):
        """Check if early stopping condition is met"""
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_epoch = epoch
            
            # Save best model
            if self.model_save_path:
                torch.save(model.state_dict(), self.model_save_path)
                if self.verbose:
                    print(f"✓ Model improved. Checkpoint saved to {self.model_save_path}")
        else:
            self.counter += 1
            if self.verbose:
                print(f"  No improvement for {self.counter}/{self.patience} epochs")
            
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f"Early stopping triggered after {self.counter} epochs without improvement")

# ============================================================================
# METRICS CALCULATION
# ============================================================================

def calculate_metrics(predictions: np.ndarray, actuals: np.ndarray) -> dict:
    """Calculate comprehensive metrics"""
    mse = mean_squared_error(actuals, predictions)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions - actuals))
    r2 = r2_score(actuals, predictions)
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }

def validate(model: nn.Module, val_loader: DataLoader,
             criterion: nn.Module, device: torch.device) -> tuple:
    """
    Validate the model on validation set
    Returns: (val_loss, metrics_dict)
    """
    model.eval()
    val_loss = 0.0
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            y = y.to(device)
            
            output = model(x)
            loss = criterion(output, y)
            val_loss += loss.item()
            
            predictions.extend(output.cpu().numpy())
            actuals.extend(y.cpu().numpy())
    
    avg_val_loss = val_loss / len(val_loader)
    
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    metrics = calculate_metrics(predictions, actuals)
    
    return avg_val_loss, metrics

# ============================================================================
# TRAINING LOOP
# ============================================================================

def train(device: torch.device, input_size: int, train_loader: DataLoader,
          val_loader: DataLoader, config: TrainingConfig, logger=None) -> dict:
    """
    Robust training loop with early stopping and validation
    
    Returns:
        training_history: Dictionary containing loss and metrics history
    """
    global TIMESTAMP
    
    if logger:
        logger.info(f"Using device: {device}")
        logger.info(f"Model type: {config.model_type}")
    else:
        print(f"Using device: {device}")
        print(f"Model type: {config.model_type}")
    
    # Create checkpoint directory
    Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    # Initialize model based on config
    if config.model_type == 'lstm':
        model = TimeSeriesLSTM(
            input_size=input_size,
            hidden_size=config.hidden_size,
            fc_size=config.fc_size,
            num_layers=config.num_layers,
            output_size=input_size
        ).to(device)
        
    elif config.model_type == 'attention':
        model = AttentionLSTM(
            input_size=input_size,
            hidden_size=config.hidden_size,
            fc_size=config.fc_size,
            num_layers=config.num_layers,
            output_size=input_size,
            lstm_dropout=config.lstm_dropout,
            attention_dropout=config.attention_dropout,
            fc_dropout=config.fc_dropout
        ).to(device)
        
    elif config.model_type == 'tft':
        model = TemporalFusionTransformer(
            input_size=input_size,
            hidden_size=config.hidden_size,
            num_heads=config.num_heads,
            num_lstm_layers=config.num_lstm_layers,
            dropout=config.dropout,
            output_size=input_size
        ).to(device)
        
    else:
        raise ValueError(f"Unknown model type: {config.model_type}")
        
    # Log model architecture
    if logger:
        logger.info(f"Model architecture:\n{model}")
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=config.lr_scheduler_factor,
        patience=config.lr_scheduler_patience
    )
    
    # Early stopping
    early_stopping = EarlyStopping(
        patience=config.early_stopping_patience,
        min_delta=config.early_stopping_min_delta,
        model_save_path=f'{config.checkpoint_dir}/model_{config.model_type}_{TIMESTAMP}.pth',
        verbose=True
    )

    # Mixed precision scaler
    scaler = GradScaler() if device.type == 'cuda' else None
    if scaler and logger:
        logger.info("Mixed precision training enabled")
    
    # Training history
    history = {
        'epoch': [],
        'train_loss': [],
        'val_loss': [],
        'val_rmse': [],
        'val_mae': [],
        'val_r2': [],
        'learning_rate': []
    }
    
    # Log configuration
    if logger:
        logger.info("Training Configuration:")
        logger.info(json.dumps(config.to_dict(), indent=2))
    
    # Training loop
    for epoch in range(config.num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        train_bar = tqdm(enumerate(train_loader), total=len(train_loader),
                        desc=f"Epoch {epoch+1}/{config.num_epochs} [Train]")

        # In training loop, replace the forward/backward pass:
        # for batch_idx, (x, y) in train_bar:
        #     x = x.to(device)
        #     y = y.to(device)
            
        #     optimizer.zero_grad()
            
        #     # Use autocast for mixed precision
        #     if scaler is not None:
        #         with autocast(device_type='cuda'):
        #             output = model(x)
        #             loss = criterion(output, y)
                
        #         scaler.scale(loss).backward()
        #         scaler.unscale_(optimizer)
        #         torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        #         scaler.step(optimizer)
        #         scaler.update()
        #     else:
        #         output = model(x)
        #         loss = criterion(output, y)
        #         loss.backward()
        #         torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        #         optimizer.step()

        #     train_loss += loss.item()
        #     train_bar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        for batch_idx, (x, y) in train_bar:
            x = x.to(device)
            y = y.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            output = model(x)
            
            # Compute loss (with regularization for attention model)
            if config.model_type == 'attention':
                mse_loss = criterion(output, y)
                attn_reg = model.get_attention_regularization(
                    l2_weight=config.attention_l2_weight
                )
                loss = mse_loss + attn_reg
            else:
                loss = criterion(output, y)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            train_bar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        avg_train_loss = train_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)
        
        # Validation phase
        avg_val_loss, metrics = validate(model, val_loader, criterion, device)
        history['val_loss'].append(avg_val_loss)
        
        # Store metrics
        history['val_rmse'].append(metrics['rmse'])
        history['val_mae'].append(metrics['mae'])
        history['val_r2'].append(metrics['r2'])
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        history['learning_rate'].append(current_lr)
        history['epoch'].append(epoch + 1)
        
        # Print progress
        metrics_str = f" | Val R²: {metrics['r2']:.4f} | Val RMSE: {metrics['rmse']:.6f}"
        print(f"Epoch {epoch+1}/{config.num_epochs} | "
              f"Train Loss: {avg_train_loss:.6f} | "
              f"Val Loss: {avg_val_loss:.6f}{metrics_str}")
        
        if logger:
            logger.info(f"Epoch {epoch+1} - Train Loss: {avg_train_loss:.6f}, "
                       f"Val Loss: {avg_val_loss:.6f}{metrics_str}")
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        # Early stopping check
        early_stopping(avg_val_loss, model, epoch)
        if early_stopping.early_stop:
            if logger:
                logger.info(f"Training stopped at epoch {epoch+1}")
            break
    
    # Summary
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"Model type: {config.model_type.upper()}")
    print(f"Total epochs trained: {len(history['train_loss'])}")
    print(f"Best epoch: {early_stopping.best_epoch + 1}")
    print(f"Best validation loss: {early_stopping.best_loss:.6f}")
    print(f"Final training loss: {history['train_loss'][-1]:.6f}")
    print(f"Final validation loss: {history['val_loss'][-1]:.6f}")
    
    if history['val_r2']:
        best_r2_idx = np.argmax(history['val_r2'])
        print(f"Best validation R²: {history['val_r2'][best_r2_idx]:.4f} (epoch {best_r2_idx+1})")
    
    if logger:
        logger.info(f"Total epochs trained: {len(history['train_loss'])}")
        logger.info(f"Best epoch: {early_stopping.best_epoch + 1}")
        logger.info(f"Best validation loss: {early_stopping.best_loss:.6f}")
    
    return model, history

# ============================================================================
# EVALUATION
# ============================================================================

def evaluate(model: nn.Module, test_loader: DataLoader,
             device: torch.device, logger=None) -> dict:
    """Generate predictions and evaluate on test set"""
    model.eval()
    predictions = []
    actuals = []
    
    print("\nEvaluating on test set...")
    with torch.no_grad():
        for x, y in tqdm(test_loader, desc="Evaluation"):
            x = x.to(device)
            output = model(x)
            
            predictions.extend(output.cpu().numpy())
            actuals.extend(y.cpu().numpy())
    
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    # Calculate metrics
    metrics = calculate_metrics(predictions, actuals)
    
    # Print results
    print("\n" + "="*70)
    print("TEST SET EVALUATION")
    print("="*70)
    print(f"MSE: {metrics['mse']:.6f}")
    print(f"RMSE: {metrics['rmse']:.6f}")
    print(f"MAE: {metrics['mae']:.6f}")
    print(f"R²: {metrics['r2']:.6f}")
    
    if logger:
        logger.info("TEST SET EVALUATION")
        logger.info(f"MSE: {metrics['mse']:.6f}")
        logger.info(f"RMSE: {metrics['rmse']:.6f}")
        logger.info(f"MAE: {metrics['mae']:.6f}")
        logger.info(f"R²: {metrics['r2']:.6f}")
    
    return predictions, actuals, metrics

# ============================================================================
# MAIN
# ============================================================================

def main():
    # Setup - CHANGE model_type HERE: 'lstm', 'attention', or 'tft'
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # SELECT YOUR MODEL TYPE
    MODEL_TYPE = 'tft'  # Change to 'lstm', 'attention', or 'tft'
    
    config = TrainingConfig(model_type=MODEL_TYPE)
    logger = setup_logging(config.checkpoint_dir)
    
    # Device setup
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    
    logger.info("="*70)
    logger.info(f"STARTING TRAINING PIPELINE - {MODEL_TYPE.upper()} MODEL")
    logger.info("="*70)
    
    # Load data
    logger.info("Loading data...")
    data = pd.read_parquet(f'{CURRENT_DIR}/data/train.parquet')
    
    # Create dataloaders
    logger.info("Creating dataloaders...")
    train_loader, val_loader, test_loader = make_dataloaders(
        data, config.window_size, config.batch_size
    )
    logger.info(f"Train batches: {len(train_loader)}")
    logger.info(f"Val batches: {len(val_loader)}")
    logger.info(f"Test batches: {len(test_loader)}")
    
    # Train model
    logger.info("Starting training...")
    model, history = train(
        device=device,
        input_size=32,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        logger=logger
    )
    
    # Evaluate on test set
    logger.info("Evaluating model...")
    predictions, actuals, metrics = evaluate(model, test_loader, device, logger)
    
    logger.info("="*70)
    logger.info("TRAINING PIPELINE COMPLETE")
    logger.info("="*70)

if __name__ == "__main__":
    main()